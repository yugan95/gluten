/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.execution.exchange

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.columnarbatch.ColumnarBatches
import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.execution.{ColumnarToRowExecBase, RowToVeloxColumnarExec, WholeStageTransformer}

import org.apache.spark.{shard, SparkContext, SparkEnv, SparkException}
import org.apache.spark.internal.config.DYN_ALLOCATION_MAX_EXECUTORS
import org.apache.spark.rdd.RDD
import org.apache.spark.shard.GlutenShardManagerJni
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning, ShardPartitioning}
import org.apache.spark.sql.execution.{ColumnarCollapseTransformStages, ColumnarInputAdapter, InputIteratorTransformer, RowToColumnarTransition, SparkPlan, SQLExecution}
import org.apache.spark.sql.execution.metric.SQLMetrics
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{GlutenBlockManagerBridge, ShardBlockId}
import org.apache.spark.util.{ThreadUtils, Utils}
import org.apache.spark.util.io.ChunkedByteBuffer

import java.nio.ByteBuffer
import java.util.UUID
import java.util.concurrent.{Future => JFuture, TimeUnit}

import scala.concurrent.{ExecutionContext, Promise, TimeoutException}

/**
 * Gluten-specific implementation of ShardExchangeExec for native Velox backend.
 *
 * Unlike the JVM-based ShardExchangeExec which serializes HashedRelation using Java serialization,
 * this implementation writes Velox-compatible Presto-serialized RowVector data to BlockManager,
 * enabling direct consumption by Velox's VeloxShardManager.
 *
 * Key differences from JVM ShardExchangeExec:
 *   1. Uses ColumnarBatch (columnar format) instead of InternalRow (row format) 2. Writes
 *      Presto-serialized RowVector as "piece0", "piece1", ... blocks (4MB each) 3. Writes metadata
 *      (pieceCount + numKeyColumns + checksums) to "meta" block 4. Writes Velox BloomFilter to
 *      "bloom" block 5. Calls JNI to load shard data into Velox VeloxShardManager's in-memory
 *      HashTable
 */
case class ColumnarShardExchangeExec(
    buildBoundKeys: Seq[Expression],
    numShards: Int,
    replicaCount: Int,
    child: SparkPlan)
  extends ShardExchangeLike {

  override val runId: UUID = UUID.randomUUID

  override lazy val metrics = Map(
    "dataSize" -> SQLMetrics.createSizeMetric(sparkContext, "data size"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"))

  override def outputPartitioning: Partitioning = ShardPartitioning(buildBoundKeys, numShards)

  override protected def doCanonicalize(): SparkPlan = {
    copy(buildBoundKeys = buildBoundKeys.map(_.canonicalized), child = child.canonicalized)
  }

  override def runtimeStatistics: Statistics = {
    val dataSize = metrics("dataSize").value
    val rowCount = metrics("numOutputRows").value
    Statistics(dataSize, Some(rowCount))
  }

  @transient
  private lazy val promise = Promise[shard.ShardSetRef]()

  @transient
  override lazy val completionFuture: scala.concurrent.Future[shard.ShardSetRef] =
    promise.future

  @transient
  private val timeout: Long = 30 // minutes

  @transient
  override lazy val relationFuture: JFuture[shard.ShardSetRef] =
    SQLExecution
      .withThreadLocalCaptured[shard.ShardSetRef](
        session,
        ColumnarShardExchangeExec.executionContext) {
        try {
          sparkContext.setJobGroup(
            runId.toString,
            s"gluten-shard exchange (runId $runId)",
            interruptOnCancel = true)
          if (!ensureExecutors(15, TimeUnit.MINUTES)) {
            logWarning(
              s"ensureExecutors timed out or SC stopped;" +
                s" active executors may be < target, continue build")
          }

          val setId = sparkContext.env.shardManager.newShardSet(numShards, replicaCount)

          // Handle all possible child shapes defensively:
          //   - ColumnarToRowExecBase: strip to recover columnar subtree
          //   - RowToColumnarTransition (Spark vanilla): replace with Gluten's
          //     RowToVeloxColumnarExec for Arrow-compatible format
          //   - Already columnar (supportsColumnar=true): use as-is
          //   - Row-based (e.g. LocalTableScan): wrap with RowToVeloxColumnarExec
          val rawColumnarChild = child match {
            case c2r: ColumnarToRowExecBase => c2r.child
            case r2c: RowToColumnarTransition =>
              RowToVeloxColumnarExec(r2c.child)
            case plan if plan.supportsColumnar => plan
            case rowPlan => RowToVeloxColumnarExec(rowPlan)
          }
          val columnarChild = stripCollapseWrappers(rawColumnarChild)

          // Use Gluten's columnar shuffle to repartition data by join keys.
          // genColumnarShuffleExchange creates ProjectExecTransformer nodes that
          // require WholeStageTransformer wrapping to be executable. Apply
          // ColumnarCollapseTransformStages to insert the necessary wrappers.
          val vanillaShuffle = ShuffleExchangeExec(
            HashPartitioning(buildBoundKeys, numShards),
            columnarChild,
            REPARTITION_BY_NUM)
          val columnarShuffle = BackendsApiManager.getSparkPlanExecApiInstance
            .genColumnarShuffleExchange(vanillaShuffle)
          val collapseRule = ColumnarCollapseTransformStages(GlutenConfig.get)
          val executableShuffle = collapseRule.apply(columnarShuffle)
          val shuffledColumnar = executableShuffle.executeColumnar()
          val sharded = new ShardedColumnarBatchRDD(
            shuffledColumnar,
            getPreferredHosts(shuffledColumnar.partitions.length))

          val numKeyColumns = buildBoundKeys.length
          val blockSize = 4 << 20 // 4MB
          val shardIds = sharded
            .mapPartitionsWithIndex {
              (shardId, batchIter) =>
                val bm = SparkEnv.get.blockManager
                val backendName = BackendsApiManager.getBackendName

                // Scale BF capacity by numShards so that the merged BF (bitwise OR
                // of all per-shard BFs) maintains the target FPR.  Each shard's BF
                // is built with this capacity; after merge the combined BF has the
                // same bit-array size, keeping load factor ≈ single-shard level.
                val bloomCapacity = (5L << 20) * numShards
                val builderHandle = GlutenShardManagerJni.createShardBuilder(
                  numKeyColumns,
                  blockSize,
                  bloomCapacity)
                try {
                  var pieceIndex = 0

                  var totalBatchRows = 0L
                  batchIter.foreach {
                    batch =>
                      totalBatchRows += batch.numRows()
                      longMetric("numOutputRows").add(batch.numRows())
                      val handle = ColumnarBatches.getNativeHandle(backendName, batch)
                      val flushedPiece =
                        GlutenShardManagerJni.appendBatchToShardBuilder(builderHandle, handle)
                      batch.close()

                      if (flushedPiece != null) {
                        longMetric("dataSize").add(flushedPiece.length)
                        val pieceId = ShardBlockId(setId, shardId, "piece" + pieceIndex)
                        bm.putBytes(
                          pieceId,
                          new ChunkedByteBuffer(ByteBuffer.wrap(flushedPiece)),
                          org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER,
                          tellMaster = true)
                        pieceIndex += 1
                      }
                  }

                  val finishResult = GlutenShardManagerJni.finishShardBuilder(builderHandle)
                  val lastPieceBytes = finishResult(0)
                  val bloomBytes = finishResult(1)
                  val checksumBytes = finishResult(2)

                  if (lastPieceBytes != null && lastPieceBytes.nonEmpty) {
                    longMetric("dataSize").add(lastPieceBytes.length)
                    val pieceId = ShardBlockId(setId, shardId, "piece" + pieceIndex)
                    bm.putBytes(
                      pieceId,
                      new ChunkedByteBuffer(ByteBuffer.wrap(lastPieceBytes)),
                      org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER,
                      tellMaster = true)
                    pieceIndex += 1
                  }

                  // Build Hive-compatible schema string for C++ HiveTypeParser.
                  // child.output gives the build side's column schema.
                  val schemaStr = child.schema.catalogString
                  val schemaBytes = schemaStr.getBytes(java.nio.charset.StandardCharsets.UTF_8)

                  // Build hash key schema from buildBoundKeys result types.
                  // This tells C++ what types to use when computing shard hash
                  // (e.g., cast(k as bigint) means hash key type is bigint,
                  // while plain k means hash key type is int).
                  import org.apache.spark.sql.types.StructType
                  val hashKeySchema = StructType(buildBoundKeys.zipWithIndex.map {
                    case (expr, i) =>
                      org.apache.spark.sql.types.StructField(s"hk$i", expr.dataType)
                  })
                  val hashKeySchemaStr = hashKeySchema.catalogString
                  val hashKeySchemaBytes =
                    hashKeySchemaStr.getBytes(java.nio.charset.StandardCharsets.UTF_8)

                  val metaBuf = ByteBuffer.allocate(
                    16 + schemaBytes.length + hashKeySchemaBytes.length + checksumBytes.length)
                  metaBuf.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                  metaBuf.putInt(pieceIndex)
                  metaBuf.putInt(numKeyColumns)
                  metaBuf.putInt(schemaBytes.length)
                  metaBuf.put(schemaBytes)
                  metaBuf.putInt(hashKeySchemaBytes.length)
                  metaBuf.put(hashKeySchemaBytes)
                  if (checksumBytes.nonEmpty) {
                    metaBuf.put(checksumBytes)
                  }
                  val metaId = ShardBlockId(setId, shardId, "meta")
                  bm.putBytes(
                    metaId,
                    new ChunkedByteBuffer(ByteBuffer.wrap(metaBuf.array())),
                    org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER,
                    tellMaster = true)

                  if (bloomBytes != null && bloomBytes.nonEmpty) {
                    // Store Velox-format bloom filter as "nativeBloom" for C++ ShardSource.
                    val nativeBloomId = ShardBlockId(setId, shardId, "nativeBloom")
                    bm.putBytes(
                      nativeBloomId,
                      new ChunkedByteBuffer(ByteBuffer.wrap(bloomBytes)),
                      org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER,
                      tellMaster = true)
                  }

                  // Always write a Spark-compatible BloomFilter to "bloom" block.
                  // When non-inner join types fall back to vanilla Spark's
                  // DistributedMapJoinExec, it reads "bloom" blocks and parses
                  // them with BloomFilterImpl.  We write an empty (pass-all)
                  // Spark BloomFilter to avoid format mismatch errors.
                  {
                    import org.apache.spark.util.sketch.BloomFilter
                    val sparkBf = BloomFilter.create(1L, 0.01)
                    val baos = new java.io.ByteArrayOutputStream()
                    sparkBf.writeTo(baos)
                    val sparkBloomBytes = baos.toByteArray
                    val bloomId = ShardBlockId(setId, shardId, "bloom")
                    bm.putBytes(
                      bloomId,
                      new ChunkedByteBuffer(ByteBuffer.wrap(sparkBloomBytes)),
                      org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER,
                      tellMaster = true)
                  }

                  // Install this shard replica locally via ShardManager.installReplica,
                  // which delegates to GlutenShardLookupService.installNativeReplica
                  // (i.e., GlutenShardManagerJni.constructShardTable) when
                  // supportsNativeLookup=true, then reports the shard location to
                  // ShardManagerMaster so probe-side can discover it via getLocations().
                  SparkEnv.get.shardManager.installReplica(setId, shardId)

                  // Also request other executors to install replicas (non-blocking).
                  // ShardManagerMasterEndpoint broadcasts InstallReplica to target
                  // executors, which also call installReplica with the native path.
                  SparkEnv.get.shardManager.installReplicaSet(setId, shardId)

                } finally {
                  GlutenShardManagerJni.destroyShardBuilder(builderHandle)
                }

                Iterator.single(shardId)
            }
            .collect()

          // Merge Velox-format bloom filters across all shards using streaming merge.
          // Read from "nativeBloom" blocks (Velox format) -- not "bloom" (Spark format).
          val bm = SparkEnv.get.blockManager
          val bmBridge = new GlutenBlockManagerBridge(bm)
          try {
            val mergerHandle = GlutenShardManagerJni.createBloomFilterMerger()
            try {
              var mergedCount = 0
              shardIds.foreach {
                sid =>
                  val bloomBytes = bmBridge.readBlock(setId, sid, "nativeBloom")
                  if (bloomBytes != null && bloomBytes.nonEmpty) {
                    GlutenShardManagerJni.mergeBloomFilterChunk(mergerHandle, bloomBytes)
                    mergedCount += 1
                  }
              }
              if (mergedCount > 0) {
                val mergedBloomBytes = GlutenShardManagerJni.finishBloomFilterMerger(mergerHandle)
                if (mergedBloomBytes.nonEmpty) {
                  val mergedBloomBlockId = ShardBlockId(setId, -1, "nativeBloom")
                  bm.putBytes(
                    mergedBloomBlockId,
                    new ChunkedByteBuffer(ByteBuffer.wrap(mergedBloomBytes)),
                    org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER,
                    tellMaster = true)
                }
              } else {
                GlutenShardManagerJni.finishBloomFilterMerger(mergerHandle)
              }
            } catch {
              case e: Exception =>
                try { GlutenShardManagerJni.finishBloomFilterMerger(mergerHandle) }
                catch { case _: Exception => () }
                throw e
            }
          } catch {
            case e: Exception =>
              logWarning(
                s"Failed to merge bloom filters for shardSet $setId, " +
                  s"probe-side will skip BF filtering.",
                e)
          }
          val setRef = shard.ShardSetRef(setId, shardIds)
          promise.trySuccess(setRef)
          sparkContext.cleaner.foreach(_.registerShardSetForCleanup(setRef))
          setRef
        } catch {
          case e: Throwable =>
            promise.tryFailure(e)
            throw e
        }
      }

  override protected def doPrepare(): Unit = {
    metrics
    relationFuture
  }

  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException("ColumnarShardExchange does not support row execution")

  override def doExecuteColumnar(): RDD[ColumnarBatch] =
    throw new UnsupportedOperationException(
      "ColumnarShardExchange does not support columnar execution")

  def buildShardSet(): shard.ShardSetRef = try {
    relationFuture.get(timeout, TimeUnit.MINUTES)
  } catch {
    case ex: TimeoutException =>
      logError(s"Could not execute shard in $timeout minutes.", ex)
      if (!relationFuture.isDone) {
        sparkContext.cancelJobGroup(runId.toString)
        relationFuture.cancel(true)
      }
      throw new SparkException(s"shard exchange timeout.", ex)
  }

  override protected def withNewChildInternal(newChild: SparkPlan): ColumnarShardExchangeExec =
    copy(child = newChild)

  /**
   * Build preferred locations at executor granularity using ExecutorCacheTaskLocation.
   *
   * The original implementation only used host-level locations, which meant Spark's
   * scheduler could assign multiple shards to the same executor on a host. By using
   * ExecutorCacheTaskLocation (format: "executor_host_executorId"), Spark will prefer
   * PROCESS_LOCAL scheduling, distributing shards evenly across executors.
   */
  private def getPreferredHosts(prevPartLen: Int): Array[Seq[String]] = {
    import org.apache.spark.scheduler.ExecutorCacheTaskLocation

    val driverId = SparkContext.DRIVER_IDENTIFIER
    val executorLocations: Array[String] = try {
      val memStatus = SparkEnv.get.blockManager.master.getMemoryStatus
      memStatus.keys.iterator
        .filter(_.executorId != driverId)
        .map(bmId => ExecutorCacheTaskLocation(bmId.host, bmId.executorId).toString)
        .toArray
    } catch {
      case scala.util.control.NonFatal(e) =>
        logWarning("Failed to get executor locations from BlockManagerMaster, " +
          "falling back to host-level locations", e)
        val driverHost = sparkContext.conf.get("spark.driver.host", "")
        sparkContext.statusTracker.getExecutorInfos.iterator
          .map(_.host)
          .filter(_ != driverHost)
          .toArray
          .distinct
    }

    val shuffled = scala.util.Random.shuffle(executorLocations.toSeq).toArray
    logInfo(s"getPreferredHosts: ${shuffled.length} executors for $prevPartLen partitions: " +
      s"${shuffled.mkString(", ")}")
    Array.tabulate(prevPartLen) {
      i => if (shuffled.nonEmpty) Seq(shuffled(i % shuffled.length)) else Nil
    }
  }

  private def ensureExecutors(length: Long, unit: TimeUnit): Boolean = {
    val sparkConf = sparkContext.getConf
    if (Utils.isDynamicAllocationEnabled(sparkConf)) {
      val minExecutors =
        math.min(numShards, sparkConf.get(DYN_ALLOCATION_MAX_EXECUTORS))

      val p = Promise[Boolean]()
      try sparkContext.requestTotalExecutors(minExecutors, 0, Map.empty)
      catch { case scala.util.control.NonFatal(_) => () }

      def activeExecutorCount: Int = {
        val ids =
          try sparkContext.getExecutorIds()
          catch { case scala.util.control.NonFatal(_) => Seq.empty[String] }
        ids.size
      }

      def distinctHostCount: Int = {
        try {
          val driverId = SparkContext.DRIVER_IDENTIFIER
          SparkEnv.get.blockManager.master.getMemoryStatus.keys.iterator
            .filter(_.executorId != driverId)
            .map(_.host)
            .toSet
            .size
        } catch {
          case scala.util.control.NonFatal(_) => 0
        }
      }

      val scheduler =
        ThreadUtils.newDaemonSingleThreadScheduledExecutor("gluten-shard-executor-scheduler")

      def shutdown(): Unit = {
        try scheduler.shutdown()
        catch { case _: Throwable => () }
      }

      val timeoutMs = unit.toMillis(length)
      val start = System.nanoTime()
      val poll = new Runnable {
        override def run(): Unit = {
          try {
            if (sparkContext.isStopped) {
              p.trySuccess(false); shutdown()
              return
            }
            val numExecutors = activeExecutorCount
            if (numExecutors == 0) {
              logWarning(
                "Active executors should not be 0, " +
                  "spark.dynamicAllocation.minExecutors should be greater than 0.")
              p.trySuccess(false); shutdown()
              return
            }
            val numHosts = distinctHostCount
            val elapsedMs = (System.nanoTime() - start) / 1000000L
            logInfo(s"ensureExecutors: $numExecutors executors on $numHosts hosts " +
              s"(target: $minExecutors executors), elapsed ${elapsedMs}ms")
            if (numExecutors >= minExecutors) {
              p.trySuccess(true); shutdown()
              return
            }
            if (elapsedMs >= timeoutMs) {
              logWarning(s"ensureExecutors timed out after ${elapsedMs}ms: " +
                s"$numExecutors/$minExecutors executors on $numHosts hosts")
              p.trySuccess(false); shutdown()
            }
          } catch {
            case scala.util.control.NonFatal(_) =>
          }
        }
      }

      scheduler.scheduleWithFixedDelay(poll, 0L, 10, TimeUnit.SECONDS)
      ThreadUtils.awaitResult(p.future, scala.concurrent.duration.Duration.apply(length + 1, unit))
    } else true
  }

  /**
   * Recursively strip WholeStageTransformer, InputIteratorTransformer, and
   * ColumnarInputAdapter wrappers that AQE's postStageCreationRules may have
   * already inserted.  This prevents collapseRule.apply from producing nested
   * WholeStageTransformers which would fail with "This operator doesn't support
   * doTransform with SubstraitContext".
   */
  private def stripCollapseWrappers(plan: SparkPlan): SparkPlan = plan match {
    case WholeStageTransformer(child, _) => stripCollapseWrappers(child)
    case InputIteratorTransformer(child) => stripCollapseWrappers(child)
    case ColumnarInputAdapter(child) => stripCollapseWrappers(child)
    case other => other.withNewChildren(other.children.map(stripCollapseWrappers))
  }
}

object ColumnarShardExchangeExec {
  private[execution] val executionContext = ExecutionContext.fromExecutorService(
    ThreadUtils.newDaemonCachedThreadPool("columnar-shard-exchange", 8))
}
