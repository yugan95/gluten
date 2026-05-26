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
package org.apache.spark.sql.execution.joins

import org.apache.gluten.execution.{JoinUtils, TransformContext, TransformSupport}
import org.apache.gluten.expression.ConverterUtils
import org.apache.gluten.extension.ValidationResult
import org.apache.gluten.metrics.MetricsUpdater
import org.apache.gluten.substrait.`type`.ColumnTypeNode
import org.apache.gluten.substrait.SubstraitContext
import org.apache.gluten.substrait.rel.RelBuilder
import org.apache.gluten.utils.SubstraitUtil

import org.apache.spark.SparkEnv
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide}
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.logical.{DistMapJoinStrategy, JoinHint}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{BinaryExecNode, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.adaptive.ShardQueryStageExec
import org.apache.spark.sql.execution.exchange.{ColumnarShardExchangeExec, ReusedExchangeExec, ShardExchangeExec}
import org.apache.spark.sql.execution.metric.SQLMetrics
import org.apache.spark.sql.vectorized.ColumnarBatch

import com.google.protobuf.{Any, StringValue}
import io.substrait.proto.JoinRel

import scala.annotation.tailrec
import scala.collection.JavaConverters._

/** Transformer for DistributedMapJoinExec to offload to Velox native backend. */
case class DistributedMapJoinExecTransformer(
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    joinType: JoinType,
    buildSide: BuildSide,
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan,
    hint: JoinHint,
    isNullAwareAntiJoin: Boolean = false)
  extends BinaryExecNode
  with TransformSupport {

  override def output: Seq[Attribute] = {
    joinType match {
      case _: InnerLike =>
        left.output ++ right.output
      case LeftOuter =>
        left.output ++ right.output.map(_.withNullability(true))
      case RightOuter =>
        left.output.map(_.withNullability(true)) ++ right.output
      case FullOuter =>
        left.output.map(_.withNullability(true)) ++ right.output.map(_.withNullability(true))
      case LeftSemi | LeftAnti =>
        left.output
      case j: ExistenceJoin =>
        left.output :+ j.exists
      case x =>
        throw new UnsupportedOperationException(s"JoinType $x not supported")
    }
  }

  override protected def withNewChildrenInternal(
      newLeft: SparkPlan,
      newRight: SparkPlan): SparkPlan =
    copy(left = newLeft, right = newRight)

  override def outputPartitioning: Partitioning = streamedPlan.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = streamedPlan.outputOrdering

  @transient override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"))

  private lazy val buildPlan: SparkPlan = buildSide match {
    case BuildLeft => left
    case BuildRight => right
  }

  private lazy val streamedPlan: SparkPlan = buildSide match {
    case BuildLeft => right
    case BuildRight => left
  }

  private lazy val buildKeyExprs: Seq[Expression] = buildSide match {
    case BuildLeft => leftKeys
    case BuildRight => rightKeys
  }

  private lazy val streamedKeyExprs: Seq[Expression] = buildSide match {
    case BuildLeft => rightKeys
    case BuildRight => leftKeys
  }

  private lazy val (numShards, replicaCount): (Int, Int) = {
    val strategy =
      (if (buildSide == BuildLeft) hint.leftHint else hint.rightHint).flatMap(_.strategy)
    strategy match {
      case Some(DistMapJoinStrategy(ns, rc)) => (ns.getOrElse(5), rc.getOrElse(1))
      case _ => (5, 1)
    }
  }

  protected lazy val substraitJoinType: JoinRel.JoinType = SubstraitUtil.toSubstrait(joinType)

  override def columnarInputRDDs: Seq[RDD[ColumnarBatch]] = {
    getColumnarInputRDDs(streamedPlan)
  }

  override def metricsUpdater(): MetricsUpdater = MetricsUpdater.Todo

  override protected def doValidateInternal(): ValidationResult = {
    if (substraitJoinType == JoinRel.JoinType.UNRECOGNIZED) {
      return ValidationResult
        .failed(s"Unsupported join type $joinType for DistributedMapJoin")
    }
    joinType match {
      case _: InnerLike | LeftOuter =>
        ValidationResult.succeeded
      case LeftSemi | LeftAnti =>
        ValidationResult.failed(
          "Semi/Anti join not yet supported in native DistributedMapJoin path")
      case _ =>
        ValidationResult.failed(
          s"Join type $joinType not supported in native DistributedMapJoin path")
    }
  }

  override protected def doTransform(context: SubstraitContext): TransformContext = {
    val streamedPlanContext = streamedPlan.asInstanceOf[TransformSupport].transform(context)
    val inputStreamedRelNode = streamedPlanContext.root
    val inputStreamedOutput = streamedPlanContext.outputAttributes

    val shardSetRef = resolveShardSetRef(buildPlan)
    val shardSetId = shardSetRef.setId

    // Build per-shard location map: shardId -> list of gRPC addresses.
    // Mirrors Spark's dynamic routing via master.getLocations and
    // preserves the shardId->location mapping for C++ routing.
    val shardLocationMap: Map[Int, Seq[String]] = shardSetRef.shardIds.map {
      shardId =>
        val locs = SparkEnv.get.shardManager.master
          .getLocations(shardSetId, shardId)
          .map(loc => s"${loc.host}:${loc.port}")
        shardId -> locs
    }.toMap

    // Debug logging for shard lookup troubleshooting
    val locMapStr =
      shardLocationMap.map { case (k, v) => s"$k->[${v.mkString(";")}]" }.mkString(",")
    logInfo(
      s"DMJ doTransform: shardSetId=$shardSetId, " +
        s"shardIds=${shardSetRef.shardIds.mkString(",")}, " +
        s"shardLocationMap=$locMapStr, " +
        s"numShards=$numShards, " +
        s"joinType=$joinType, buildSide=$buildSide")

    val operatorId = context.nextOperatorId(this.nodeName)

    // Build side data is served by the native shard server, but the Substrait
    // JoinRel still requires a right RelNode for schema inference.  Provide a
    // schema-only ReadRel (no input iterator) that describes the build output.
    val buildAttributes = buildPlan.output
    val buildTypeList = ConverterUtils.collectAttributeTypeNodes(buildAttributes.asJava)
    val buildNameList = ConverterUtils.collectAttributeNamesWithExprId(buildAttributes.asJava)
    val inputBuildRelNode = RelBuilder.makeReadRel(
      buildTypeList,
      buildNameList,
      new java.util.ArrayList[ColumnTypeNode](),
      null, // no filter
      null, // no extension
      context,
      operatorId,
      "shard_virtual_build")

    // Extract buildBoundKeys type info for hash key schema.
    // This tells the probe side which types were used for hash partitioning
    // on the build side (e.g., cast(key as bigint) → bigint), so it can
    // compute shard hashes consistently even when the probe executor has
    // no local shard tables.
    val hashKeySchema = resolveBuildBoundKeys(buildPlan) match {
      case Some(keys) =>
        import org.apache.spark.sql.types.StructType
        StructType(keys.zipWithIndex.map {
          case (expr, i) =>
            org.apache.spark.sql.types.StructField(s"hk$i", expr.dataType)
        }).catalogString
      case None => ""
    }

    val joinRel = JoinUtils.createJoinRel(
      streamedKeyExprs,
      buildKeyExprs,
      condition,
      substraitJoinType,
      exchangeTable = false,
      joinType,
      genJoinParameters(shardSetId, numShards, shardLocationMap, hashKeySchema),
      inputStreamedRelNode,
      inputBuildRelNode,
      inputStreamedOutput,
      buildPlan.output,
      context,
      operatorId
    )

    TransformContext(output, joinRel)
  }

  @tailrec
  private def resolveShardSetRef(plan: SparkPlan): org.apache.spark.shard.ShardSetRef = plan match {
    case s: ShardExchangeExec => s.buildShardSet()
    case s: ColumnarShardExchangeExec => s.buildShardSet()
    case s: ShardQueryStageExec => s.shardSetRef
    case r: ReusedExchangeExec => resolveShardSetRef(r.child)
    // AQE may wrap the build plan with InputIteratorTransformer / RowToColumnarExec
    // and other unary wrappers; unwrap them to reach the underlying ShardQueryStageExec.
    case u: UnaryExecNode => resolveShardSetRef(u.child)
    case other =>
      throw new IllegalStateException(s"Unexpected build plan for DistributedMapJoin: $other")
  }

  /** Extract buildBoundKeys from the build plan's ShardExchange node. */
  @tailrec
  private def resolveBuildBoundKeys(plan: SparkPlan): Option[Seq[Expression]] = plan match {
    case s: ShardExchangeExec => Some(s.buildBoundKeys)
    case s: ColumnarShardExchangeExec => Some(s.buildBoundKeys)
    case s: ShardQueryStageExec => resolveBuildBoundKeys(s.plan)
    case r: ReusedExchangeExec => resolveBuildBoundKeys(r.child)
    case u: UnaryExecNode => resolveBuildBoundKeys(u.child)
    case _ => None
  }

  private def genJoinParameters(
      shardSetId: Long,
      numShards: Int,
      shardLocationMap: Map[Int, Seq[String]],
      hashKeySchema: String): Any = {
    val joinParametersStr = new StringBuffer("JoinParameters:")
    joinParametersStr.append("\n")
    joinParametersStr.append("isShardLookupJoin=1")
    joinParametersStr.append("\n")
    joinParametersStr.append(s"shardSetId=$shardSetId")
    joinParametersStr.append("\n")
    joinParametersStr.append(s"numShards=$numShards")
    joinParametersStr.append("\n")
    // Per-shard location map: "0=host1:port1|host2:port2,1=host3:port3,..."
    // Each entry is shardId=location1|location2|... with '|' separating
    // multiple replicas and ',' separating shards.
    val shardLocStr = shardLocationMap.toSeq
      .sortBy(_._1)
      .map { case (shardId, locs) => s"$shardId=${locs.mkString("|")}" }
      .mkString(",")
    joinParametersStr.append(s"shardLocationMap=$shardLocStr")
    joinParametersStr.append("\n")
    joinParametersStr.append(
      s"maxInflightRpcs=${conf.distributedMapJoinMaxInFlightNum}")
    joinParametersStr.append("\n")
    joinParametersStr.append(
      s"maxBatchSize=${conf.distributedMapJoinMaxBatchSize}")
    // Hash key schema tells the probe side which types were used for shard
    // hash partitioning on the build side (e.g., struct<hk0:bigint>).
    // Without this, probe executors that have no local shard tables cannot
    // determine whether to cast keys before hashing, causing hash mismatch.
    if (hashKeySchema.nonEmpty) {
      joinParametersStr.append("\n")
      joinParametersStr.append(s"hashKeySchema=$hashKeySchema")
    }

    val message = StringValue.newBuilder().setValue(joinParametersStr.toString).build()
    Any.pack(message)
  }
}
