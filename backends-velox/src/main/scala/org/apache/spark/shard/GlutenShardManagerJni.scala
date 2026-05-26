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
package org.apache.spark.shard

import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.storage.GlutenBlockManagerBridge

/** JNI entry point for Velox shard components (VeloxShardManager + VeloxShardRpcServer). */
object GlutenShardManagerJni extends Logging {

  // Shard JNI symbols are compiled into libvelox.so, which is loaded by
  // the Gluten plugin (VeloxListenerApi + JniLibLoader).  No separate
  // native library needs to be loaded here.

  def initialize(host: String, port: Int): Int = {
    val bmb = new GlutenBlockManagerBridge(SparkEnv.get.blockManager)
    nativeInitialize(host, port, bmb)
  }

  @native
  private def nativeInitialize(host: String, port: Int, bmb: GlutenBlockManagerBridge): Int

  @native
  def shutdown(): Unit

  @native
  def constructShardTable(shardSetId: Long, shardId: Int): Unit

  @native
  def destroyShardTable(shardSetId: Long): Unit

  @native
  def createShardBuilder(
      numKeyColumns: Int,
      blockSize: Int,
      bloomCapacity: Long): Long

  @native
  def appendBatchToShardBuilder(builderHandle: Long, batchHandle: Long): Array[Byte]

  @native
  def finishShardBuilder(builderHandle: Long): Array[Array[Byte]]

  @native
  def destroyShardBuilder(builderHandle: Long): Unit

  @native
  def mergeBloomFilters(bloomFilters: Array[Array[Byte]]): Array[Byte]

  @native
  def createBloomFilterMerger(): Long

  @native
  def mergeBloomFilterChunk(mergerHandle: Long, bloomFilter: Array[Byte]): Unit

  @native
  def finishBloomFilterMerger(mergerHandle: Long): Array[Byte]

  /**
   * Refresh shard locations from ShardManagerMaster.
   *
   * Invalidates the Spark-side location cache and fetches the latest
   * locations, including newly installed replicas after executor crash
   * recovery. Mirrors Spark DMJ's `locations(refresh = true)` in
   * `ShardManager.fetchRemoteBatch`.
   *
   * Called from C++ via JNI when all known replicas for a shard are
   * exhausted during gRPC lookup.
   *
   * @return Array of "host:port" strings, or empty array if unavailable
   */
  def refreshShardLocations(shardSetId: Long, shardId: Int): Array[String] = {
    try {
      val shardManager = SparkEnv.get.shardManager
      val locations = shardManager.master.getLocations(shardSetId, shardId, refresh = true)
      val result = locations.map(loc => s"${loc.host}:${loc.port}").toArray
      logInfo(s"[ShardManagerJni] Refreshed shard locations ($shardSetId, $shardId): " +
        s"${result.mkString("[", ", ", "]")}")
      result
    } catch {
      case e: Exception =>
        logWarning(s"[ShardManagerJni] Failed to refresh shard locations " +
          s"($shardSetId, $shardId): ${e.getMessage}")
        Array.empty[String]
    }
  }
}
