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
package org.apache.spark.network

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.network.shard.ShardLookupListener
import org.apache.spark.shard.{GlutenShardManagerJni, ShardManager}

import scala.concurrent.Future

/**
 * GlutenShardLookupService manages the lifecycle of Velox native shard components
 * (VeloxShardManager + VeloxShardRpcServer).
 */
class GlutenShardLookupService(
    conf: SparkConf,
    bindAddress: String,
    val hostName: String,
    requestedPort: Int)
  extends ShardLookupService
  with Logging {

  private var actualPort: Int = -1

  override def init(shardManager: ShardManager): Unit = {
    actualPort = GlutenShardManagerJni.initialize(hostName, 0)
    logInfo(s"Velox native shard service started on $hostName:$actualPort")
  }

  override def port: Int = actualPort

  override def supportsNativeLookup: Boolean = true

  // ShardLookupService (Scala) override: returns Future[ManagedBuffer]
  override def fetchBatch(host: String, port: Int, reqMsg: ManagedBuffer): Future[ManagedBuffer] = {
    throw new UnsupportedOperationException(
      "GlutenShardLookupService does not support JVM fetchBatch; " +
        "probe-side lookup is handled by Velox gRPC ShardSource")
  }

  // ShardStoreClient (Java) override: void + listener
  override def fetchBatch(
      host: String,
      port: Int,
      reqMsg: ManagedBuffer,
      listener: ShardLookupListener): Unit = {
    throw new UnsupportedOperationException(
      "GlutenShardLookupService does not support JVM fetchBatch; " +
        "probe-side lookup is handled by Velox gRPC ShardSource")
  }

  override def installNativeReplica(setId: Long, shardId: Int): Unit = {
    GlutenShardManagerJni.constructShardTable(setId, shardId)
  }

  override def close(): Unit = {
    GlutenShardManagerJni.shutdown()
  }
}
