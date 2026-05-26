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

import org.apache.spark.{SecurityManager, SparkConf}
import org.apache.spark.internal.Logging
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.network.shard.ShardLookupListener
import org.apache.spark.rpc.RpcEndpointRef
import org.apache.spark.shard.{GlutenShardCleanupCallback, GlutenShardManagerJni, ShardManager}

/**
 * GlutenShardLookupService manages the lifecycle of Velox native shard components
 * (VeloxShardManager + VeloxShardRpcServer).
 *
 * Constructor signature matches what ShardLookupServiceFactory expects
 * for custom implementations loaded via reflection.
 */
class GlutenShardLookupService(
    conf: SparkConf,
    securityManager: SecurityManager,
    bindAddress: String,
    val hostName: String,
    requestedPort: Int,
    numCores: Int,
    masterEndpoint: RpcEndpointRef)
  extends ShardLookupService
  with Logging {

  private var actualPort: Int = -1

  override def init(shardManager: ShardManager): Unit = {
    actualPort = GlutenShardManagerJni.initialize(hostName, 0)
    shardManager.registerCleanupCallback(GlutenShardCleanupCallback)
    logInfo(s"Velox native shard service started on $hostName:$actualPort")
  }

  override def port: Int = actualPort

  override def onInstallReplica(shardManager: ShardManager, setId: Long, shardId: Int): Unit = {
    GlutenShardManagerJni.constructShardTable(setId, shardId)
  }

  override def fetchBatch(
      host: String,
      port: Int,
      reqMsg: ManagedBuffer,
      listener: ShardLookupListener): Unit = {
    throw new UnsupportedOperationException(
      "GlutenShardLookupService does not support JVM fetchBatch; " +
        "probe-side lookup is handled by Velox gRPC ShardSource")
  }

  override def close(): Unit = {
    GlutenShardManagerJni.shutdown()
  }
}
