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
package org.apache.spark.storage

import org.apache.spark.internal.Logging
import org.apache.spark.util.io.ChunkedByteBuffer

/**
 * JNI callback bridge that allows C++ code to read blocks from Spark's BlockManager.
 *
 * C++ side holds a JNI global reference to this object and calls [[readBlock]] via
 * `JniBlockManagerBridge` (see BlockManagerBridge.h). The read path tries local bytes first, then
 * falls back to remote fetch with local caching.
 *
 * Thread safety: instances are NOT thread-safe. Each should be used by a single thread, or
 * externally synchronized.
 */
class GlutenBlockManagerBridge(blockManager: BlockManager) extends Logging {

  /**
   * Read a block from BlockManager, trying local first then remote.
   *
   * @param shardSetId
   *   Shard set ID
   * @param shardId
   *   Shard ID (-1 for shard-set-level blocks such as merged bloom filter)
   * @param blockType
   *   Block type tag (e.g., "piece0", "meta", "bloom")
   * @return
   *   Block data as Array[Byte], or null if not found
   */
  def readBlock(shardSetId: Long, shardId: Int, blockType: String): Array[Byte] = {
    val blockId = ShardBlockId(shardSetId, shardId, blockType)
    val localResult = blockManager.getLocalBytes(blockId)
    if (localResult.isDefined) {
      val blockData = localResult.get
      val buf = blockData.toByteBuffer()
      val bytes = new Array[Byte](buf.remaining())
      buf.get(bytes)
      blockData.dispose()
      bytes
    } else {
      // Not found locally; try remote fetch.
      val remoteResult =
        try {
          blockManager.getRemoteBytes(blockId)
        } catch {
          case e: Exception =>
            logWarning(s"[BMBridge] getRemoteBytes EXCEPTION for $blockId: ${e.getMessage}")
            None
        }
      if (remoteResult.isDefined) {
        val bytes = remoteResult.get.toArray
        logInfo(s"[BMBridge] Remote fetch OK for $blockId, size=${bytes.length}")
        blockManager.putBytes(
          blockId,
          new ChunkedByteBuffer(java.nio.ByteBuffer.wrap(bytes)),
          StorageLevel.MEMORY_AND_DISK_SER,
          tellMaster = true)
        bytes
      } else {
        logWarning(
          s"[BMBridge] Block NOT FOUND (local=false, remote=false) for $blockId, " +
            s"bmId=${blockManager.blockManagerId}")
        null
      }
    }
  }
}
