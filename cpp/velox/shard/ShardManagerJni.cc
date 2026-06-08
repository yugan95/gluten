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

#include <jni.h>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "BlockManagerBridge.h"
#include "ShardBuilder.h"
#include "VeloxShardManager.h"
#include "VeloxShardRpcServer.h"
#include "memory/VeloxColumnarBatch.h"
#include "operators/serializer/VeloxColumnarBatchSerializer.h"
#include "utils/ObjectStore.h"
// Package: org.apache.spark.shard  →  org_apache_spark_shard
namespace {

// Cached JavaVM pointer – set once in JNI_OnLoad.
static JavaVM* gJavaVm = nullptr;

// Rethrow a C++ exception as a Java RuntimeException so the JVM sees it.
void rethrowAsJavaException(JNIEnv* env, const std::exception& ex) {
  jclass runtimeExceptionClass =
      env->FindClass("java/lang/RuntimeException");
  if (runtimeExceptionClass != nullptr) {
    env->ThrowNew(runtimeExceptionClass, ex.what());
  }
}

// Cast a jlong handle back to a typed pointer.
template <typename T>
T* handleToPtr(jlong handle) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(handle));
}

} // namespace

// ---------------------------------------------------------------------------
// initialize(host: String, port: Int): Int
//
// Initializes VeloxShardManager (memory pool) and starts VeloxShardRpcServer.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_nativeInitialize(
    JNIEnv* env,
    jobject /*obj*/,
    jstring host,
    jint port,
    jobject javaBlockManagerBridge) {
  try {
    // Obtain JavaVM* from the current JNIEnv (no separate JNI_OnLoad needed).
    if (!gJavaVm) {
      env->GetJavaVM(&gJavaVm);
    }

    const char* hostChars = env->GetStringUTFChars(host, nullptr);
    std::string hostStr(hostChars);
    env->ReleaseStringUTFChars(host, hostChars);

    auto bridge = std::make_shared<gluten::shard::JavaBlockManagerBridge>(
        gJavaVm, javaBlockManagerBridge);
    auto* shardManager = gluten::shard::VeloxShardManager::getInstance();
    shardManager->initialize(std::move(bridge), gJavaVm);

    return static_cast<jint>(
        gluten::shard::VeloxShardRpcServer::getInstance()->startServer(
            hostStr, static_cast<int>(port)));
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
    return -1;
  }
}

// ---------------------------------------------------------------------------
// shutdown(): Unit
//
// Stops VeloxShardRpcServer and shuts down VeloxShardManager.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_shutdown(
    JNIEnv* env,
    jobject /*obj*/) {
  try {
    gluten::shard::VeloxShardRpcServer::getInstance()->stopServer();
    gluten::shard::VeloxShardManager::getInstance()->shutdown();
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
  }
}

// ---------------------------------------------------------------------------
// constructShardTable(shardSetId: Long, shardId: Int): Unit
//
// Constructs a HashTable from BlockManager data using the BlockManagerBridge
// that was set during initialize().
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_constructShardTable(
    JNIEnv* env,
    jobject /*obj*/,
    jlong shardSetId,
    jint shardId) {
  try {
    gluten::shard::VeloxShardManager::getInstance()->constructShardTable(
        static_cast<int64_t>(shardSetId),
        static_cast<int32_t>(shardId));
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
  }
}

// ---------------------------------------------------------------------------
// destroyShardTable(shardSetId: Long): Unit
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_destroyShardTable(
    JNIEnv* env,
    jobject /*obj*/,
    jlong shardSetId) {
  try {
    gluten::shard::VeloxShardManager::getInstance()->destroyShardTable(
        static_cast<int64_t>(shardSetId));
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
  }
}



// ---------------------------------------------------------------------------
// createShardBuilder(numKeyColumns: Int, blockSize: Int,
//                    bloomCapacity: Long): Long
//
// Creates a ShardBuilder that streams ColumnarBatches into BlockManager pieces
// and simultaneously builds a BloomFilter.  Returns a native handle (pointer).
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jlong JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_createShardBuilder(
    JNIEnv* env,
    jobject /*obj*/,
    jint numKeyColumns,
    jint blockSize,
    jlong bloomCapacity) {
  try {
    static std::atomic<uint64_t> builderCounter{0};
    auto pool = facebook::velox::memory::memoryManager()->addLeafPool(
        fmt::format("ShardBuilder_{}", builderCounter++));
    auto* builder = new gluten::shard::ShardBuilder(
        static_cast<int32_t>(numKeyColumns),
        static_cast<int32_t>(blockSize),
        static_cast<int64_t>(bloomCapacity),
        std::move(pool));
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(builder));
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
    return 0;
  }
}

// ---------------------------------------------------------------------------
// appendBatchToShardBuilder(builderHandle: Long, batchHandle: Long): Array[Byte]
//
// Appends a ColumnarBatch to the ShardBuilder.  If the internal buffer exceeds
// blockSize, returns the flushed piece bytes; otherwise returns null.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jbyteArray JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_appendBatchToShardBuilder(
    JNIEnv* env,
    jobject /*obj*/,
    jlong builderHandle,
    jlong batchHandle) {
  try {
    auto* builder = handleToPtr<gluten::shard::ShardBuilder>(builderHandle);
    auto batch = gluten::ObjectStore::retrieve<gluten::ColumnarBatch>(
        static_cast<int64_t>(batchHandle));
    auto veloxBatch =
        std::dynamic_pointer_cast<gluten::VeloxColumnarBatch>(batch);
    VELOX_CHECK_NOT_NULL(
        veloxBatch, "Expected VeloxColumnarBatch in appendBatchToShardBuilder");

    auto piece = builder->appendBatch(veloxBatch->getRowVector());
    if (!piece.has_value()) {
      return nullptr; // no piece flushed yet
    }

    // Return flushed piece bytes as Java byte array.
    const auto& pieceData = piece.value();
    jbyteArray result =
        env->NewByteArray(static_cast<jsize>(pieceData.bytes.size()));
    env->SetByteArrayRegion(
        result, 0, static_cast<jsize>(pieceData.bytes.size()),
        reinterpret_cast<const jbyte*>(pieceData.bytes.data()));
    return result;
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// finishShardBuilder(builderHandle: Long): Array[Array[Byte]]
//
// Finishes the ShardBuilder and returns a 3-element array:
//   [0] = last piece bytes (or empty if no remaining data)
//   [1] = bloom filter bytes
//   [2] = checksums as int32 array encoded as bytes (4 bytes per checksum)
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_finishShardBuilder(
    JNIEnv* env,
    jobject /*obj*/,
    jlong builderHandle) {
  try {
    auto* builder = handleToPtr<gluten::shard::ShardBuilder>(builderHandle);
    auto result = builder->finish();

    jclass byteArrayClass = env->FindClass("[B");
    jobjectArray output = env->NewObjectArray(3, byteArrayClass, nullptr);

    // [0] last piece bytes
    if (result.lastPiece.has_value()) {
      const auto& lastPiece = result.lastPiece.value();
      jbyteArray lastPieceArr =
          env->NewByteArray(static_cast<jsize>(lastPiece.bytes.size()));
      env->SetByteArrayRegion(
          lastPieceArr, 0, static_cast<jsize>(lastPiece.bytes.size()),
          reinterpret_cast<const jbyte*>(lastPiece.bytes.data()));
      env->SetObjectArrayElement(output, 0, lastPieceArr);
      env->DeleteLocalRef(lastPieceArr);
    }
    // If no lastPiece, element [0] remains null (set by NewObjectArray default).

    // [1] bloom filter bytes
    jbyteArray bloomArr =
        env->NewByteArray(static_cast<jsize>(result.bloomBytes.size()));
    if (bloomArr == nullptr) {
      env->ExceptionClear();
      env->ThrowNew(
          env->FindClass("java/lang/OutOfMemoryError"),
          ("Failed to allocate bloom filter byte array of size " +
           std::to_string(result.bloomBytes.size()))
              .c_str());
      return nullptr;
    }
    if (!result.bloomBytes.empty()) {
      env->SetByteArrayRegion(
          bloomArr, 0, static_cast<jsize>(result.bloomBytes.size()),
          reinterpret_cast<const jbyte*>(result.bloomBytes.data()));
    }
    env->SetObjectArrayElement(output, 1, bloomArr);
    env->DeleteLocalRef(bloomArr);

    // [2] checksums as byte array (4 bytes per uint32)
    jsize checksumBytes =
        static_cast<jsize>(result.checksums.size() * sizeof(uint32_t));
    jbyteArray checksumArr = env->NewByteArray(checksumBytes);
    if (checksumBytes > 0) {
      env->SetByteArrayRegion(
          checksumArr, 0, checksumBytes,
          reinterpret_cast<const jbyte*>(result.checksums.data()));
    }
    env->SetObjectArrayElement(output, 2, checksumArr);
    env->DeleteLocalRef(checksumArr);

    return output;
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// destroyShardBuilder(builderHandle: Long): Unit
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_destroyShardBuilder(
    JNIEnv* env,
    jobject /*obj*/,
    jlong builderHandle) {
  try {
    delete handleToPtr<gluten::shard::ShardBuilder>(builderHandle);
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
  }
}

// ---------------------------------------------------------------------------
// mergeBloomFilters: merge multiple serialized Velox BloomFilters into one.
// (Legacy batch API – kept for backward compatibility.)
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jbyteArray JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_mergeBloomFilters(
    JNIEnv* env,
    jobject /*obj*/,
    jobjectArray bloomFilters) {
  try {
    jsize count = env->GetArrayLength(bloomFilters);
    if (count == 0) {
      return env->NewByteArray(0);
    }

    gluten::shard::BloomFilter64 merged;

    for (jsize i = 0; i < count; ++i) {
      auto jBytes =
          static_cast<jbyteArray>(env->GetObjectArrayElement(bloomFilters, i));
      if (jBytes == nullptr) {
        continue;
      }
      jsize len = env->GetArrayLength(jBytes);
      if (len == 0) {
        env->DeleteLocalRef(jBytes);
        continue;
      }
      std::vector<char> buf(len);
      env->GetByteArrayRegion(
          jBytes, 0, len, reinterpret_cast<jbyte*>(buf.data()));
      env->DeleteLocalRef(jBytes);

      merged.merge(buf.data());
    }

    if (!merged.isSet()) {
      return env->NewByteArray(0);
    }

    auto serializedSize = merged.serializedSize();
    std::vector<char> serializedData(serializedSize);
    merged.serialize(serializedData.data());

    jbyteArray result =
        env->NewByteArray(static_cast<jsize>(serializedSize));
    env->SetByteArrayRegion(
        result, 0, static_cast<jsize>(serializedSize),
        reinterpret_cast<const jbyte*>(serializedData.data()));
    return result;
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Streaming BloomFilter merge API – avoids loading all shard BFs at once.
//
// Usage:
//   long handle = createBloomFilterMerger();
//   for each shard:
//     byte[] bf = readBloomFromBM(shard);
//     mergeBloomFilterChunk(handle, bf);
//   byte[] merged = finishBloomFilterMerger(handle);
// ---------------------------------------------------------------------------

// createBloomFilterMerger(): Long
extern "C" JNIEXPORT jlong JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_createBloomFilterMerger(
    JNIEnv* env,
    jobject /*obj*/) {
  try {
    auto* merger = new gluten::shard::BloomFilter64();
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(merger));
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
    return 0;
  }
}

// mergeBloomFilterChunk(mergerHandle: Long, bloomFilter: Array[Byte]): Unit
extern "C" JNIEXPORT void JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_mergeBloomFilterChunk(
    JNIEnv* env,
    jobject /*obj*/,
    jlong mergerHandle,
    jbyteArray bloomFilter) {
  try {
    auto* merger = handleToPtr<gluten::shard::BloomFilter64>(mergerHandle);
    if (bloomFilter == nullptr) {
      return;
    }
    jsize len = env->GetArrayLength(bloomFilter);
    if (len == 0) {
      return;
    }
    std::vector<char> buf(len);
    env->GetByteArrayRegion(
        bloomFilter, 0, len, reinterpret_cast<jbyte*>(buf.data()));
    merger->merge(buf.data());
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
  }
}

// finishBloomFilterMerger(mergerHandle: Long): Array[Byte]
// Also destroys the native merger.
extern "C" JNIEXPORT jbyteArray JNICALL
Java_org_apache_spark_shard_GlutenShardManagerJni_00024_finishBloomFilterMerger(
    JNIEnv* env,
    jobject /*obj*/,
    jlong mergerHandle) {
  try {
    auto* merger = handleToPtr<gluten::shard::BloomFilter64>(mergerHandle);

    jbyteArray result;
    if (!merger->isSet()) {
      result = env->NewByteArray(0);
    } else {
      auto serializedSize = merger->serializedSize();
      std::vector<char> serializedData(serializedSize);
      merger->serialize(serializedData.data());

      result = env->NewByteArray(static_cast<jsize>(serializedSize));
      env->SetByteArrayRegion(
          result, 0, static_cast<jsize>(serializedSize),
          reinterpret_cast<const jbyte*>(serializedData.data()));
    }

    delete merger;
    return result;
  } catch (const std::exception& ex) {
    rethrowAsJavaException(env, ex);
    return nullptr;
  }
}
