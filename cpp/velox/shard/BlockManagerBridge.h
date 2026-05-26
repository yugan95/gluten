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
#pragma once

#include <jni.h>
#include <memory>
#include <string>

#include <folly/io/IOBuf.h>

namespace gluten {
namespace shard {

class BlockManagerBridge {
 public:
  virtual ~BlockManagerBridge() = default;

  // Read a block from BlockManager.
  // Returns an IOBuf containing the block data, or an empty IOBuf if not found.
  virtual std::unique_ptr<folly::IOBuf> readBlock(
      int64_t setId, int32_t shardId, const std::string& tag) = 0;
};

class JavaBlockManagerBridge : public BlockManagerBridge {
 public:
  JavaBlockManagerBridge(JavaVM* vm, jobject javaReader)
      : vm_(vm), javaReader_(nullptr) {
    JNIEnv* env;
    vm_->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8);
    javaReader_ = env->NewGlobalRef(javaReader);
    jclass cls = env->GetObjectClass(javaReader_);
    // Scala BlockManagerBridge.readBlock returns Array[Byte] (or null if not found)
    readBlockMethod_ = env->GetMethodID(
        cls, "readBlock", "(JILjava/lang/String;)[B");
    env->DeleteLocalRef(cls);
  }

  ~JavaBlockManagerBridge() override {
    if (javaReader_) {
      JNIEnv* env;
      int envStatus = vm_->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8);
      if (envStatus == JNI_OK) {
        env->DeleteGlobalRef(javaReader_);
      } else if (envStatus == JNI_EDETACHED) {
        // Attach temporarily to release the global ref, then detach.
        if (vm_->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) == JNI_OK) {
          env->DeleteGlobalRef(javaReader_);
          vm_->DetachCurrentThread();
        }
      }
      javaReader_ = nullptr;
    }
  }

  std::unique_ptr<folly::IOBuf> readBlock(
      int64_t setId, int32_t shardId, const std::string& tag) override {
    JNIEnv* env;
    int envStatus = vm_->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8);
    bool needsDetach = false;
    if (envStatus == JNI_EDETACHED) {
      vm_->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr);
      needsDetach = true;
    }

    jstring jTag = env->NewStringUTF(tag.c_str());
    jbyteArray result = static_cast<jbyteArray>(env->CallObjectMethod(
        javaReader_, readBlockMethod_, setId, shardId, jTag));
    env->DeleteLocalRef(jTag);

    // Check for Java exception thrown by BlockManagerBridge.readBlock().
    if (env->ExceptionCheck()) {
      env->ExceptionDescribe();
      env->ExceptionClear();
      if (needsDetach) {
        vm_->DetachCurrentThread();
      }
      return folly::IOBuf::create(0);
    }

    std::unique_ptr<folly::IOBuf> ioBuf;
    if (result != nullptr) {
      jsize length = env->GetArrayLength(result);
      if (length > 0) {
        ioBuf = folly::IOBuf::create(length);
        env->GetByteArrayRegion(
            result,
            0,
            length,
            reinterpret_cast<jbyte*>(ioBuf->writableData()));
        ioBuf->append(length);
      }
      env->DeleteLocalRef(result);
    }

    if (needsDetach) {
      vm_->DetachCurrentThread();
    }

    return ioBuf ? std::move(ioBuf) : folly::IOBuf::create(0);
  }

 private:
  JavaVM* vm_;
  jobject javaReader_;
  jmethodID readBlockMethod_;
};

} // namespace shard
} // namespace gluten
