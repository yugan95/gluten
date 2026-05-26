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
#include "VeloxShardManager.h"

#include <jni.h>
#include <chrono>
#include <cstdlib>
#include <numeric>

#include <folly/json.h>
#include <folly/io/IOBuf.h>

#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/StreamArena.h"
#include "velox/core/Expressions.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/VectorHasher.h"
#include "velox/expression/Expr.h"
#include "velox/row/CompactRow.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/Type.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/DictionaryVector.h"

#include "CompactRowWireFormat.h"

// The meta block binary format uses little-endian int32 fields, written by
// JVM (ByteBuffer with ByteOrder.LITTLE_ENDIAN) and read here via
// reinterpret_cast.  This is correct on little-endian platforms (x86_64,
// aarch64 in LE mode).  Fail at compile time on big-endian platforms to
// prevent silent data corruption.
static_assert(
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
    "VeloxShardManager meta block parsing assumes little-endian byte order. "
    "Big-endian platforms are not supported.");

namespace gluten {
namespace shard {

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

// Merge multiple CompactRow wire-format payloads into a single payload.
// Each input must be a valid wire payload (magic + version + numRows + body).
// The output payload's numRows is the sum of all inputs and its body is the
// concatenation of all input bodies (preserving row order).
// Only used by lookupAndSerialize below.
std::string mergeCompactRowWirePayloads(
    const std::vector<std::string>& payloads) {
  uint32_t totalRows = 0;
  size_t totalBodyBytes = 0;
  for (const auto& payload : payloads) {
    if (payload.size() < kCompactRowWireHeaderSize) {
      continue;
    }
    const auto magic = detail::readUint32LE(payload.data());
    VELOX_USER_CHECK_EQ(magic, kCompactRowWireMagic,
        "mergeCompactRowWirePayloads: magic mismatch");
    const auto version = detail::readUint32LE(payload.data() + 4);
    VELOX_USER_CHECK_EQ(version, kCompactRowWireVersion,
        "mergeCompactRowWirePayloads: version mismatch");
    const auto numRows = detail::readUint32LE(payload.data() + 8);
    if (numRows == 0) {
      continue;
    }
    totalRows += numRows;
    totalBodyBytes += payload.size() - kCompactRowWireHeaderSize;
  }

  std::string out(kCompactRowWireHeaderSize + totalBodyBytes, '\0');
  char* base = out.data();
  detail::writeUint32LE(base, kCompactRowWireMagic);
  detail::writeUint32LE(base + 4, kCompactRowWireVersion);
  detail::writeUint32LE(base + 8, totalRows);

  size_t cursor = kCompactRowWireHeaderSize;
  for (const auto& payload : payloads) {
    if (payload.size() <= kCompactRowWireHeaderSize) {
      continue;
    }
    const auto numRows = detail::readUint32LE(payload.data() + 8);
    if (numRows == 0) {
      continue;
    }
    size_t bodySize = payload.size() - kCompactRowWireHeaderSize;
    std::memcpy(base + cursor,
        payload.data() + kCompactRowWireHeaderSize, bodySize);
    cursor += bodySize;
  }
  VELOX_CHECK_EQ(cursor, out.size());
  return out;
}

// Helper: wrap raw bytes in a BufferInputStream for PrestoVectorSerde.
std::unique_ptr<ByteInputStream> toByteStream(
    const uint8_t* data,
    size_t size) {
  std::vector<ByteRange> ranges;
  ranges.push_back(
      ByteRange{const_cast<uint8_t*>(data), static_cast<int32_t>(size), 0});
  return std::make_unique<BufferInputStream>(std::move(ranges));
}

// Helper: deserialize Presto-format bytes into RowVectors using the given
// RowType.  A single piece may contain multiple concatenated Presto-serialized
// batches (ShardBuilder.appendBatch serializes each batch independently and
// appends the bytes to a buffer).  We must loop until the stream is fully
// consumed to recover all batches.
std::vector<RowVectorPtr> deserializeAllRowVectors(
    const uint8_t* data,
    size_t size,
    const RowTypePtr& rowType,
    memory::MemoryPool* pool) {
  auto serde = std::make_unique<serializer::presto::PrestoVectorSerde>();
  auto byteStream = toByteStream(data, size);
  serializer::presto::PrestoVectorSerde::PrestoOptions opts;
  opts.useLosslessTimestamp = true;

  std::vector<RowVectorPtr> batches;
  while (!byteStream->atEnd()) {
    RowVectorPtr result;
    serde->deserialize(byteStream.get(), pool, rowType, &result, &opts);
    if (result && result->size() > 0) {
      batches.push_back(std::move(result));
    }
  }
  return batches;
}

} // namespace

// ---------------------------------------------------------------------------
// VeloxShardManager lifecycle
// ---------------------------------------------------------------------------
void VeloxShardManager::initialize(
    std::shared_ptr<BlockManagerBridge> bridge,
    JavaVM* javaVm) {
  bool expected = false;
  if (initialized_.compare_exchange_strong(expected, true)) {
    const char* dbgEnv = std::getenv("GLUTEN_SHARD_DMJ_DEBUG");
    debugEnabled_ = (dbgEnv != nullptr && std::string(dbgEnv) == "1");

    const char* ttlEnv = std::getenv("GLUTEN_SHARD_DEAD_ADDR_TTL_SECS");
    if (ttlEnv) {
      deadAddressTtl_ = std::chrono::seconds(std::atoi(ttlEnv));
    }

    pool_ = memory::memoryManager()->addRootPool(
        "VeloxShardManager", memory::kMaxMemory);
    blockManagerBridge_ = std::move(bridge);
    javaVm_ = javaVm;

    // Cache JNI handles for refreshShardLocations() so the hot-path
    // avoids repeated FindClass / GetMethodID lookups.
    if (javaVm_) {
      JNIEnv* env = nullptr;
      int envStatus = javaVm_->GetEnv(
          reinterpret_cast<void**>(&env), JNI_VERSION_1_8);
      bool needsDetach = false;
      if (envStatus == JNI_EDETACHED) {
        javaVm_->AttachCurrentThread(
            reinterpret_cast<void**>(&env), nullptr);
        needsDetach = true;
      }
      jclass localClass = env->FindClass(
          "org/apache/spark/shard/GlutenShardManagerJni$");
      if (localClass != nullptr && !env->ExceptionCheck()) {
        jniGlobalClass_ =
            static_cast<jclass>(env->NewGlobalRef(localClass));
        jfieldID moduleField = env->GetStaticFieldID(
            jniGlobalClass_, "MODULE$",
            "Lorg/apache/spark/shard/GlutenShardManagerJni$;");
        jobject localModule = env->GetStaticObjectField(
            jniGlobalClass_, moduleField);
        jniGlobalModule_ = env->NewGlobalRef(localModule);
        jniRefreshMethod_ = env->GetMethodID(
            jniGlobalClass_, "refreshShardLocations",
            "(JI)[Ljava/lang/String;");
        env->DeleteLocalRef(localModule);
        env->DeleteLocalRef(localClass);
      } else {
        if (env->ExceptionCheck()) {
          env->ExceptionClear();
        }
        LOG(WARNING)
            << "VeloxShardManager::initialize: failed to cache JNI handles "
               "for GlutenShardManagerJni$; refreshShardLocations will "
               "fall back to per-call lookup";
      }
      if (needsDetach) {
        javaVm_->DetachCurrentThread();
      }
    }
  }
}

void VeloxShardManager::shutdown() {
  // Clear HashTables first, then leaf pools, then set pools, then root pool.
  // Order matters: each level must be destroyed before its parent pool.
  {
    auto tables = shardTables_.wlock();
    tables->clear();
  }
  {
    auto types = hashKeyTypes_.wlock();
    types->clear();
  }
  {
    auto pools = shardPools_.wlock();
    pools->clear();
  }
  {
    auto pools = setPools_.wlock();
    pools->clear();
  }
  {
    auto filters = bloomFilters_.wlock();
    filters->clear();
  }
  // Clear filter cache before releasing pools (CachedFilter holds child pools).
  {
    std::lock_guard<std::mutex> lock(filterCacheMutex_);
    filterCache_.clear();
  }
  // Clear gRPC channel cache to release connections and references.
  {
    std::lock_guard<std::mutex> lock(channelsMutex_);
    grpcChannels_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(deadAddressMutex_);
    deadAddresses_.clear();
  }
  // Release cached JNI GlobalRefs before dropping javaVm_.
  if (javaVm_ && jniGlobalClass_) {
    JNIEnv* env = nullptr;
    int envStatus = javaVm_->GetEnv(
        reinterpret_cast<void**>(&env), JNI_VERSION_1_8);
    bool needsDetach = false;
    if (envStatus == JNI_EDETACHED) {
      javaVm_->AttachCurrentThread(
          reinterpret_cast<void**>(&env), nullptr);
      needsDetach = true;
    }
    if (env) {
      env->DeleteGlobalRef(jniGlobalModule_);
      env->DeleteGlobalRef(jniGlobalClass_);
    }
    jniGlobalClass_ = nullptr;
    jniGlobalModule_ = nullptr;
    jniRefreshMethod_ = nullptr;
    if (needsDetach) {
      javaVm_->DetachCurrentThread();
    }
  }
  blockManagerBridge_.reset();
  pool_.reset();
  initialized_.store(false);
}

void VeloxShardManager::constructShardTable(int64_t setId, int32_t shardId) {
  VELOX_CHECK(
      blockManagerBridge_,
      "BlockManagerBridge not set; call initialize() first");
  // Read metadata block.
  // Meta format: [int32 pieceCount] [int32 numKeyColumns]
  //              [int32 schemaLen] [char[schemaLen] schemaStr]
  auto metaBuf = blockManagerBridge_->readBlock(setId, shardId, "meta");
  if (!metaBuf || metaBuf->empty()) {
    return;
  }
  metaBuf->coalesce();
  const uint8_t* metaPtr = metaBuf->data();
  size_t metaLen = metaBuf->length();
  constexpr size_t kFixedHeaderSize = 3 * sizeof(int32_t);
  if (metaLen < kFixedHeaderSize) {
    return;
  }
  int32_t pieceCount = *reinterpret_cast<const int32_t*>(metaPtr);
  int32_t numKeys =
      *reinterpret_cast<const int32_t*>(metaPtr + sizeof(int32_t));
  int32_t schemaLen =
      *reinterpret_cast<const int32_t*>(metaPtr + 2 * sizeof(int32_t));
  if (pieceCount < 0) {
    return;
  }
  if (schemaLen <= 0 ||
      metaLen < kFixedHeaderSize + static_cast<size_t>(schemaLen)) {
    return;
  }
  std::string schemaStr(
      reinterpret_cast<const char*>(metaPtr + kFixedHeaderSize), schemaLen);

  // Parse hash key schema (types used for hash partitioning, may differ from
  // data types when buildBoundKeys contains cast expressions).
  size_t hashKeySchemaOffset = kFixedHeaderSize + static_cast<size_t>(schemaLen);
  std::string hashKeySchemaStr;
  if (hashKeySchemaOffset + sizeof(int32_t) <= metaLen) {
    int32_t hashKeySchemaLen =
        *reinterpret_cast<const int32_t*>(metaPtr + hashKeySchemaOffset);
    if (hashKeySchemaLen > 0 &&
        hashKeySchemaOffset + sizeof(int32_t) + hashKeySchemaLen <= metaLen) {
      hashKeySchemaStr = std::string(
          reinterpret_cast<const char*>(
              metaPtr + hashKeySchemaOffset + sizeof(int32_t)),
          hashKeySchemaLen);
    }
  }

  type::fbhive::HiveTypeParser typeParser;
  auto parsedType = typeParser.parse(schemaStr);
  auto buildType = std::dynamic_pointer_cast<const RowType>(parsedType);
  VELOX_CHECK_NOT_NULL(
      buildType,
      "Failed to parse schema from meta block: {}",
      schemaStr);

  // Parse hash key type for shard routing.
  RowTypePtr hashKeyType;
  if (!hashKeySchemaStr.empty()) {
    auto parsedHashKeyType = typeParser.parse(hashKeySchemaStr);
    hashKeyType = std::dynamic_pointer_cast<const RowType>(parsedHashKeyType);
  }

  // Get or create a per-setId pool.  All shards in the same shard set share
  // one pool, providing query-level memory accounting without per-shard overhead.
  std::shared_ptr<memory::MemoryPool> setPool;
  {
    auto pools = setPools_.wlock();
    auto it = pools->find(setId);
    if (it != pools->end()) {
      setPool = it->second;
    } else {
      setPool = pool_->addAggregateChild(fmt::format("shardSet_{}", setId));
      (*pools)[setId] = setPool;
    }
  }
  auto leafPool = setPool->addLeafChild(
      fmt::format("shard_{}_{}", setId, shardId));

  // Read and deserialize pieces using the schema from the meta block.
  // Use a separate leaf pool for deserialization to avoid buffer end guard
  // issues in debug builds when the same pool is shared with RowContainer.
  auto deserPool = setPool->addLeafChild(
      fmt::format("deser_{}_{}", setId, shardId));
  std::vector<RowVectorPtr> allBatches;
  vector_size_t totalRows = 0;

  for (int32_t i = 0; i < pieceCount; ++i) {
    auto pieceBuf = blockManagerBridge_->readBlock(
        setId, shardId, "piece" + std::to_string(i));
    if (!pieceBuf || pieceBuf->empty()) {
      continue;
    }
    pieceBuf->coalesce();
    auto batches = deserializeAllRowVectors(
        pieceBuf->data(), pieceBuf->length(), buildType, deserPool.get());
    for (auto& batch : batches) {
      if (batch && batch->size() > 0) {
        totalRows += batch->size();
        allBatches.push_back(std::move(batch));
      }
    }
  }

  if (totalRows == 0) {
    // Empty shard (no data rows).  Register a nullptr entry in shardTables_
    // so that lookup() can quickly return empty results without retrying
    // lazy construction every time.
    allBatches.clear();
    ShardKey key{setId, shardId};
    {
      auto tables = shardTables_.wlock();
      (*tables)[key] = nullptr;
    }
    // Store hash key type even for empty shards so that ShardLookupJoin can
    // compute shard IDs consistently with the build-side partitioning.
    // Without this, getHashKeyType(setId, 0) returns nullptr when shard 0
    // is empty, causing incorrect hash computation (hashInt vs hashLong).
    if (hashKeyType) {
      auto types = hashKeyTypes_.wlock();
      (*types)[key] = hashKeyType;
    }
    return;
  }

  // Create VectorHashers for key columns.
  size_t effectiveNumKeys = std::min(
      static_cast<size_t>(numKeys), static_cast<size_t>(buildType->size()));
  std::vector<std::unique_ptr<VectorHasher>> keyHashers;
  keyHashers.reserve(effectiveNumKeys);
  for (size_t i = 0; i < effectiveNumKeys; ++i) {
    keyHashers.push_back(VectorHasher::create(
        buildType->childAt(i), static_cast<column_index_t>(i)));
  }

  // Determine dependent (non-key) column types.
  std::vector<TypePtr> dependentTypes;
  for (size_t i = effectiveNumKeys; i < buildType->size(); ++i) {
    dependentTypes.push_back(buildType->childAt(i));
  }

  // Create HashTable.
  // Use HashTable<false> (ignoreNullKeys=false).  HashTable<true> skips
  // null-key rows during rehash which can cause hash-mode mismatches
  // between build and probe.
  auto hashTable = HashTable<false>::createForJoin(
      std::move(keyHashers),
      dependentTypes,
      /*allowDuplicates=*/true,
      /*hasProbedFlag=*/false,
      /*minTableSizeForParallelJoinBuild=*/0,
      leafPool.get());

  // Insert all batches into RowContainer following the standard HashBuild
  // flow: decode key columns through VectorHashers (collecting cardinality
  // statistics for decideHashMode), then store rows into RowContainer.
  auto& hashers = hashTable->hashers();
  auto* rowContainer = hashTable->rows();
  bool analyzeKeys = true;
  raw_vector<uint64_t> hashes;

  // Prepare DecodedVectors for dependent (non-key) columns.
  std::vector<DecodedVector> dependentDecoders(dependentTypes.size());

  for (const auto& batch : allBatches) {
    auto numRows = batch->size();
    auto activeRows = SelectivityVector(numRows);

    // Step 1: Decode key columns through VectorHashers.
    for (size_t i = 0; i < hashers.size(); ++i) {
      auto key = batch->childAt(hashers[i]->channel())->loadedVector();
      hashers[i]->decode(*key, activeRows);
    }

    // Step 2: Collect cardinality statistics for decideHashMode.
    // This mirrors HashBuild::addInput's analyzeKeys logic.
    if (analyzeKeys) {
      if (hashes.size() < static_cast<size_t>(activeRows.end())) {
        hashes.resize(activeRows.end());
      }
      for (auto& hasher : hashers) {
        if (analyzeKeys) {
          hasher->computeValueIds(activeRows, hashes);
          analyzeKeys = hasher->mayUseValueIds();
        }
      }
    }

    // Step 3: Decode dependent columns.
    for (size_t i = 0; i < dependentTypes.size(); ++i) {
      dependentDecoders[i].decode(
          *batch->childAt(effectiveNumKeys + i)->loadedVector(), activeRows);
    }

    // Step 4: Store rows into RowContainer.
    activeRows.applyToSelected([&](auto rowIndex) {
      char* newRow = rowContainer->newRow();
      for (size_t i = 0; i < hashers.size(); ++i) {
        rowContainer->store(
            hashers[i]->decodedVector(), rowIndex, newRow, i);
      }
      for (size_t i = 0; i < dependentTypes.size(); ++i) {
        rowContainer->store(
            dependentDecoders[i], rowIndex, newRow, i + hashers.size());
      }
    });
  }

  // Release deserialized batches and the deserialization pool now that all
  // data has been stored into the RowContainer.
  allBatches.clear();
  deserPool.reset();

  // Build hash index.
  // Use kNoSpillInputStartPartitionBit (-1).  Passing 0 can trigger
  // VELOX_CHECK failures in checkHashBitsOverlap for non-array hash modes.
  hashTable->prepareJoinTable(
      /*tables=*/{},
      BaseHashTable::kNoSpillInputStartPartitionBit,
      /*executor=*/nullptr);

  // Store the HashTable, its leaf pool, and the hash key type.
  ShardKey key{setId, shardId};
  {
    auto tables = shardTables_.wlock();
    (*tables)[key] = std::move(hashTable);
  }
  {
    auto pools = shardPools_.wlock();
    (*pools)[key] = std::move(leafPool);
  }
  if (hashKeyType) {
    auto types = hashKeyTypes_.wlock();
    (*types)[key] = hashKeyType;
  }
}

void VeloxShardManager::destroyShardTable(int64_t setId) {
  // Order matters: HashTables must be destroyed before their leaf pools,
  // and leaf pools before the set-level aggregate pool.
  {
    auto tables = shardTables_.wlock();
    for (auto it = tables->begin(); it != tables->end();) {
      if (it->first.setId == setId) {
        it = tables->erase(it);
      } else {
        ++it;
      }
    }
  }
  {
    auto types = hashKeyTypes_.wlock();
    for (auto it = types->begin(); it != types->end();) {
      if (it->first.setId == setId) {
        it = types->erase(it);
      } else {
        ++it;
      }
    }
  }
  {
    auto pools = shardPools_.wlock();
    for (auto it = pools->begin(); it != pools->end();) {
      if (it->first.setId == setId) {
        it = pools->erase(it);
      } else {
        ++it;
      }
    }
  }
  {
    auto pools = setPools_.wlock();
    pools->erase(setId);
  }
  {
    auto filters = bloomFilters_.wlock();
    filters->erase(setId);
  }
  {
    std::lock_guard<std::mutex> lock(filterCacheMutex_);
    filterCache_.erase(setId);
  }
}

RowTypePtr VeloxShardManager::getKeyType(int64_t setId, int32_t shardId) {
  auto tables = shardTables_.rlock();
  auto it = tables->find(ShardKey{setId, shardId});
  if (it == tables->end()) {
    return nullptr;
  }
  auto& hashTable = it->second;
  if (!hashTable) {
    // Empty shard (no data rows).
    return nullptr;
  }
  auto& hashers = hashTable->hashers();
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(hashers.size());
  types.reserve(hashers.size());
  for (size_t i = 0; i < hashers.size(); ++i) {
    names.push_back(fmt::format("k{}", i));
    types.push_back(hashers[i]->type());
  }
  return ROW(std::move(names), std::move(types));
}

std::string VeloxShardManager::fetchBloomFilter(
    int64_t setId,
    int32_t shardId) {
  if (!blockManagerBridge_) {
    return "";
  }
  auto buf = blockManagerBridge_->readBlock(setId, shardId, "nativeBloom");
  if (!buf || buf->empty()) {
    return "";
  }
  buf->coalesce();
  return std::string(
      reinterpret_cast<const char*>(buf->data()), buf->length());
}

std::shared_ptr<const BloomFilter64>
VeloxShardManager::getOrLoadBloomFilter(int64_t setId) {
  {
    auto filters = bloomFilters_.rlock();
    auto it = filters->find(setId);
    if (it != filters->end()) {
      return it->second;
    }
  }
  auto filters = bloomFilters_.wlock();
  auto it = filters->find(setId);
  if (it != filters->end()) {
    return it->second;
  }
  auto bfData = fetchBloomFilter(setId, -1);
  if (bfData.empty()) {
    return nullptr;
  }
  auto bf = std::make_shared<BloomFilter64>();
  bf->merge(bfData.data());
  auto constBf = std::const_pointer_cast<const BloomFilter64>(bf);
  filters->emplace(setId, constBf);
  return constBf;
}

// ---------------------------------------------------------------------------
// Dead address blacklist
// ---------------------------------------------------------------------------
void VeloxShardManager::markAddressDead(const std::string& address) {
  auto expiry = std::chrono::steady_clock::now() + deadAddressTtl_;
  std::lock_guard<std::mutex> lock(deadAddressMutex_);
  deadAddresses_[address] = expiry;
  if (debugEnabled_) {
    LOG(INFO) << "[ShardMgr-dbg] markAddressDead: " << address
              << " ttl=" << deadAddressTtl_.count() << "s";
  }
}

bool VeloxShardManager::isAddressDead(const std::string& address) {
  std::lock_guard<std::mutex> lock(deadAddressMutex_);
  auto it = deadAddresses_.find(address);
  if (it == deadAddresses_.end()) {
    return false;
  }
  if (std::chrono::steady_clock::now() >= it->second) {
    deadAddresses_.erase(it);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// lookupAndSerialize – probe + batched extract + serialize
//
// Identical probe logic to lookup(), but instead of materializing all matched
// rows into a single RowVector, extracts them in batches of kSerializeBatchSize
// rows, serializes each batch to CompactRow wire format immediately, then
// releases the batch columns.  The per-batch payloads are merged at the end.
//
// Peak memory: O(kSerializeBatchSize × numCols) instead of
//              O(totalMatches × numCols).
// ---------------------------------------------------------------------------
VeloxShardManager::SerializedLookupResult VeloxShardManager::lookupAndSerialize(
    int64_t setId,
    int32_t shardId,
    const RowVectorPtr& probeKeys,
    const std::vector<int32_t>& inputIndices,
    memory::MemoryPool* rpcPool,
    const RowVectorPtr& probeFilterColumns,
    const std::string& filterExprJson,
    const RowTypePtr& probeColumnsType,
    const RowTypePtr& filterInputType) {
  SerializedLookupResult result;

  if (!probeKeys || probeKeys->size() == 0) {
    return result;
  }

  VELOX_CHECK_EQ(
      probeKeys->size(),
      inputIndices.size(),
      "probeKeys row count ({}) must match inputIndices size ({})",
      probeKeys->size(),
      inputIndices.size());

  // --- Ensure shard table exists (lazy construction) ---
  {
    auto tables = shardTables_.rlock();
    if (tables->find(ShardKey{setId, shardId}) == tables->end()) {
      tables.unlock();
      try {
        constructShardTable(setId, shardId);
      } catch (const std::exception& ex) {
        LOG(WARNING) << "Lazy constructShardTable failed for shard "
                     << setId << "/" << shardId << ": " << ex.what();
        return result;
      }
    }
  }
  auto tables = shardTables_.rlock();
  auto it = tables->find(ShardKey{setId, shardId});
  if (it == tables->end()) {
    LOG(WARNING) << "Shard " << setId << "/" << shardId
                 << " not found after lazy construction.";
    return result;
  }

  auto& hashTable = it->second;
  if (!hashTable) {
    return result;
  }
  auto* rowContainer = hashTable->rows();

  auto numProbeRows = probeKeys->size();

  // --- Clone VectorHashers for thread safety ---
  const auto& originalHashers = hashTable->hashers();
  std::vector<std::unique_ptr<VectorHasher>> clonedHashers;
  clonedHashers.reserve(originalHashers.size());
  for (const auto& h : originalHashers) {
    clonedHashers.push_back(VectorHasher::create(h->type(), h->channel()));
  }
  HashLookup hashLookup(clonedHashers);

  auto activeRows = SelectivityVector(numProbeRows);

  hashTable->prepareForJoinProbe(
      hashLookup, probeKeys, activeRows, /*decodeAndRemoveNulls=*/true);

  if (hashLookup.rows.empty()) {
    return result;
  }

  constexpr int32_t kSimdPadding = 16;
  auto hitsSize = numProbeRows + kSimdPadding;
  hashLookup.hits.resize(hitsSize);
  std::fill(
      hashLookup.hits.data(),
      hashLookup.hits.data() + hitsSize,
      nullptr);

  hashTable->joinProbe(hashLookup);

  // --- Collect matched rows ---
  std::vector<int32_t> hits;
  std::vector<char*> matchedRows;

  for (auto row : hashLookup.rows) {
    auto* hit = hashLookup.hits[row];
    if (!hit) {
      continue;
    }
    auto* nextRows = rowContainer->getNextRowVector(hit);
    if (nextRows) {
      for (auto* dupRow : *nextRows) {
        hits.push_back(inputIndices[row]);
        matchedRows.push_back(dupRow);
      }
    } else {
      hits.push_back(inputIndices[row]);
      matchedRows.push_back(hit);
    }
  }

  if (matchedRows.empty()) {
    return result;
  }

  // --- Server-side join condition eval ---
  if (!filterExprJson.empty() && probeFilterColumns && filterInputType) {
    auto cachedFilter = getOrCompileFilter(setId, filterExprJson, filterInputType);
    evaluateFilter(
        *cachedFilter, probeFilterColumns, hits, matchedRows,
        rowContainer, rpcPool);
    if (matchedRows.empty()) {
      return result;
    }
  }

  // --- Batched extract + serialize ---
  const auto& colTypes = rowContainer->columnTypes();
  auto numColumns = colTypes.size();

  // Build RowType once (shared across batches).
  std::vector<std::string> names;
  names.reserve(numColumns);
  for (size_t i = 0; i < numColumns; ++i) {
    names.push_back(fmt::format("c{}", i));
  }
  auto outputType = ROW(
      std::move(names),
      std::vector<TypePtr>(colTypes.begin(), colTypes.end()));

  auto totalMatches = static_cast<vector_size_t>(matchedRows.size());
  constexpr vector_size_t kSerializeBatchSize = 1024;

  // Serialize matched rows in batches.  Each batch is extracted from the
  // RowContainer, serialized to a complete CompactRow wire payload via
  // serializeCompactRowBatch, then the batch RowVector is released.
  // Per-batch payloads are merged at the end with mergeCompactRowWirePayloads.
  //
  // This approach has ⌈N/kSerializeBatchSize⌉ std::string allocations
  // (one per batch) instead of N (one per row), and the final merge is
  // a single memcpy per batch rather than per row.
  auto numBatches =
      (totalMatches + kSerializeBatchSize - 1) / kSerializeBatchSize;
  std::vector<std::string> batchPayloads;
  batchPayloads.reserve(numBatches);

  for (vector_size_t offset = 0; offset < totalMatches;
       offset += kSerializeBatchSize) {
    auto batchSize =
        std::min(kSerializeBatchSize, totalMatches - offset);

    // Extract this batch of rows from RowContainer.
    std::vector<VectorPtr> batchColumns(numColumns);
    for (size_t col = 0; col < numColumns; ++col) {
      batchColumns[col] =
          BaseVector::create(colTypes[col], batchSize, rpcPool);
      rowContainer->extractColumn(
          matchedRows.data() + offset,
          batchSize,
          static_cast<int32_t>(col),
          batchColumns[col]);
    }

    auto batchVector = std::make_shared<RowVector>(
        rpcPool,
        outputType,
        nullptr,
        batchSize,
        std::move(batchColumns));

    // Serialize the entire batch into one wire payload.
    batchPayloads.push_back(
        serializeCompactRowBatch(batchVector, rpcPool));

    // batchVector and batchColumns go out of scope here, releasing
    // the memory allocated from rpcPool for this batch.
  }

  // Merge all batch payloads into a single wire payload.
  if (batchPayloads.size() == 1) {
    // Single batch — no merge needed, avoid the extra copy.
    result.outputBytes = std::move(batchPayloads[0]);
  } else {
    result.outputBytes = mergeCompactRowWirePayloads(batchPayloads);
  }
  result.inputHits = std::move(hits);

  return result;
}

VeloxShardManager::LookupResult VeloxShardManager::lookup(
    int64_t setId,
    int32_t shardId,
    const RowVectorPtr& probeKeys,
    const std::vector<int32_t>& inputIndices,
    memory::MemoryPool* callerPool,
    const RowVectorPtr& probeFilterColumns,
    const std::string& filterExprJson,
    const RowTypePtr& probeColumnsType,
    const RowTypePtr& filterInputType) {
  LookupResult result;

  if (!probeKeys || probeKeys->size() == 0) {
    return result;
  }

  VELOX_CHECK_EQ(
      probeKeys->size(),
      inputIndices.size(),
      "probeKeys row count ({}) must match inputIndices size ({})",
      probeKeys->size(),
      inputIndices.size());

  {
    auto tables = shardTables_.rlock();
    if (tables->find(ShardKey{setId, shardId}) == tables->end()) {
      // Lazy construction: build the HashTable from BlockManager on first
      // lookup if it hasn't been constructed yet (e.g., probe executor in
      // local-cluster mode where build ran on a different JVM process).
      tables.unlock();
      try {
        constructShardTable(setId, shardId);
      } catch (const std::exception& ex) {
        LOG(WARNING) << "Lazy constructShardTable failed for shard "
                     << setId << "/" << shardId << ": " << ex.what();
        return result;
      }
    }
  }
  auto tables = shardTables_.rlock();
  auto it = tables->find(ShardKey{setId, shardId});
  if (it == tables->end()) {
    LOG(WARNING) << "Shard " << setId << "/" << shardId
                 << " not found in HashTable map after lazy construction.";
    return result;
  }

  auto& hashTable = it->second;
  if (!hashTable) {
    // Empty shard (no data rows).  Return empty result.
    return result;
  }
  auto* rowContainer = hashTable->rows();

  // When callerPool is provided, allocate output directly in the caller's
  // pool, eliminating the need for a cross-pool copy in mergeOutputs.
  // When callerPool is null (e.g., gRPC server path), create a temporary
  // leaf pool that the caller must keep alive until output is consumed.
  static std::atomic<uint64_t> lookupCounter{0};
  std::shared_ptr<memory::MemoryPool> ownedPool;
  memory::MemoryPool* allocPool;
  if (callerPool) {
    allocPool = callerPool;
  } else {
    ownedPool = pool_->addLeafChild(
        fmt::format("lookup_{}_{}_{}", setId, shardId, lookupCounter++));
    allocPool = ownedPool.get();
  }

  auto numProbeRows = probeKeys->size();

  // Create per-call VectorHasher clones so that each concurrent lookup
  // has its own DecodedVector state.  This eliminates the per-shard mutex
  // entirely: prepareForJoinProbe calls decode()/hash() on cloned hashers,
  // and joinProbe→compareKeys reads decodedVector() from the same clones.
  // The only shared-state access is hashers_[i]->lookupValueIds() which
  // is a const method (thread-safe) called directly by prepareForJoinProbe
  // in kArray/kNormalizedKey modes.
  const auto& originalHashers = hashTable->hashers();
  std::vector<std::unique_ptr<VectorHasher>> clonedHashers;
  clonedHashers.reserve(originalHashers.size());
  for (const auto& h : originalHashers) {
    clonedHashers.push_back(VectorHasher::create(h->type(), h->channel()));
  }
  HashLookup hashLookup(clonedHashers);

  auto activeRows = SelectivityVector(numProbeRows);

  hashTable->prepareForJoinProbe(
      hashLookup, probeKeys, activeRows, /*decodeAndRemoveNulls=*/true);

  if (hashLookup.rows.empty()) {
    return result;
  }

  // Resize hits to cover all row indices plus SIMD padding.
  // arrayJoinProbe uses SIMD gather/store_unaligned which may write
  // up to kStep (typically 4-8) entries past the last row index.
  // We add extra padding to prevent out-of-bounds writes.
  constexpr int32_t kSimdPadding = 16;
  auto hitsSize = numProbeRows + kSimdPadding;
  hashLookup.hits.resize(hitsSize);
  std::fill(
      hashLookup.hits.data(),
      hashLookup.hits.data() + hitsSize,
      nullptr);

  hashTable->joinProbe(hashLookup);

  // Collect results.  joinProbe sets hits[row] to the first matching
  // build-side row pointer (or nullptr if no match).  When the hash table
  // contains duplicate keys, additional matching rows are stored in a
  // NextRowVector linked from the primary hit row via nextOffset_.
  // We must iterate all duplicates to produce correct results for joins
  // where the build side has multiple rows per key.
  std::vector<int32_t> hits;
  std::vector<char*> matchedRows;

  for (auto row : hashLookup.rows) {
    auto* hit = hashLookup.hits[row];
    if (!hit) {
      continue;
    }
    // NextRowVector already includes the primary hit row (see
    // RowContainer::appendNextRow which pushes `current` into the vector
    // on first duplicate).  So when duplicates exist we iterate only
    // the NextRowVector; when there are none we emit the hit itself.
    auto* nextRows = rowContainer->getNextRowVector(hit);
    if (nextRows) {
      for (auto* dupRow : *nextRows) {
        hits.push_back(inputIndices[row]);
        matchedRows.push_back(dupRow);
      }
    } else {
      hits.push_back(inputIndices[row]);
      matchedRows.push_back(hit);
    }
  }

  if (matchedRows.empty()) {
    return result;
  }

  // --- Server-side join condition eval ---
  if (!filterExprJson.empty() && probeFilterColumns && filterInputType) {
    auto cachedFilter = getOrCompileFilter(setId, filterExprJson, filterInputType);
    evaluateFilter(
        *cachedFilter, probeFilterColumns, hits, matchedRows,
        rowContainer, allocPool);
    if (matchedRows.empty()) {
      return result;
    }
  }

  // Extract matched rows from RowContainer into RowVector using extractColumn.
  auto numMatches = static_cast<vector_size_t>(matchedRows.size());
  const auto& colTypes = rowContainer->columnTypes();
  auto numColumns = colTypes.size();
  std::vector<VectorPtr> outputColumns(numColumns);

  for (size_t col = 0; col < numColumns; ++col) {
    outputColumns[col] =
        BaseVector::create(colTypes[col], numMatches, allocPool);
    rowContainer->extractColumn(
        matchedRows.data(),
        numMatches,
        static_cast<int32_t>(col),
        outputColumns[col]);
  }

  // Build RowType from columnTypes.
  std::vector<std::string> names;
  names.reserve(numColumns);
  for (size_t i = 0; i < numColumns; ++i) {
    names.push_back(fmt::format("c{}", i));
  }
  auto outputType = ROW(
      std::move(names),
      std::vector<TypePtr>(colTypes.begin(), colTypes.end()));
  result.output = std::make_shared<RowVector>(
      allocPool, outputType, nullptr, numMatches, std::move(outputColumns));
  result.inputHits = std::move(hits);
  // Only transfer pool ownership when we created a temporary pool.
  // When callerPool is used, the caller already owns the pool lifetime.
  result.outputPool = std::move(ownedPool);

  return result;
}

// ---------------------------------------------------------------------------
// getOrCompileFilter – compile and cache a filter ExprSet from JSON
// ---------------------------------------------------------------------------
std::shared_ptr<VeloxShardManager::CachedFilter>
VeloxShardManager::getOrCompileFilter(
    int64_t setId,
    const std::string& filterExprJson,
    const RowTypePtr& filterInputType) {
  std::lock_guard<std::mutex> lock(filterCacheMutex_);
  auto it = filterCache_.find(setId);
  if (it != filterCache_.end()) {
    return it->second;
  }

  if (debugEnabled_) {
    LOG(INFO) << "[ShardMgr-dbg] compiling filter ExprSet"
              << " inputType=" << filterInputType->toString()
              << " exprJsonLen=" << filterExprJson.size();
  }

  Type::registerSerDe();
  core::ITypedExpr::registerSerDe();

  auto json = folly::parseJson(filterExprJson);
  static std::atomic<uint64_t> filterPoolCounter{0};
  auto filterPool = pool_->addLeafChild(
      fmt::format("filter_cache_{}", filterPoolCounter++));
  auto queryCtx = core::QueryCtx::create();
  auto execCtx = std::make_unique<core::ExecCtx>(
      filterPool.get(), queryCtx.get());
  auto expr = ISerializable::deserialize<core::ITypedExpr>(
      json, filterPool.get());
  auto exprSet = std::make_unique<exec::ExprSet>(
      std::vector<core::TypedExprPtr>{expr}, execCtx.get());

  auto cached = std::make_shared<CachedFilter>();
  cached->pool = std::move(filterPool);
  cached->queryCtx = std::move(queryCtx);
  cached->execCtx = std::move(execCtx);
  cached->exprSet = std::move(exprSet);
  cached->inputType = filterInputType;

  filterCache_[setId] = cached;
  return cached;
}

// ---------------------------------------------------------------------------
// evaluateFilter – eval join condition on (probe filter cols, build cols)
//
// For each matched row i:
//   - probe columns are expanded from probeFilterColumns[inputHits[i]]
//   - build columns are extracted from rowContainer at matchedRows[i]
// The filter is evaluated on the assembled row, and only rows that pass
// are retained in matchedRows and inputHits (compacted in-place).
// ---------------------------------------------------------------------------
void VeloxShardManager::evaluateFilter(
    CachedFilter& filter,
    const RowVectorPtr& probeFilterColumns,
    std::vector<int32_t>& inputHits,
    std::vector<char*>& matchedRows,
    exec::RowContainer* rowContainer,
    memory::MemoryPool* pool) {
  auto numRows = static_cast<vector_size_t>(matchedRows.size());
  if (numRows == 0) {
    return;
  }

  auto filterStart = debugEnabled_
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};

  auto numProbeCols = probeFilterColumns->type()->size();
  const auto& buildColTypes = rowContainer->columnTypes();
  auto numBuildCols = buildColTypes.size();

  // Build index buffer: for each matched row, the probe row index to use.
  auto indexBuf = AlignedBuffer::allocate<vector_size_t>(numRows, pool);
  auto* indices = indexBuf->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    indices[i] = static_cast<vector_size_t>(inputHits[i]);
  }

  // Expand probe filter columns via DictionaryVector.
  std::vector<VectorPtr> filterInputCols;
  filterInputCols.reserve(numProbeCols + numBuildCols);
  for (size_t col = 0; col < numProbeCols; ++col) {
    filterInputCols.push_back(BaseVector::wrapInDictionary(
        nullptr, indexBuf, numRows, probeFilterColumns->childAt(col)));
  }

  // Extract build columns from RowContainer.
  for (size_t col = 0; col < numBuildCols; ++col) {
    auto buildCol = BaseVector::create(buildColTypes[col], numRows, pool);
    rowContainer->extractColumn(
        matchedRows.data(), numRows, static_cast<int32_t>(col), buildCol);
    filterInputCols.push_back(std::move(buildCol));
  }

  auto filterInput = std::make_shared<RowVector>(
      pool, filter.inputType, nullptr, numRows, std::move(filterInputCols));

  // Evaluate filter expression (serialized access per CachedFilter).
  exec::EvalCtx evalCtx(filter.execCtx.get(), filter.exprSet.get(), filterInput.get());
  auto activeRows = SelectivityVector(numRows);
  std::vector<VectorPtr> filterResult(1);

  {
    std::lock_guard<std::mutex> evalLock(filter.evalMutex);
    filter.exprSet->eval(activeRows, evalCtx, filterResult);
  }

  // Decode boolean result and compact matchedRows + inputHits.
  DecodedVector decoded(*filterResult[0], activeRows);
  vector_size_t passed = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    if (!decoded.isNullAt(i) && decoded.valueAt<bool>(i)) {
      matchedRows[passed] = matchedRows[i];
      inputHits[passed] = inputHits[i];
      ++passed;
    }
  }
  matchedRows.resize(passed);
  inputHits.resize(passed);

  if (debugEnabled_) {
    static std::atomic<uint64_t> evalCount{0};
    auto cnt = ++evalCount;
    if (cnt <= 5 || cnt % 200 == 0) {
      auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - filterStart).count();
      LOG(INFO) << "[ShardMgr-dbg] evaluateFilter#" << cnt
                << " inputRows=" << numRows
                << " passed=" << passed
                << " filtered=" << (numRows - passed)
                << " elapsedUs=" << elapsedUs;
    }
  }
}

// ---------------------------------------------------------------------------
// refreshShardLocations – JNI call to GlutenShardManagerJni.refreshShardLocations
// Uses cached JNI handles (jniGlobalClass_ / jniGlobalModule_ / jniRefreshMethod_)
// populated during initialize().  Falls back to per-call lookup if caching
// failed (e.g. class not on classpath during initialize()).
// Uses thread_local attach: native threads stay attached for their lifetime
// to avoid repeated Attach/Detach overhead on the hot retry path.
// ---------------------------------------------------------------------------
std::vector<std::string> VeloxShardManager::refreshShardLocations(
    int64_t setId, int32_t shardId) {
  if (!javaVm_) {
    // Unit test mode — no JVM available.
    return {};
  }

  // thread_local attach: each native thread attaches once and stays attached.
  // The JVM tolerates AttachCurrentThread on an already-attached thread
  // (it simply returns the existing JNIEnv*), so this is safe even for
  // threads that were already attached by other code paths.
  thread_local JNIEnv* tlsEnv = [this]() -> JNIEnv* {
    JNIEnv* e = nullptr;
    int status = javaVm_->GetEnv(
        reinterpret_cast<void**>(&e), JNI_VERSION_1_8);
    if (status == JNI_EDETACHED) {
      javaVm_->AttachCurrentThread(
          reinterpret_cast<void**>(&e), nullptr);
    }
    return e;
  }();
  JNIEnv* env = tlsEnv;

  // Use cached handles if available; otherwise fall back to per-call lookup.
  jclass jniClass = jniGlobalClass_;
  jobject moduleInstance = jniGlobalModule_;
  jmethodID refreshMethod = jniRefreshMethod_;
  bool usedLocalRefs = false;

  if (!jniClass || !moduleInstance || !refreshMethod) {
    // Fallback: per-call lookup (should be rare after successful initialize).
    jniClass = env->FindClass(
        "org/apache/spark/shard/GlutenShardManagerJni$");
    if (jniClass == nullptr || env->ExceptionCheck()) {
      env->ExceptionClear();
      LOG(WARNING) << "refreshShardLocations: GlutenShardManagerJni$ class "
                      "not found";
      return {};
    }
    jfieldID moduleField = env->GetStaticFieldID(
        jniClass, "MODULE$",
        "Lorg/apache/spark/shard/GlutenShardManagerJni$;");
    moduleInstance = env->GetStaticObjectField(jniClass, moduleField);
    refreshMethod = env->GetMethodID(
        jniClass, "refreshShardLocations",
        "(JI)[Ljava/lang/String;");
    usedLocalRefs = true;
  }

  auto result = static_cast<jobjectArray>(env->CallObjectMethod(
      moduleInstance, refreshMethod,
      static_cast<jlong>(setId), static_cast<jint>(shardId)));

  if (env->ExceptionCheck()) {
    env->ExceptionDescribe();
    env->ExceptionClear();
    if (usedLocalRefs) {
      env->DeleteLocalRef(jniClass);
      env->DeleteLocalRef(moduleInstance);
    }
    return {};
  }

  std::vector<std::string> locations;
  if (result != nullptr) {
    jsize length = env->GetArrayLength(result);
    locations.reserve(length);
    for (jsize i = 0; i < length; ++i) {
      auto jstr = static_cast<jstring>(
          env->GetObjectArrayElement(result, i));
      if (jstr != nullptr) {
        const char* chars = env->GetStringUTFChars(jstr, nullptr);
        locations.emplace_back(chars);
        env->ReleaseStringUTFChars(jstr, chars);
        env->DeleteLocalRef(jstr);
      }
    }
    env->DeleteLocalRef(result);
  }

  if (usedLocalRefs) {
    env->DeleteLocalRef(moduleInstance);
    env->DeleteLocalRef(jniClass);
  }
  // NOTE: no DetachCurrentThread — thread stays attached (thread_local).
  return locations;
}

} // namespace shard
} // namespace gluten
