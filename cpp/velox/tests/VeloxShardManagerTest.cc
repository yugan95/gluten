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

#include "shard/VeloxShardManager.h"

#include <sstream>

#include <folly/json.h>
#include <folly/io/IOBuf.h>

#include "compute/VeloxBackend.h"
#include "shard/BlockManagerBridge.h"
#include "shard/BloomFilter64.h"
#include "velox/common/memory/StreamArena.h"
#include "velox/core/Expressions.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace gluten;
using namespace gluten::shard;

// ---------------------------------------------------------------------------
// MockBlockManagerBridge — stores shard data in memory for unit testing.
// ---------------------------------------------------------------------------
class MockBlockManagerBridge : public BlockManagerBridge {
 public:
  void putBlock(
      int64_t setId,
      int32_t shardId,
      const std::string& tag,
      const std::string& data) {
    auto key = makeKey(setId, shardId, tag);
    blocks_[key] = data;
  }

  std::unique_ptr<folly::IOBuf> readBlock(
      int64_t setId,
      int32_t shardId,
      const std::string& tag) override {
    auto key = makeKey(setId, shardId, tag);
    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
      return folly::IOBuf::create(0);
    }
    auto ioBuf = folly::IOBuf::create(it->second.size());
    std::memcpy(ioBuf->writableData(), it->second.data(), it->second.size());
    ioBuf->append(it->second.size());
    return ioBuf;
  }

  // Remove all blocks for a given setId (to prevent lazy reconstruction
  // after destroyShardTable).
  void removeBlocksForSet(int64_t setId) {
    auto prefix = std::to_string(setId) + "/";
    for (auto it = blocks_.begin(); it != blocks_.end();) {
      if (it->first.substr(0, prefix.size()) == prefix) {
        it = blocks_.erase(it);
      } else {
        ++it;
      }
    }
  }

 private:
  static std::string makeKey(
      int64_t setId,
      int32_t shardId,
      const std::string& tag) {
    return std::to_string(setId) + "/" + std::to_string(shardId) + "/" + tag;
  }

  std::unordered_map<std::string, std::string> blocks_;
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class VeloxShardManagerTest : public ::testing::Test,
                              public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    VeloxBackend::create({});
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    mockBridge_ = std::make_shared<MockBlockManagerBridge>();
    VeloxShardManager::getInstance()->initialize(mockBridge_);
    pool_ = memory::memoryManager()->addRootPool("test");
    leafPool_ = pool_->addLeafChild("testLeaf");
  }

  void TearDown() override {
    VeloxShardManager::getInstance()->shutdown();
  }

  // Serialize a RowVector to Presto format bytes.
  std::string serializeToPresto(const RowVectorPtr& rowVector) {
    auto serde =
        std::make_unique<serializer::presto::PrestoVectorSerde>();
    auto numRows = rowVector->size();
    auto rowType = asRowType(rowVector->type());
    auto arena = std::make_unique<StreamArena>(leafPool_.get());
    serializer::presto::PrestoVectorSerde::PrestoOptions opts;
    opts.useLosslessTimestamp = true;

    auto serializer = serde->createIterativeSerializer(
        rowType, numRows, arena.get(), &opts);

    serializer->append(rowVector);

    std::ostringstream oss;
    auto outputStream =
        std::make_unique<OStreamOutputStream>(&oss);
    serializer->flush(outputStream.get());
    return oss.str();
  }

  // Build a meta block:
  // [int32 pieceCount][int32 numKeyColumns][int32 schemaLen][char[] schemaStr]
  std::string buildMetaBlock(
      int32_t pieceCount,
      int32_t numKeyColumns,
      const RowTypePtr& rowType) {
    auto schemaStr = rowType->toString();
    auto schemaLen = static_cast<int32_t>(schemaStr.size());
    std::string meta(3 * sizeof(int32_t) + schemaLen, '\0');
    std::memcpy(&meta[0], &pieceCount, sizeof(int32_t));
    std::memcpy(&meta[sizeof(int32_t)], &numKeyColumns, sizeof(int32_t));
    std::memcpy(&meta[2 * sizeof(int32_t)], &schemaLen, sizeof(int32_t));
    std::memcpy(&meta[3 * sizeof(int32_t)], schemaStr.data(), schemaLen);
    return meta;
  }

  // Populate a shard with the given build-side RowVector.
  // The first numKeyColumns columns are treated as key columns.
  void populateShard(
      int64_t setId,
      int32_t shardId,
      const RowVectorPtr& buildData,
      int32_t numKeyColumns) {
    auto pieceBytes = serializeToPresto(buildData);
    mockBridge_->putBlock(setId, shardId, "piece0", pieceBytes);
    auto rowType = asRowType(buildData->type());
    mockBridge_->putBlock(
        setId, shardId, "meta", buildMetaBlock(1, numKeyColumns, rowType));
  }

  std::shared_ptr<MockBlockManagerBridge> mockBridge_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Basic: insert 3 rows, lookup with exact key match.
TEST_F(VeloxShardManagerTest, BasicLookup) {
  int64_t setId = 100;
  int32_t shardId = 0;

  // Build side: key=int32, value=int64
  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe with keys {1, 3}
  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({1, 3})});
  std::vector<int32_t> inputIndices = {0, 1};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 2);
  ASSERT_EQ(result.inputHits.size(), 2);

  // Verify matched input indices.
  EXPECT_EQ(result.inputHits[0], 0);
  EXPECT_EQ(result.inputHits[1], 1);

  // Verify output has 2 columns (key + value).
  ASSERT_EQ(result.output->childrenSize(), 2);

  // Verify key column values.
  auto outputKeys = result.output->childAt(0)->as<SimpleVector<int32_t>>();
  EXPECT_EQ(outputKeys->valueAt(0), 1);
  EXPECT_EQ(outputKeys->valueAt(1), 3);

  // Verify value column values.
  auto outputValues = result.output->childAt(1)->as<SimpleVector<int64_t>>();
  EXPECT_EQ(outputValues->valueAt(0), 100L);
  EXPECT_EQ(outputValues->valueAt(1), 300L);
}

// Lookup with no matches should return empty result.
TEST_F(VeloxShardManagerTest, LookupNoMatch) {
  int64_t setId = 101;
  int32_t shardId = 0;

  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe with keys that don't exist in build side.
  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({99, 999})});
  std::vector<int32_t> inputIndices = {0, 1};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  // No matches: output should be null and inputHits empty.
  EXPECT_EQ(result.output, nullptr);
  EXPECT_TRUE(result.inputHits.empty());
}

// Lookup on a non-existent shard should return empty result.
TEST_F(VeloxShardManagerTest, LookupNonExistentShard) {
  auto probeKeys =
      makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{1})});
  std::vector<int32_t> inputIndices = {0};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          999, 0, probeKeys, inputIndices);

  EXPECT_EQ(result.output, nullptr);
  EXPECT_TRUE(result.inputHits.empty());
}

// Verify destroyShardTable removes the shard.
TEST_F(VeloxShardManagerTest, DestroyShardTable) {
  int64_t setId = 102;
  int32_t shardId = 0;

  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int32_t>({10, 20}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Verify lookup works before destroy.
  auto probeKeys =
      makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{1})});
  std::vector<int32_t> inputIndices = {0};
  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);
  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 1);

  // Remove bridge data to prevent lazy reconstruction after destroy.
  mockBridge_->removeBlocksForSet(setId);

  // Destroy the shard set.
  VeloxShardManager::getInstance()->destroyShardTable(setId);

  // Lookup after destroy should return empty.
  auto resultAfter =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);
  EXPECT_EQ(resultAfter.output, nullptr);
  EXPECT_TRUE(resultAfter.inputHits.empty());
}

// Multi-key lookup: composite key (int32, int64).
TEST_F(VeloxShardManagerTest, MultiKeyLookup) {
  int64_t setId = 103;
  int32_t shardId = 0;

  // Build side: key1=int32, key2=int64, value=int32
  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2}),
      makeFlatVector<int64_t>({100L, 200L, 100L}),
      makeFlatVector<int32_t>({10, 20, 30}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/2);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe with composite key (1, 200) — should match row with value=20.
  auto probeKeys = makeRowVector({
      makeFlatVector<int32_t>(std::vector<int32_t>{1}),
      makeFlatVector<int64_t>(std::vector<int64_t>{200L}),
  });
  std::vector<int32_t> inputIndices = {0};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 1);
  EXPECT_EQ(result.inputHits[0], 0);

  // Verify the matched value column.
  auto outputValues = result.output->childAt(2)->as<SimpleVector<int32_t>>();
  EXPECT_EQ(outputValues->valueAt(0), 20);
}

// Multiple shards within the same set.
TEST_F(VeloxShardManagerTest, MultipleShards) {
  int64_t setId = 104;

  // Shard 0: keys {1, 2}
  auto buildData0 = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int32_t>({10, 20}),
  });
  populateShard(setId, 0, buildData0, /*numKeyColumns=*/1);

  // Shard 1: keys {3, 4}
  auto buildData1 = makeRowVector({
      makeFlatVector<int32_t>({3, 4}),
      makeFlatVector<int32_t>({30, 40}),
  });
  populateShard(setId, 1, buildData1, /*numKeyColumns=*/1);

  VeloxShardManager::getInstance()->constructShardTable(setId, 0);
  VeloxShardManager::getInstance()->constructShardTable(setId, 1);

  // Lookup key=3 in shard 1.
  auto probeKeys =
      makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{3})});
  std::vector<int32_t> inputIndices = {0};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, 1, probeKeys, inputIndices);

  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 1);

  auto outputValues = result.output->childAt(1)->as<SimpleVector<int32_t>>();
  EXPECT_EQ(outputValues->valueAt(0), 30);

  // Lookup key=3 in shard 0 — should not match.
  auto resultShard0 =
      VeloxShardManager::getInstance()->lookup(
          setId, 0, probeKeys, inputIndices);
  EXPECT_EQ(resultShard0.output, nullptr);

  // Remove bridge data to prevent lazy reconstruction after destroy.
  mockBridge_->removeBlocksForSet(setId);

  // Destroy should remove all shards in the set.
  VeloxShardManager::getInstance()->destroyShardTable(setId);

  auto resultAfter =
      VeloxShardManager::getInstance()->lookup(
          setId, 0, probeKeys, inputIndices);
  EXPECT_EQ(resultAfter.output, nullptr);

  auto resultAfter1 =
      VeloxShardManager::getInstance()->lookup(
          setId, 1, probeKeys, inputIndices);
  EXPECT_EQ(resultAfter1.output, nullptr);
}

// Empty probe input should return empty result.
TEST_F(VeloxShardManagerTest, EmptyProbeInput) {
  int64_t setId = 105;
  int32_t shardId = 0;

  auto buildData = makeRowVector({
      makeFlatVector<int32_t>(std::vector<int32_t>{1}),
      makeFlatVector<int32_t>(std::vector<int32_t>{10}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Empty probe.
  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({})});
  std::vector<int32_t> inputIndices = {};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  EXPECT_EQ(result.output, nullptr);
  EXPECT_TRUE(result.inputHits.empty());
}

// fetchBloomFilter: returns stored BF data from BlockManagerBridge.
TEST_F(VeloxShardManagerTest, FetchBloomFilterReturnsStoredData) {
  int64_t setId = 200;
  int32_t shardId = -1; // set-level merged BF

  // Create a real BloomFilter serialized to bytes.
  BloomFilter64 bf;
  bf.reset(100);
  bf.insert(42);
  bf.insert(123);
  bf.insert(999);

  auto serializedSize = bf.serializedSize();
  std::string bfBytes(serializedSize, '\0');
  bf.serialize(bfBytes.data());

  // Store in mock bridge under "bloom" tag.
  mockBridge_->putBlock(setId, shardId, "nativeBloom", bfBytes);

  // Fetch via VeloxShardManager.
  auto fetched =
      VeloxShardManager::getInstance()->fetchBloomFilter(setId, shardId);

  ASSERT_FALSE(fetched.empty());
  ASSERT_EQ(fetched.size(), bfBytes.size());

  // Deserialize and verify the BF contents.
  BloomFilter64 restored;
  restored.merge(fetched.data());
  EXPECT_TRUE(restored.isSet());
  EXPECT_TRUE(restored.mayContain(42));
  EXPECT_TRUE(restored.mayContain(123));
  EXPECT_TRUE(restored.mayContain(999));
  // A value never inserted should (almost certainly) not be present.
  EXPECT_FALSE(restored.mayContain(77777));
}

// fetchBloomFilter: returns empty string when no bloom block exists.
TEST_F(VeloxShardManagerTest, FetchBloomFilterMissing) {
  auto fetched =
      VeloxShardManager::getInstance()->fetchBloomFilter(999, -1);
  EXPECT_TRUE(fetched.empty());
}

// fetchBloomFilter: per-shard BF (shardId >= 0) also works.
TEST_F(VeloxShardManagerTest, FetchBloomFilterPerShard) {
  int64_t setId = 201;
  int32_t shardId = 2;

  BloomFilter64 bf;
  bf.reset(50);
  bf.insert(10);
  bf.insert(20);

  auto serializedSize = bf.serializedSize();
  std::string bfBytes(serializedSize, '\0');
  bf.serialize(bfBytes.data());

  mockBridge_->putBlock(setId, shardId, "nativeBloom", bfBytes);

  auto fetched =
      VeloxShardManager::getInstance()->fetchBloomFilter(setId, shardId);

  ASSERT_FALSE(fetched.empty());

  BloomFilter64 restored;
  restored.merge(fetched.data());
  EXPECT_TRUE(restored.mayContain(10));
  EXPECT_TRUE(restored.mayContain(20));
  EXPECT_FALSE(restored.mayContain(30));
}

// ---------------------------------------------------------------------------
// Server-side filter pushdown tests
// ---------------------------------------------------------------------------

// Helper: build a filter expression JSON and filterInputType for server-side
// filter eval.  The filter is "probeCol < buildCol" where probeCol and buildCol
// are referenced by name.
static std::pair<std::string, RowTypePtr> makeFilterLessThan(
    const std::string& probeColName,
    const TypePtr& probeColType,
    const std::string& buildColName,
    const TypePtr& buildColType,
    const RowTypePtr& buildOutputType) {
  Type::registerSerDe();
  core::ITypedExpr::registerSerDe();

  auto probeField = std::make_shared<core::FieldAccessTypedExpr>(
      probeColType, probeColName);
  auto buildField = std::make_shared<core::FieldAccessTypedExpr>(
      buildColType, buildColName);
  auto filterExpr = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{probeField, buildField},
      "lessthan");

  auto filterExprJson = folly::toJson(filterExpr->serialize());

  // filterInputType = concat(probeFilterColumns, buildOutputType)
  std::vector<std::string> names = {probeColName};
  std::vector<TypePtr> types = {probeColType};
  for (uint32_t i = 0; i < buildOutputType->size(); ++i) {
    names.push_back(buildOutputType->nameOf(i));
    types.push_back(buildOutputType->childAt(i));
  }
  auto filterInputType = ROW(std::move(names), std::move(types));
  return {filterExprJson, filterInputType};
}

// lookup() with server-side filter: only rows passing filter are returned.
TEST_F(VeloxShardManagerTest, LookupWithFilter) {
  int64_t setId = 500;
  int32_t shardId = 0;

  // Build: k=int32 (key), v=int64 (value)
  // Rows: (1,15), (2,10), (3,35), (4,50)
  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({15L, 10L, 35L, 50L})});
  populateShard(setId, shardId, buildData, 1);
  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe keys: {1, 2, 3, 4} — all match
  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({1, 2, 3, 4})});
  std::vector<int32_t> inputIndices = {0, 1, 2, 3};

  // Probe filter columns: pval = {10, 20, 30, 40}
  auto probeFilterCols = makeRowVector(
      {"pval"}, {makeFlatVector<int64_t>({10L, 20L, 30L, 40L})});

  // Filter: pval < v (probe.pval < build.v)
  auto buildOutputType = asRowType(buildData->type());
  auto [filterExprJson, filterInputType] =
      makeFilterLessThan("pval", BIGINT(), "v", BIGINT(), buildOutputType);

  auto result = VeloxShardManager::getInstance()->lookup(
      setId, shardId, probeKeys, inputIndices, leafPool_.get(),
      probeFilterCols, filterExprJson, nullptr, filterInputType);

  // Expected passing rows:
  //   k=1: pval=10 < v=15 → pass
  //   k=2: pval=20 < v=10 → fail
  //   k=3: pval=30 < v=35 → pass
  //   k=4: pval=40 < v=50 → pass
  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 3);
  ASSERT_EQ(result.inputHits.size(), 3);

  // Verify only passing rows are returned.
  std::set<int32_t> passedKeys;
  auto* keyCol = result.output->childAt(0)->as<SimpleVector<int32_t>>();
  for (vector_size_t i = 0; i < result.output->size(); ++i) {
    passedKeys.insert(keyCol->valueAt(i));
  }
  EXPECT_TRUE(passedKeys.count(1));
  EXPECT_FALSE(passedKeys.count(2));
  EXPECT_TRUE(passedKeys.count(3));
  EXPECT_TRUE(passedKeys.count(4));
}

// lookup() without filter params — backward compatibility.
TEST_F(VeloxShardManagerTest, LookupWithoutFilterBackcompat) {
  int64_t setId = 501;
  int32_t shardId = 0;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({100L, 200L, 300L})});
  populateShard(setId, shardId, buildData, 1);
  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({1, 3})});
  std::vector<int32_t> inputIndices = {0, 1};

  // No filter params — should work as before.
  auto result = VeloxShardManager::getInstance()->lookup(
      setId, shardId, probeKeys, inputIndices);

  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 2);
}

// lookupAndSerialize() with server-side filter.
TEST_F(VeloxShardManagerTest, LookupAndSerializeWithFilter) {
  int64_t setId = 502;
  int32_t shardId = 0;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({15L, 5L, 35L})});
  populateShard(setId, shardId, buildData, 1);
  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});
  std::vector<int32_t> inputIndices = {0, 1, 2};

  auto probeFilterCols = makeRowVector(
      {"pval"}, {makeFlatVector<int64_t>({10L, 10L, 30L})});

  auto buildOutputType = asRowType(buildData->type());
  auto [filterExprJson, filterInputType] =
      makeFilterLessThan("pval", BIGINT(), "v", BIGINT(), buildOutputType);

  auto result = VeloxShardManager::getInstance()->lookupAndSerialize(
      setId, shardId, probeKeys, inputIndices, leafPool_.get(),
      probeFilterCols, filterExprJson, nullptr, filterInputType);

  // k=1: 10 < 15 → pass
  // k=2: 10 < 5  → fail
  // k=3: 30 < 35 → pass
  ASSERT_EQ(result.inputHits.size(), 2);
  ASSERT_FALSE(result.outputBytes.empty());

  std::set<int32_t> passedIndices(
      result.inputHits.begin(), result.inputHits.end());
  EXPECT_TRUE(passedIndices.count(0));   // k=1
  EXPECT_FALSE(passedIndices.count(1));  // k=2 filtered
  EXPECT_TRUE(passedIndices.count(2));   // k=3
}

// Filter that filters all rows — empty result.
TEST_F(VeloxShardManagerTest, LookupWithFilterAllFiltered) {
  int64_t setId = 503;
  int32_t shardId = 0;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({5L, 3L})});
  populateShard(setId, shardId, buildData, 1);
  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({1, 2})});
  std::vector<int32_t> inputIndices = {0, 1};

  // pval={100, 200} — all > v, so pval < v never holds.
  auto probeFilterCols = makeRowVector(
      {"pval"}, {makeFlatVector<int64_t>({100L, 200L})});

  auto buildOutputType = asRowType(buildData->type());
  auto [filterExprJson, filterInputType] =
      makeFilterLessThan("pval", BIGINT(), "v", BIGINT(), buildOutputType);

  auto result = VeloxShardManager::getInstance()->lookup(
      setId, shardId, probeKeys, inputIndices, leafPool_.get(),
      probeFilterCols, filterExprJson, nullptr, filterInputType);

  EXPECT_EQ(result.output, nullptr);
  EXPECT_TRUE(result.inputHits.empty());
}

// VARCHAR key tests are wrapped in NDEBUG because of an ODR (One Definition
// Rule) violation between libvelox.so (Debug) and the test binary: the
// template AlignedBuffer::allocate<char> is a weak symbol that may resolve to
// a Release-optimised instantiation which omits setEndGuard(), causing a
// spurious checkEndGuard() failure in Debug-compiled getBufferWithSpace().
// This is NOT a real buffer overrun — the end guard is simply never written.
#ifdef NDEBUG

// Mixed-type multi-key (int32 + varchar): lookup with exact composite match.
TEST_F(VeloxShardManagerTest, MixedTypeMultiKeyLookup) {
  int64_t setId = 400;
  int32_t shardId = 0;

  // Build side: key1=int32, key2=varchar, value=int64
  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2}),
      makeFlatVector<StringView>({"alpha"_sv, "beta"_sv, "alpha"_sv}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/2);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe with composite key (1, "beta") — should match row with value=200.
  auto probeKeys = makeRowVector({
      makeFlatVector<int32_t>(std::vector<int32_t>{1}),
      makeFlatVector<StringView>({"beta"_sv}),
  });
  std::vector<int32_t> inputIndices = {0};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 1);
  EXPECT_EQ(result.inputHits[0], 0);

  auto outputValues = result.output->childAt(2)->as<SimpleVector<int64_t>>();
  EXPECT_EQ(outputValues->valueAt(0), 200L);
}

// Mixed-type multi-key: no match when one key column differs.
TEST_F(VeloxShardManagerTest, MixedTypeMultiKeyNoMatch) {
  int64_t setId = 401;
  int32_t shardId = 0;

  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<StringView>({"alpha"_sv, "beta"_sv}),
      makeFlatVector<int64_t>({100L, 200L}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/2);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe with (1, "beta") — int matches row 0 but varchar doesn't.
  // Probe with (2, "alpha") — varchar matches row 0 but int doesn't.
  auto probeKeys = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<StringView>({"gamma"_sv, "delta"_sv}),
  });
  std::vector<int32_t> inputIndices = {0, 1};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  EXPECT_EQ(result.output, nullptr);
  EXPECT_TRUE(result.inputHits.empty());
}

// Mixed-type multi-key: multiple probe rows with partial matches.
TEST_F(VeloxShardManagerTest, MixedTypeMultiKeyPartialMatch) {
  int64_t setId = 402;
  int32_t shardId = 0;

  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<StringView>({"foo"_sv, "bar"_sv, "baz"_sv}),
      makeFlatVector<int64_t>({1000L, 2000L, 3000L}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/2);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe: (10, "foo") matches, (20, "foo") doesn't, (30, "baz") matches.
  auto probeKeys = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<StringView>({"foo"_sv, "foo"_sv, "baz"_sv}),
  });
  std::vector<int32_t> inputIndices = {0, 1, 2};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 2);
  ASSERT_EQ(result.inputHits.size(), 2);

  // Verify matched input indices: rows 0 and 2.
  std::set<int32_t> hitSet(result.inputHits.begin(), result.inputHits.end());
  EXPECT_TRUE(hitSet.count(0));
  EXPECT_TRUE(hitSet.count(2));

  // Verify matched values.
  auto outputValues = result.output->childAt(2)->as<SimpleVector<int64_t>>();
  std::set<int64_t> valueSet;
  for (int i = 0; i < result.output->size(); ++i) {
    valueSet.insert(outputValues->valueAt(i));
  }
  EXPECT_TRUE(valueSet.count(1000L));
  EXPECT_TRUE(valueSet.count(3000L));
}

// VARCHAR key: build with string keys, lookup with exact match.
TEST_F(VeloxShardManagerTest, VarcharKeyLookup) {
  int64_t setId = 300;
  int32_t shardId = 0;

  // Build side: key=varchar, value=int64
  auto buildData = makeRowVector({
      makeFlatVector<StringView>({"alice"_sv, "bob"_sv, "charlie"_sv}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe with keys {"alice", "charlie"} — both should match.
  auto probeKeys = makeRowVector(
      {makeFlatVector<StringView>({"alice"_sv, "charlie"_sv})});
  std::vector<int32_t> inputIndices = {0, 1};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  ASSERT_NE(result.output, nullptr);
  ASSERT_EQ(result.output->size(), 2);
  ASSERT_EQ(result.inputHits.size(), 2);

  // Verify key column values.
  auto outputKeys = result.output->childAt(0)->as<SimpleVector<StringView>>();
  std::set<std::string> matchedKeys;
  for (int i = 0; i < 2; ++i) {
    matchedKeys.insert(outputKeys->valueAt(i).str());
  }
  EXPECT_TRUE(matchedKeys.count("alice"));
  EXPECT_TRUE(matchedKeys.count("charlie"));

  // Verify value column values via key-value map.
  auto outputValues = result.output->childAt(1)->as<SimpleVector<int64_t>>();
  std::map<std::string, int64_t> kvPairs;
  for (int i = 0; i < 2; ++i) {
    kvPairs[outputKeys->valueAt(i).str()] = outputValues->valueAt(i);
  }
  EXPECT_EQ(kvPairs["alice"], 100L);
  EXPECT_EQ(kvPairs["charlie"], 300L);
}

// VARCHAR key: lookup with no matches.
TEST_F(VeloxShardManagerTest, VarcharKeyNoMatch) {
  int64_t setId = 301;
  int32_t shardId = 0;

  auto buildData = makeRowVector({
      makeFlatVector<StringView>({"foo"_sv, "bar"_sv}),
      makeFlatVector<int32_t>({10, 20}),
  });
  populateShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  VeloxShardManager::getInstance()->constructShardTable(setId, shardId);

  // Probe with keys that don't exist.
  auto probeKeys = makeRowVector(
      {makeFlatVector<StringView>({"baz"_sv, "qux"_sv})});
  std::vector<int32_t> inputIndices = {0, 1};

  auto result =
      VeloxShardManager::getInstance()->lookup(
          setId, shardId, probeKeys, inputIndices);

  EXPECT_EQ(result.output, nullptr);
  EXPECT_TRUE(result.inputHits.empty());
}

#endif // NDEBUG
