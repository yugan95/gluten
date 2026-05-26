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

#include "shard/ShardLookupJoin.h"
#include "shard/ShardLookupJoinNode.h"

#include <sstream>

#include <folly/io/IOBuf.h>

#include "compute/VeloxBackend.h"
#include "shard/BlockManagerBridge.h"
#include "shard/SparkMurmurHash.h"
#include "shard/VeloxShardManager.h"
#include "velox/common/memory/StreamArena.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace gluten;
using namespace gluten::shard;

namespace {

// ---------------------------------------------------------------------------
// MockBlockManagerBridge — stores shard data in memory for unit testing.
// (Same pattern as VeloxShardManagerTest.cc)
// ---------------------------------------------------------------------------
class MockBlockManagerBridge : public BlockManagerBridge {
 public:
  void putBlock(
      int64_t setId,
      int32_t shardId,
      const std::string& tag,
      const std::string& data) {
    blocks_[makeKey(setId, shardId, tag)] = data;
  }

  std::unique_ptr<folly::IOBuf> readBlock(
      int64_t setId,
      int32_t shardId,
      const std::string& tag) override {
    auto it = blocks_.find(makeKey(setId, shardId, tag));
    if (it == blocks_.end()) {
      return folly::IOBuf::create(0);
    }
    auto ioBuf = folly::IOBuf::create(it->second.size());
    std::memcpy(ioBuf->writableData(), it->second.data(), it->second.size());
    ioBuf->append(it->second.size());
    return ioBuf;
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

} // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class ShardLookupJoinTest : public ::testing::Test,
                            public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    VeloxBackend::create({});
    memory::MemoryManager::testingSetInstance({});
    // Register the ShardLookupJoinTranslator once per test process so that
    // the LocalPlanner can convert ShardLookupJoinNode to a ShardLookupJoin
    // Operator.  Idempotent — registering twice is harmless.
    static std::once_flag flag;
    std::call_once(flag, []() {
      facebook::velox::exec::Operator::registerOperator(
          std::make_unique<ShardLookupJoinTranslator>());
    });
  }

  void SetUp() override {
    mockBridge_ = std::make_shared<MockBlockManagerBridge>();
    VeloxShardManager::getInstance()->initialize(mockBridge_);
  }

  void TearDown() override {
    VeloxShardManager::getInstance()->shutdown();
  }

  // Serialize a RowVector to Presto format bytes (build-side wire format).
  std::string serializeToPresto(const RowVectorPtr& rowVector) {
    auto serde = std::make_unique<serializer::presto::PrestoVectorSerde>();
    auto numRows = rowVector->size();
    auto rowType = asRowType(rowVector->type());
    auto arena = std::make_unique<StreamArena>(pool_.get());
    serializer::presto::PrestoVectorSerde::PrestoOptions opts;
    opts.useLosslessTimestamp = true;

    auto serializer = serde->createIterativeSerializer(
        rowType, numRows, arena.get(), &opts);
    serializer->append(rowVector);

    std::ostringstream oss;
    auto outputStream = std::make_unique<OStreamOutputStream>(&oss);
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

  // Populate one shard with the given build-side data. The first
  // numKeyColumns columns are the join keys.
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
    VeloxShardManager::getInstance()->constructShardTable(setId, shardId);
  }

  // Compute Spark Murmur3 shard id for a single int32 key (matches the
  // routing logic used by ShardLookupJoin::computeShardId).
  static int32_t computeShardIdInt32(int32_t key, int32_t numShards) {
    int32_t hash = SparkMurmurHash::hashInt(key, 42);
    int32_t shard = hash % numShards;
    return shard < 0 ? shard + numShards : shard;
  }

  // Compose probe + build output type, dropping the right-side key columns
  // (they're equal to the left-side keys for equi-joins, no need to
  // duplicate).
  static RowTypePtr makeOutputType(
      const RowTypePtr& probeType,
      const RowTypePtr& buildType,
      const std::vector<std::string>& rightKeyNames) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (uint32_t i = 0; i < probeType->size(); ++i) {
      names.emplace_back(probeType->nameOf(i));
      types.emplace_back(probeType->childAt(i));
    }
    std::unordered_set<std::string> rightKeys(
        rightKeyNames.begin(), rightKeyNames.end());
    for (uint32_t i = 0; i < buildType->size(); ++i) {
      const auto& name = buildType->nameOf(i);
      if (rightKeys.count(name) > 0) {
        continue;
      }
      names.emplace_back(name);
      types.emplace_back(buildType->childAt(i));
    }
    return ROW(std::move(names), std::move(types));
  }

  // Build a complete plan: ValuesNode (probe) → ShardLookupJoinNode.
  // Avoids PlanBuilder to side-step the parse-library link dependency.
  core::PlanNodePtr makeJoinPlan(
      const std::vector<RowVectorPtr>& probeBatches,
      core::JoinType joinType,
      const std::vector<std::string>& leftKeyNames,
      const std::vector<std::string>& rightKeyNames,
      const RowTypePtr& buildOutputType,
      int64_t shardSetId,
      int32_t numShards,
      int32_t maxBatchSize = 1024,
      int32_t maxInflightRpcs = 4,
      core::TypedExprPtr filter = nullptr) {
    auto probeNode =
        std::make_shared<core::ValuesNode>("0", probeBatches);
    return makeShardLookupJoinNode(
        "1",
        joinType,
        leftKeyNames,
        rightKeyNames,
        probeNode,
        buildOutputType,
        shardSetId,
        numShards,
        maxBatchSize,
        maxInflightRpcs,
        filter);
  }

  std::shared_ptr<ShardLookupJoinNode> makeShardLookupJoinNode(
      const core::PlanNodeId& id,
      core::JoinType joinType,
      const std::vector<std::string>& leftKeyNames,
      const std::vector<std::string>& rightKeyNames,
      const core::PlanNodePtr& probeChild,
      const RowTypePtr& buildOutputType,
      int64_t shardSetId,
      int32_t numShards,
      int32_t maxBatchSize,
      int32_t maxInflightRpcs,
      core::TypedExprPtr filter = nullptr) {
    std::vector<core::FieldAccessTypedExprPtr> leftKeys;
    leftKeys.reserve(leftKeyNames.size());
    for (const auto& name : leftKeyNames) {
      auto idx = probeChild->outputType()->getChildIdx(name);
      leftKeys.emplace_back(std::make_shared<core::FieldAccessTypedExpr>(
          probeChild->outputType()->childAt(idx), name));
    }
    std::vector<core::FieldAccessTypedExprPtr> rightKeys;
    rightKeys.reserve(rightKeyNames.size());
    for (const auto& name : rightKeyNames) {
      auto idx = buildOutputType->getChildIdx(name);
      rightKeys.emplace_back(std::make_shared<core::FieldAccessTypedExpr>(
          buildOutputType->childAt(idx), name));
    }

    auto outputType =
        makeOutputType(probeChild->outputType(), buildOutputType, rightKeyNames);

    std::unordered_map<int32_t, std::vector<ShardServerLocation>>
        shardLocationMap;
    for (int32_t s = 0; s < numShards; ++s) {
      shardLocationMap[s] = {}; // empty == local lookup
    }

    return std::make_shared<ShardLookupJoinNode>(
        id,
        joinType,
        std::move(leftKeys),
        std::move(rightKeys),
        probeChild,
        buildOutputType,
        outputType,
        shardSetId,
        numShards,
        std::move(shardLocationMap),
        maxInflightRpcs,
        maxBatchSize,
        /*hashKeyType=*/nullptr,
        filter);
  }

  std::shared_ptr<MockBlockManagerBridge> mockBridge_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Sanity: fixture initializes manager and translator successfully.
TEST_F(ShardLookupJoinTest, FixtureSetup) {
  EXPECT_NE(VeloxShardManager::getInstance(), nullptr);
}

// Inner Join, single shard, single probe batch — the simplest happy path.
// Probe keys {1,2,3,4}; build {1→100, 2→200, 5→500}; expected {1→100, 2→200}.
TEST_F(ShardLookupJoinTest, InnerJoinSingleShardSingleBatch) {
  const int64_t setId = 1;
  const int32_t numShards = 1;

  // Build side: key + value column.
  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2, 5}),
       makeFlatVector<int64_t>({100L, 200L, 500L})});
  populateShard(setId, /*shardId=*/0, buildData, /*numKeyColumns=*/1);

  // Probe side: keys 1..4 — only 1 and 2 should match.
  auto probeData = makeRowVector(
      {"pk"}, {makeFlatVector<int32_t>({1, 2, 3, 4})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      /*leftKeyNames=*/{"pk"},
      /*rightKeyNames=*/{"k"},
      asRowType(buildData->type()),
      setId,
      numShards);

  // Expected: probe rows that matched + their build values, key column from
  // the probe side ('pk').
  auto expected = makeRowVector(
      {"pk", "v"},
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({100L, 200L})});

  AssertQueryBuilder(plan).assertResults(expected);
}

// Inner Join with NO matches — output should be empty.
TEST_F(ShardLookupJoinTest, InnerJoinNoMatches) {
  const int64_t setId = 2;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({100, 200}),
       makeFlatVector<int64_t>({1000L, 2000L})});
  populateShard(setId, 0, buildData, 1);

  // Probe keys that don't exist in build side.
  auto probeData =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>({1, 2, 3})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards);

  AssertQueryBuilder(plan).assertEmptyResults();
}

// Left Join, single shard — unmatched probe rows should produce null-filled
// build columns.
TEST_F(ShardLookupJoinTest, LeftJoinSingleShardWithUnmatched) {
  const int64_t setId = 3;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 3}),
       makeFlatVector<int64_t>({100L, 300L})});
  populateShard(setId, 0, buildData, 1);

  // Probe keys: 1 (match), 2 (no match), 3 (match), 4 (no match).
  auto probeData =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>({1, 2, 3, 4})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kLeft,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards);

  // Left Join semantics: every probe row appears once; unmatched rows have
  // null build columns.  Output order is per-shard completion (single shard
  // here, so order matches input order).
  auto expected = makeRowVector(
      {"pk", "v"},
      {makeFlatVector<int32_t>({1, 3, 2, 4}),
       makeNullableFlatVector<int64_t>(
           {100L, 300L, std::nullopt, std::nullopt})});

  AssertQueryBuilder(plan).assertResults(expected);
}

// Inner Join across multiple shards.  Probe rows are routed to different
// shards via Spark Murmur3 hash, then results are joined per-shard.
TEST_F(ShardLookupJoinTest, InnerJoinMultipleShards) {
  const int64_t setId = 4;
  const int32_t numShards = 4;

  // Build side: 12 keys spread across the shards.  Distribute by hash so
  // each shard gets its expected keys.
  std::vector<std::vector<int32_t>> keysPerShard(numShards);
  std::vector<std::vector<int64_t>> valuesPerShard(numShards);
  for (int32_t k = 1; k <= 12; ++k) {
    int32_t s = computeShardIdInt32(k, numShards);
    keysPerShard[s].push_back(k);
    valuesPerShard[s].push_back(static_cast<int64_t>(k) * 10L);
  }
  for (int32_t s = 0; s < numShards; ++s) {
    if (keysPerShard[s].empty()) {
      // Empty shard still needs to exist so manager can route to it.
      // Populate with a sentinel row that won't match any probe key.
      std::vector<int32_t> sentinelKeys{-9999};
      std::vector<int64_t> sentinelVals{0L};
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(sentinelKeys),
           makeFlatVector<int64_t>(sentinelVals)});
      populateShard(setId, s, sentinel, 1);
    } else {
      auto buildData = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(keysPerShard[s]),
           makeFlatVector<int64_t>(valuesPerShard[s])});
      populateShard(setId, s, buildData, 1);
    }
  }

  // Probe with all 12 keys.  All should match.
  std::vector<int32_t> probeKeys;
  for (int32_t k = 1; k <= 12; ++k) {
    probeKeys.push_back(k);
  }
  auto probeData =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>(probeKeys)});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards);

  // Expected: every probe key matched, value = key * 10.  Output order is
  // shard-completion order, so we sort by 'pk' for stable comparison via
  // assertResults below (DuckDB-free path).  We build expected as the
  // sorted key set.
  std::vector<int32_t> expectedKeys;
  std::vector<int64_t> expectedValues;
  for (int32_t k = 1; k <= 12; ++k) {
    expectedKeys.push_back(k);
    expectedValues.push_back(static_cast<int64_t>(k) * 10L);
  }
  // The result-row order is non-deterministic across shards, so use
  // a sort-then-compare strategy: count rows and verify set membership
  // by collecting the result with copyResults and sorting.
  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 12);

  // Extract (pk, v) pairs and sort by pk for stable comparison.
  std::vector<std::pair<int32_t, int64_t>> got;
  got.reserve(results->size());
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  for (vector_size_t r = 0; r < results->size(); ++r) {
    got.emplace_back(pkVec->valueAt(r), vVec->valueAt(r));
  }
  std::sort(got.begin(), got.end());

  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].first, expectedKeys[i]);
    EXPECT_EQ(got[i].second, expectedValues[i]);
  }
}

// Multiple input batches feeding the same shards — verifies cross-batch
// per-shard buffering: rows from batch1 and batch2 destined for shard X
// should accumulate in the same buffer, then flush together.
TEST_F(ShardLookupJoinTest, InnerJoinMultipleBatchesCrossShardBuffering) {
  const int64_t setId = 5;
  const int32_t numShards = 4;

  // Build side: keys 1..20 distributed across shards.
  std::vector<std::vector<int32_t>> keysPerShard(numShards);
  std::vector<std::vector<int64_t>> valuesPerShard(numShards);
  for (int32_t k = 1; k <= 20; ++k) {
    int32_t s = computeShardIdInt32(k, numShards);
    keysPerShard[s].push_back(k);
    valuesPerShard[s].push_back(static_cast<int64_t>(k) + 1000L);
  }
  for (int32_t s = 0; s < numShards; ++s) {
    if (keysPerShard[s].empty()) {
      std::vector<int32_t> sentinelKeys{-9999};
      std::vector<int64_t> sentinelVals{0L};
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(sentinelKeys),
           makeFlatVector<int64_t>(sentinelVals)});
      populateShard(setId, s, sentinel, 1);
    } else {
      auto buildData = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(keysPerShard[s]),
           makeFlatVector<int64_t>(valuesPerShard[s])});
      populateShard(setId, s, buildData, 1);
    }
  }

  // Three small probe batches, each with overlapping shard targets.
  auto batch1 =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto batch2 =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>({6, 7, 8, 9, 10})});
  auto batch3 = makeRowVector(
      {"pk"}, {makeFlatVector<int32_t>({11, 12, 13, 14, 15})});

  // Use a large maxBatchSize so per-shard buffers hold rows across batches
  // and only flush at noMoreInput — this exercises the cross-batch
  // accumulation path.
  auto plan = makeJoinPlan(
      {batch1, batch2, batch3},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards,
      /*maxBatchSize=*/4096,
      /*maxInflightRpcs=*/4);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 15);

  std::vector<std::pair<int32_t, int64_t>> got;
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  for (vector_size_t r = 0; r < results->size(); ++r) {
    got.emplace_back(pkVec->valueAt(r), vVec->valueAt(r));
  }
  std::sort(got.begin(), got.end());

  for (int32_t k = 1; k <= 15; ++k) {
    EXPECT_EQ(got[k - 1].first, k);
    EXPECT_EQ(got[k - 1].second, static_cast<int64_t>(k) + 1000L);
  }
}

// Per-shard flush triggered by maxBatchSize: verifies that when a shard's
// buffer hits the threshold, it flushes immediately mid-batch.
TEST_F(ShardLookupJoinTest, InnerJoinPerShardFlushByThreshold) {
  const int64_t setId = 6;
  const int32_t numShards = 2;

  // Build side: keys 1..20 across 2 shards.
  std::vector<std::vector<int32_t>> keysPerShard(numShards);
  std::vector<std::vector<int64_t>> valuesPerShard(numShards);
  for (int32_t k = 1; k <= 20; ++k) {
    int32_t s = computeShardIdInt32(k, numShards);
    keysPerShard[s].push_back(k);
    valuesPerShard[s].push_back(static_cast<int64_t>(k));
  }
  for (int32_t s = 0; s < numShards; ++s) {
    if (keysPerShard[s].empty()) {
      std::vector<int32_t> sentinelKeys{-9999};
      std::vector<int64_t> sentinelVals{0L};
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(sentinelKeys),
           makeFlatVector<int64_t>(sentinelVals)});
      populateShard(setId, s, sentinel, 1);
      continue;
    }
    auto buildData = makeRowVector(
        {"k", "v"},
        {makeFlatVector<int32_t>(keysPerShard[s]),
         makeFlatVector<int64_t>(valuesPerShard[s])});
    populateShard(setId, s, buildData, 1);
  }

  // 20 probe rows in a single batch, but maxBatchSize=4 forces multiple
  // per-shard flushes within the same input batch.
  std::vector<int32_t> probeKeys;
  for (int32_t k = 1; k <= 20; ++k) {
    probeKeys.push_back(k);
  }
  auto probeData =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>(probeKeys)});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards,
      /*maxBatchSize=*/4,
      /*maxInflightRpcs=*/8);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 20);

  std::vector<std::pair<int32_t, int64_t>> got;
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  for (vector_size_t r = 0; r < results->size(); ++r) {
    got.emplace_back(pkVec->valueAt(r), vVec->valueAt(r));
  }
  std::sort(got.begin(), got.end());

  for (int32_t k = 1; k <= 20; ++k) {
    EXPECT_EQ(got[k - 1].first, k);
    EXPECT_EQ(got[k - 1].second, static_cast<int64_t>(k));
  }
}

// Edge: empty probe input — operator should produce no output and finish
// cleanly without any RPCs.
TEST_F(ShardLookupJoinTest, EmptyProbeInput) {
  const int64_t setId = 7;
  const int32_t numShards = 2;

  // Need at least one shard table for routing to be valid.
  for (int32_t s = 0; s < numShards; ++s) {
    std::vector<int32_t> sentinelKeys{s + 100};
    std::vector<int64_t> sentinelVals{static_cast<int64_t>(s + 100)};
    auto buildData = makeRowVector(
        {"k", "v"},
        {makeFlatVector<int32_t>(sentinelKeys),
         makeFlatVector<int64_t>(sentinelVals)});
    populateShard(setId, s, buildData, 1);
  }

  // ValuesNode with no rows is supported via a zero-row RowVector.
  auto emptyProbe = makeRowVector({"pk"}, {makeFlatVector<int32_t>({})});

  auto plan = makeJoinPlan(
      {emptyProbe},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards);

  AssertQueryBuilder(plan).assertEmptyResults();
}

// Edge: Left Join with empty build shards — every probe row should appear
// with null-filled build columns.
TEST_F(ShardLookupJoinTest, LeftJoinEmptyBuild) {
  const int64_t setId = 8;
  const int32_t numShards = 2;

  // Populate all shards with rows that won't match any probe key.
  for (int32_t s = 0; s < numShards; ++s) {
    std::vector<int32_t> sentinelKeys{-(s + 1)};
    std::vector<int64_t> sentinelVals{0L};
    auto buildData = makeRowVector(
        {"k", "v"},
        {makeFlatVector<int32_t>(sentinelKeys),
         makeFlatVector<int64_t>(sentinelVals)});
    populateShard(setId, s, buildData, 1);
  }

  auto probeData =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>({1, 2, 3})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kLeft,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 3);

  // All probe rows present; 'v' column must be all null.
  std::vector<int32_t> gotPks;
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1);
  for (vector_size_t r = 0; r < results->size(); ++r) {
    gotPks.push_back(pkVec->valueAt(r));
    EXPECT_TRUE(vVec->isNullAt(r))
        << "Expected null v at output row " << r;
  }
  std::sort(gotPks.begin(), gotPks.end());
  EXPECT_EQ(gotPks, (std::vector<int32_t>{1, 2, 3}));
}

// Left Join with a single completed lookup whose total output spans many
// `outputBatchRows`-sized slices.  Specifically targets the pre-computed
// `unmatchedProbeRows` cache: phase-2 (null-fill) is reentered on every
// slice, so any per-slice hash-set rebuild would either show up as wrong
// results or significant slowdown.  We use 5000 probe rows with ~70%
// unmatched, which forces multiple phase-2 slices at the default
// outputBatchRows (1024).
TEST_F(ShardLookupJoinTest, LeftJoinUnmatchedCacheAcrossOutputSlices) {
  const int64_t setId = 9;
  const int32_t numShards = 1;

  // Build side: only multiples of 3 from [0, 5000).  Probe rows in
  // [0, 5000) whose key is not divisible by 3 will be unmatched.
  std::vector<int32_t> buildKeys;
  std::vector<int64_t> buildVals;
  for (int32_t k = 0; k < 5000; k += 3) {
    buildKeys.push_back(k);
    buildVals.push_back(static_cast<int64_t>(k) + 1'000'000L);
  }
  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>(buildKeys),
       makeFlatVector<int64_t>(buildVals)});
  populateShard(setId, 0, buildData, 1);

  // Probe side: 0..4999 in a single batch.
  std::vector<int32_t> probeKeys(5000);
  std::iota(probeKeys.begin(), probeKeys.end(), 0);
  auto probeData =
      makeRowVector({"pk"}, {makeFlatVector<int32_t>(probeKeys)});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kLeft,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards,
      /*maxBatchSize=*/8192,
      /*maxInflightRpcs=*/4);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 5000);

  // Verify every probe key appears exactly once and v is correct.
  std::vector<int32_t> seenPks;
  seenPks.reserve(results->size());
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1);
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    seenPks.push_back(pk);
    if (pk % 3 == 0) {
      ASSERT_FALSE(vVec->isNullAt(r))
          << "expected match at pk=" << pk;
      auto v = vVec->as<SimpleVector<int64_t>>()->valueAt(r);
      EXPECT_EQ(v, static_cast<int64_t>(pk) + 1'000'000L);
    } else {
      EXPECT_TRUE(vVec->isNullAt(r))
          << "expected null v at pk=" << pk;
    }
  }
  std::sort(seenPks.begin(), seenPks.end());
  for (int32_t i = 0; i < 5000; ++i) {
    EXPECT_EQ(seenPks[i], i);
  }
}

// Inner Join across many small probe batches (refCount stress test).
// Each input batch contributes rows to multiple shards; correctness here
// implies that the new BatchEntry refCount machinery (increment in
// addInput, decrement in releaseProbeRefs) is consistent across many
// flush / RPC / consume cycles, including the case where some batches'
// rows are split across multiple CompletedLookups.
TEST_F(ShardLookupJoinTest, InnerJoinManyBatchesRefCountStress) {
  const int64_t setId = 10;
  const int32_t numShards = 4;

  // Build: keys 0..199 distributed across 4 shards, value = key * 10.
  std::vector<std::vector<int32_t>> keysPerShard(numShards);
  std::vector<std::vector<int64_t>> valuesPerShard(numShards);
  for (int32_t k = 0; k < 200; ++k) {
    int32_t s = computeShardIdInt32(k, numShards);
    keysPerShard[s].push_back(k);
    valuesPerShard[s].push_back(static_cast<int64_t>(k) * 10L);
  }
  for (int32_t s = 0; s < numShards; ++s) {
    if (keysPerShard[s].empty()) {
      std::vector<int32_t> sentinelKeys{-9999};
      std::vector<int64_t> sentinelVals{0L};
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(sentinelKeys),
           makeFlatVector<int64_t>(sentinelVals)});
      populateShard(setId, s, sentinel, 1);
    } else {
      auto buildData = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(keysPerShard[s]),
           makeFlatVector<int64_t>(valuesPerShard[s])});
      populateShard(setId, s, buildData, 1);
    }
  }

  // 20 probe batches × 10 rows each = 200 rows total, each batch's rows
  // spread across all 4 shards.  Small maxBatchSize forces many flushes
  // per batch, exercising the refCount accumulation/release path.
  std::vector<RowVectorPtr> probeBatches;
  probeBatches.reserve(20);
  for (int32_t b = 0; b < 20; ++b) {
    std::vector<int32_t> keys;
    keys.reserve(10);
    for (int32_t i = 0; i < 10; ++i) {
      keys.push_back(b * 10 + i);
    }
    probeBatches.push_back(
        makeRowVector({"pk"}, {makeFlatVector<int32_t>(keys)}));
  }

  auto plan = makeJoinPlan(
      probeBatches,
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards,
      /*maxBatchSize=*/8,   // forces multiple flushes per shard per batch
      /*maxInflightRpcs=*/16);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 200);

  std::vector<std::pair<int32_t, int64_t>> got;
  got.reserve(200);
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  for (vector_size_t r = 0; r < results->size(); ++r) {
    got.emplace_back(pkVec->valueAt(r), vVec->valueAt(r));
  }
  std::sort(got.begin(), got.end());
  for (int32_t k = 0; k < 200; ++k) {
    EXPECT_EQ(got[k].first, k);
    EXPECT_EQ(got[k].second, static_cast<int64_t>(k) * 10L);
  }
}

// Inner Join where some batches have ZERO rows surviving the (implicit)
// shard route — degenerate stress for the refCount==0 fast-erase path
// in addInput.  We can't easily inject a BloomFilter from the test
// fixture, but we can still confirm that the operator handles a mix of
// empty-output and non-empty-output input batches without leaking or
// crashing.  This also validates that nextBatchId_ continues to advance
// correctly across many addInput calls (including ones that erase
// immediately).
TEST_F(ShardLookupJoinTest, InnerJoinMixedEmptyAndPopulatedBatches) {
  const int64_t setId = 11;
  const int32_t numShards = 2;

  // Build: only key 42 lives in the table.
  for (int32_t s = 0; s < numShards; ++s) {
    if (computeShardIdInt32(42, numShards) == s) {
      std::vector<int32_t> ks{42};
      std::vector<int64_t> vs{4200L};
      auto buildData = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(ks),
           makeFlatVector<int64_t>(vs)});
      populateShard(setId, s, buildData, 1);
    } else {
      std::vector<int32_t> ks{-9999};
      std::vector<int64_t> vs{0L};
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(ks),
           makeFlatVector<int64_t>(vs)});
      populateShard(setId, s, sentinel, 1);
    }
  }

  // Alternating batches: odd-indexed batches carry the matching key 42,
  // even-indexed batches carry only never-matching keys.  Inner join =>
  // even batches contribute zero output rows but their refCounts must
  // still drop to zero so the entries don't accumulate.
  std::vector<RowVectorPtr> probeBatches;
  probeBatches.reserve(10);
  for (int32_t b = 0; b < 10; ++b) {
    std::vector<int32_t> keys;
    if (b % 2 == 1) {
      keys = {42};
    } else {
      keys = {1000 + b, 2000 + b};
    }
    probeBatches.push_back(
        makeRowVector({"pk"}, {makeFlatVector<int32_t>(keys)}));
  }

  auto plan = makeJoinPlan(
      probeBatches,
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards,
      /*maxBatchSize=*/4,
      /*maxInflightRpcs=*/4);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  // 5 odd batches × 1 matching row each.
  ASSERT_EQ(results->size(), 5);
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  for (vector_size_t r = 0; r < results->size(); ++r) {
    EXPECT_EQ(pkVec->valueAt(r), 42);
    EXPECT_EQ(vVec->valueAt(r), 4200L);
  }
}

// Left Join correctness across many input batches with a high unmatched
// ratio per shard.  Combines the two new code paths under one query:
//   - per-shard lookups produce CompletedLookups whose unmatchedProbeRows
//     vector spans rows from several input batches
//   - releaseProbeRefs must decrement refCounts on the right inputBatches_
//     entries (one decrement per BufferedRow, regardless of whether the
//     row ended up matched, unmatched, or split across slices)
TEST_F(ShardLookupJoinTest, LeftJoinManyBatchesHighUnmatchedRatio) {
  const int64_t setId = 12;
  const int32_t numShards = 3;

  // Build side: only keys that are multiples of 7 in [0, 300).
  std::vector<std::vector<int32_t>> keysPerShard(numShards);
  std::vector<std::vector<int64_t>> valuesPerShard(numShards);
  for (int32_t k = 0; k < 300; k += 7) {
    int32_t s = computeShardIdInt32(k, numShards);
    keysPerShard[s].push_back(k);
    valuesPerShard[s].push_back(static_cast<int64_t>(k) + 7'000'000L);
  }
  for (int32_t s = 0; s < numShards; ++s) {
    if (keysPerShard[s].empty()) {
      std::vector<int32_t> ks{-9999};
      std::vector<int64_t> vs{0L};
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(ks),
           makeFlatVector<int64_t>(vs)});
      populateShard(setId, s, sentinel, 1);
    } else {
      auto buildData = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(keysPerShard[s]),
           makeFlatVector<int64_t>(valuesPerShard[s])});
      populateShard(setId, s, buildData, 1);
    }
  }

  // 6 probe batches, each containing 50 keys in a contiguous range.
  std::vector<RowVectorPtr> probeBatches;
  for (int32_t b = 0; b < 6; ++b) {
    std::vector<int32_t> keys(50);
    std::iota(keys.begin(), keys.end(), b * 50);
    probeBatches.push_back(
        makeRowVector({"pk"}, {makeFlatVector<int32_t>(keys)}));
  }

  auto plan = makeJoinPlan(
      probeBatches,
      core::JoinType::kLeft,
      {"pk"},
      {"k"},
      asRowType(makeRowVector({"k", "v"},
                              {makeFlatVector<int32_t>({}),
                               makeFlatVector<int64_t>({})})
                    ->type()),
      setId,
      numShards,
      /*maxBatchSize=*/64,
      /*maxInflightRpcs=*/4);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 300);

  std::vector<int32_t> seenPks;
  seenPks.reserve(results->size());
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1);
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    seenPks.push_back(pk);
    if (pk % 7 == 0) {
      ASSERT_FALSE(vVec->isNullAt(r))
          << "expected match at pk=" << pk;
      EXPECT_EQ(
          vVec->as<SimpleVector<int64_t>>()->valueAt(r),
          static_cast<int64_t>(pk) + 7'000'000L);
    } else {
      EXPECT_TRUE(vVec->isNullAt(r))
          << "expected null v at pk=" << pk;
    }
  }
  std::sort(seenPks.begin(), seenPks.end());
  for (int32_t i = 0; i < 300; ++i) {
    EXPECT_EQ(seenPks[i], i);
  }
}

// Regression test for the multi-batch DictionaryVector gather paths.
//
// Background: Optimization 2/3 wrap probe-side rows in DictionaryVector to
// avoid per-row copies.  When all rows in a slice come from the same input
// batch, a single dictionary wraps the base column.  When rows span multiple
// input batches, the gather function falls back to per-segment dictionary
// wraps + concat.
//
// The earlier multi-batch implementation interleaved `indices.push_back`
// with the segment-boundary check, leaving the boundary row missing from
// the previous segment.  Effect: probe key RPC payloads were silently
// truncated by one row per segment boundary, and probe-side output dropped
// the same rows.  At scale this manifested as the probe lookup-join stage
// hanging because the join produced fewer output rows than the downstream
// expected (no error, just stuck driver loops chasing missing data).
//
// This test forces every input batch to fan out across two shards with
// small per-shard buffers, producing many cross-batch segment boundaries
// in both flushShard (Optimization 3) and produceOutput (Optimization 2).
// We assert exact row count and full key/value coverage; the missing-row
// bug would surface as a < count mismatch.
TEST_F(ShardLookupJoinTest, MultiBatchDictionaryGatherCorrectness) {
  const int64_t setId = 42;
  const int32_t numShards = 2;

  // Build keys 0..199 across 2 shards.
  std::vector<std::vector<int32_t>> keysPerShard(numShards);
  std::vector<std::vector<int64_t>> valuesPerShard(numShards);
  for (int32_t k = 0; k < 200; ++k) {
    int32_t s = computeShardIdInt32(k, numShards);
    keysPerShard[s].push_back(k);
    valuesPerShard[s].push_back(static_cast<int64_t>(k) + 100'000L);
  }
  for (int32_t s = 0; s < numShards; ++s) {
    if (keysPerShard[s].empty()) {
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(std::vector<int32_t>{-9999}),
           makeFlatVector<int64_t>(std::vector<int64_t>{0L})});
      populateShard(setId, s, sentinel, 1);
    } else {
      auto buildData = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(keysPerShard[s]),
           makeFlatVector<int64_t>(valuesPerShard[s])});
      populateShard(setId, s, buildData, 1);
    }
  }

  // Many small probe batches.  Each batch has rows that route to BOTH
  // shards, so per-shard buffers accumulate rows from multiple input
  // batches before flushing.  With maxBatchSize=8 and per-batch rowsPerBatch=6,
  // a shard buffer fills only after ~3 input batches contribute, giving
  // us multi-batch segments inside flushShard's gather path.
  const int32_t numBatches = 10;
  const int32_t rowsPerBatch = 6;
  std::vector<RowVectorPtr> probeBatches;
  probeBatches.reserve(numBatches);
  std::set<int32_t> expectedKeys;
  for (int32_t b = 0; b < numBatches; ++b) {
    std::vector<int32_t> pks;
    pks.reserve(rowsPerBatch);
    for (int32_t r = 0; r < rowsPerBatch; ++r) {
      auto pk = b * rowsPerBatch + r;
      pks.push_back(pk);
      expectedKeys.insert(pk);
    }
    probeBatches.push_back(
        makeRowVector({"pk"}, {makeFlatVector<int32_t>(pks)}));
  }

  auto buildOutType = asRowType(makeRowVector(
                                    {"k", "v"},
                                    {makeFlatVector<int32_t>({}),
                                     makeFlatVector<int64_t>({})})
                                    ->type());

  auto plan = makeJoinPlan(
      probeBatches,
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      buildOutType,
      setId,
      numShards,
      /*maxBatchSize=*/8,
      /*maxInflightRpcs=*/4);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());

  // Every probe row matches exactly one build row, so output count must
  // equal input row count.  The pre-fix bug truncated by one row per
  // multi-batch segment boundary, so this assertion would fail at < N.
  ASSERT_EQ(results->size(), numBatches * rowsPerBatch);

  std::vector<std::pair<int32_t, int64_t>> got;
  got.reserve(results->size());
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  for (vector_size_t r = 0; r < results->size(); ++r) {
    got.emplace_back(pkVec->valueAt(r), vVec->valueAt(r));
  }
  std::sort(got.begin(), got.end());

  for (int32_t i = 0; i < static_cast<int32_t>(got.size()); ++i) {
    EXPECT_EQ(got[i].first, i)
        << "missing or mis-ordered probe row at output index " << i;
    EXPECT_EQ(got[i].second, static_cast<int64_t>(i) + 100'000L)
        << "wrong build value joined to probe key " << i;
  }
}

// Same multi-batch gather correctness check, but for Left Join — exercises
// produceOutputForLeftJoin's probe-side gather path (Optimization 2) where
// matched + unmatched rows interleave across input batches.
TEST_F(ShardLookupJoinTest, MultiBatchDictionaryGatherLeftJoin) {
  const int64_t setId = 43;
  const int32_t numShards = 2;

  // Build only EVEN keys 0,2,4,...,198 (probe will see odd keys as
  // unmatched, forcing phase-2 null-fill across batch boundaries).
  std::vector<std::vector<int32_t>> keysPerShard(numShards);
  std::vector<std::vector<int64_t>> valuesPerShard(numShards);
  for (int32_t k = 0; k < 200; k += 2) {
    int32_t s = computeShardIdInt32(k, numShards);
    keysPerShard[s].push_back(k);
    valuesPerShard[s].push_back(static_cast<int64_t>(k) + 100'000L);
  }
  for (int32_t s = 0; s < numShards; ++s) {
    if (keysPerShard[s].empty()) {
      auto sentinel = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(std::vector<int32_t>{-9999}),
           makeFlatVector<int64_t>(std::vector<int64_t>{0L})});
      populateShard(setId, s, sentinel, 1);
    } else {
      auto buildData = makeRowVector(
          {"k", "v"},
          {makeFlatVector<int32_t>(keysPerShard[s]),
           makeFlatVector<int64_t>(valuesPerShard[s])});
      populateShard(setId, s, buildData, 1);
    }
  }

  const int32_t numBatches = 10;
  const int32_t rowsPerBatch = 6;
  std::vector<RowVectorPtr> probeBatches;
  probeBatches.reserve(numBatches);
  for (int32_t b = 0; b < numBatches; ++b) {
    std::vector<int32_t> pks;
    pks.reserve(rowsPerBatch);
    for (int32_t r = 0; r < rowsPerBatch; ++r) {
      pks.push_back(b * rowsPerBatch + r);
    }
    probeBatches.push_back(
        makeRowVector({"pk"}, {makeFlatVector<int32_t>(pks)}));
  }

  auto buildOutType = asRowType(makeRowVector(
                                    {"k", "v"},
                                    {makeFlatVector<int32_t>({}),
                                     makeFlatVector<int64_t>({})})
                                    ->type());

  auto plan = makeJoinPlan(
      probeBatches,
      core::JoinType::kLeft,
      {"pk"},
      {"k"},
      buildOutType,
      setId,
      numShards,
      /*maxBatchSize=*/8,
      /*maxInflightRpcs=*/4);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), numBatches * rowsPerBatch);

  std::vector<int32_t> seenPks;
  seenPks.reserve(results->size());
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1);
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    seenPks.push_back(pk);
    if (pk % 2 == 0) {
      ASSERT_FALSE(vVec->isNullAt(r))
          << "even key " << pk << " should match";
      EXPECT_EQ(
          vVec->as<SimpleVector<int64_t>>()->valueAt(r),
          static_cast<int64_t>(pk) + 100'000L);
    } else {
      EXPECT_TRUE(vVec->isNullAt(r))
          << "odd key " << pk << " should be null-filled";
    }
  }
  std::sort(seenPks.begin(), seenPks.end());
  for (int32_t i = 0; i < numBatches * rowsPerBatch; ++i) {
    EXPECT_EQ(seenPks[i], i)
        << "probe key " << i << " missing from Left Join output";
  }
}

// ---------------------------------------------------------------------------
// Key dedup: Inner Join with duplicate probe keys.
//
// Probe keys: {1, 2, 1, 3, 2, 1}  (1 appears 3x, 2 appears 2x, 3 appears 1x)
// Build side: {1→100, 2→200, 3→300}
// Expected output: 6 rows, each probe row matched with its build value.
// ---------------------------------------------------------------------------
TEST_F(ShardLookupJoinTest, InnerJoinDuplicateProbeKeysDedup) {
  const int64_t setId = 100;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({100L, 200L, 300L})});
  populateShard(setId, 0, buildData, 1);

  auto probeData = makeRowVector(
      {"pk"}, {makeFlatVector<int32_t>({1, 2, 1, 3, 2, 1})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 6);

  // Verify each output row has the correct (pk, v) pair.
  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  std::unordered_map<int32_t, int64_t> expectedMap{{1, 100L}, {2, 200L}, {3, 300L}};
  std::unordered_map<int32_t, int32_t> keyCounts;
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    auto v = vVec->valueAt(r);
    EXPECT_EQ(v, expectedMap.at(pk))
        << "row " << r << ": pk=" << pk << " expected v=" << expectedMap.at(pk);
    keyCounts[pk]++;
  }
  EXPECT_EQ(keyCounts[1], 3) << "key 1 should appear 3 times";
  EXPECT_EQ(keyCounts[2], 2) << "key 2 should appear 2 times";
  EXPECT_EQ(keyCounts[3], 1) << "key 3 should appear 1 time";
}

// ---------------------------------------------------------------------------
// Key dedup: Inner Join with duplicate probe keys AND build-side fan-out.
//
// Probe keys: {1, 1, 2}  (key 1 appears 2x)
// Build side: {1→10, 1→11, 2→20}  (key 1 has 2 build rows)
// Expected: 5 output rows (2 probe × 2 build + 1 probe × 1 build).
// ---------------------------------------------------------------------------
TEST_F(ShardLookupJoinTest, InnerJoinDuplicateKeysWithBuildFanout) {
  const int64_t setId = 101;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 1, 2}),
       makeFlatVector<int64_t>({10L, 11L, 20L})});
  populateShard(setId, 0, buildData, 1);

  auto probeData = makeRowVector(
      {"pk"}, {makeFlatVector<int32_t>({1, 1, 2})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  // 2 probe rows with key=1 × 2 build rows = 4, plus 1×1 for key=2 = 5 total.
  ASSERT_EQ(results->size(), 5);

  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  std::unordered_map<int32_t, int32_t> keyCounts;
  std::set<int64_t> valuesForKey1;
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    keyCounts[pk]++;
    if (pk == 1) {
      valuesForKey1.insert(vVec->valueAt(r));
    } else {
      EXPECT_EQ(vVec->valueAt(r), 20L);
    }
  }
  // key=1: 2 probe × 2 build = 4 rows
  EXPECT_EQ(keyCounts[1], 4);
  // key=2: 1 probe × 1 build = 1 row
  EXPECT_EQ(keyCounts[2], 1);
  // Both build values for key=1 should appear
  EXPECT_TRUE(valuesForKey1.count(10L)) << "build value 10 missing for key 1";
  EXPECT_TRUE(valuesForKey1.count(11L)) << "build value 11 missing for key 1";
}

// ---------------------------------------------------------------------------
// Key dedup: Left Join with duplicate probe keys — unmatched duplicates
// should all get null-filled build columns.
//
// Probe keys: {1, 2, 1, 99, 99}  (99 has no match, appears 2x)
// Build side: {1→100, 2→200}
// Expected: 5 rows, 2 of which have null build values.
// ---------------------------------------------------------------------------
TEST_F(ShardLookupJoinTest, LeftJoinDuplicateProbeKeysDedup) {
  const int64_t setId = 102;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({100L, 200L})});
  populateShard(setId, 0, buildData, 1);

  auto probeData = makeRowVector(
      {"pk"}, {makeFlatVector<int32_t>({1, 2, 1, 99, 99})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kLeft,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 5);

  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vVec = results->childAt(1);
  int32_t nullCount = 0;
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    if (pk == 99) {
      EXPECT_TRUE(vVec->isNullAt(r))
          << "key 99 should be null-filled";
      nullCount++;
    } else if (pk == 1) {
      EXPECT_EQ(vVec->as<SimpleVector<int64_t>>()->valueAt(r), 100L);
    } else if (pk == 2) {
      EXPECT_EQ(vVec->as<SimpleVector<int64_t>>()->valueAt(r), 200L);
    }
  }
  EXPECT_EQ(nullCount, 2) << "key 99 appears 2x, both should be null-filled";
}

// ---------------------------------------------------------------------------
// Post-join filter: Inner Join with non-equi condition.
//
// Probe: pk={1,2,3,4}, pval={10,20,30,40}
// Build: k={1,2,3,4}, v={15,10,35,50}
// Equi-join on pk=k, then filter: pval < v
// Expected: only (1,10,15) and (3,30,35) pass (pval < v).
// ---------------------------------------------------------------------------
TEST_F(ShardLookupJoinTest, InnerJoinWithPostFilter) {
  const int64_t setId = 200;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({15L, 10L, 35L, 50L})});
  populateShard(setId, 0, buildData, 1);

  auto probeData = makeRowVector(
      {"pk", "pval"},
      {makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({10L, 20L, 30L, 40L})});

  // Build filter expression: pval < v
  // The filter input type is concat(probeType, buildOutputType) =
  // {pk:INTEGER, pval:BIGINT, k:INTEGER, v:BIGINT}
  auto pvalField = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(), "pval");
  auto vField = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(), "v");
  auto filterExpr = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{pvalField, vField},
      "lessthan");

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards,
      /*maxBatchSize=*/1024,
      /*maxInflightRpcs=*/4,
      filterExpr);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  // pk=1: pval=10 < v=15 → pass
  // pk=2: pval=20 < v=10 → fail (20 >= 10)
  // pk=3: pval=30 < v=35 → pass
  // pk=4: pval=40 < v=50 → pass
  ASSERT_EQ(results->size(), 3);

  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto pvalVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  auto vIdx = results->type()->as<TypeKind::ROW>().getChildIdx("v");
  auto vVec = results->childAt(vIdx)->as<SimpleVector<int64_t>>();

  std::set<int32_t> seenPks;
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    seenPks.insert(pk);
    EXPECT_LT(pvalVec->valueAt(r), vVec->valueAt(r))
        << "filter pval < v must hold for pk=" << pk;
  }
  EXPECT_TRUE(seenPks.count(1)) << "pk=1 should pass filter (10<15)";
  EXPECT_TRUE(seenPks.count(3)) << "pk=3 should pass filter (30<35)";
  EXPECT_TRUE(seenPks.count(4)) << "pk=4 should pass filter (40<50)";
  EXPECT_FALSE(seenPks.count(2)) << "pk=2 should be filtered (20>=10)";
}

// ---------------------------------------------------------------------------
// Post-join filter: no filter (nullptr) — baseline correctness.
// Same as InnerJoinSingleShardSingleBatch but explicitly passes nullptr filter.
// ---------------------------------------------------------------------------
TEST_F(ShardLookupJoinTest, InnerJoinNullFilterPassthrough) {
  const int64_t setId = 201;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({100L, 200L})});
  populateShard(setId, 0, buildData, 1);

  auto probeData = makeRowVector(
      {"pk"}, {makeFlatVector<int32_t>({1, 2})});

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards,
      1024, 4,
      /*filter=*/nullptr);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 2);
}

// ---------------------------------------------------------------------------
// Left Join with server-side filter: unmatched + filter-rejected rows
// get null-filled build columns.
//
// Probe: pk={1,2,3,5}, pval={10,20,30,99}
// Build: k={1,2,3,4}, v={15,10,35,50}
// Equi-join on pk=k, then filter: pval < v
// Expected:
//   pk=1: pval=10 < v=15 → pass → (1,10,1,15)
//   pk=2: pval=20 < v=10 → fail → LEFT null-fill → (2,20,null,null)
//   pk=3: pval=30 < v=35 → pass → (3,30,3,35)
//   pk=5: no match        → LEFT null-fill → (5,99,null,null)
// ---------------------------------------------------------------------------
TEST_F(ShardLookupJoinTest, LeftJoinWithFilter) {
  const int64_t setId = 300;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({15L, 10L, 35L, 50L})});
  populateShard(setId, 0, buildData, 1);

  auto probeData = makeRowVector(
      {"pk", "pval"},
      {makeFlatVector<int32_t>({1, 2, 3, 5}),
       makeFlatVector<int64_t>({10L, 20L, 30L, 99L})});

  auto pvalField = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(), "pval");
  auto vField = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(), "v");
  auto filterExpr = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{pvalField, vField},
      "lessthan");

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kLeft,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards,
      1024, 4,
      filterExpr);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  // 4 output rows: 2 matched + 2 null-filled (pk=2 filtered, pk=5 unmatched)
  ASSERT_EQ(results->size(), 4);

  auto pkVec = results->childAt(0)->as<SimpleVector<int32_t>>();
  auto vIdx = results->type()->as<TypeKind::ROW>().getChildIdx("v");
  auto vVec = results->childAt(vIdx);

  int nullCount = 0;
  std::set<int32_t> matchedPks;
  for (vector_size_t r = 0; r < results->size(); ++r) {
    auto pk = pkVec->valueAt(r);
    if (vVec->isNullAt(r)) {
      nullCount++;
      EXPECT_TRUE(pk == 2 || pk == 5)
          << "pk=" << pk << " should be null-filled";
    } else {
      matchedPks.insert(pk);
    }
  }
  EXPECT_EQ(nullCount, 2);
  EXPECT_TRUE(matchedPks.count(1));
  EXPECT_TRUE(matchedPks.count(3));
}

// ---------------------------------------------------------------------------
// Inner Join with filter + duplicate probe keys.
// Verifies that dedup key extension (concat probeKeys + filterProbeColumns)
// correctly handles duplicate keys with different filter column values.
//
// Probe: pk={1,1,2,2}, pval={5,50,5,50}
// Build: k={1,2}, v={10,10}
// Filter: pval < v
// Expected:
//   (pk=1, pval=5): 5 < 10 → pass
//   (pk=1, pval=50): 50 < 10 → fail
//   (pk=2, pval=5): 5 < 10 → pass
//   (pk=2, pval=50): 50 < 10 → fail
// Result: 2 rows.
// ---------------------------------------------------------------------------
TEST_F(ShardLookupJoinTest, InnerJoinFilterWithDuplicateKeys) {
  const int64_t setId = 301;
  const int32_t numShards = 1;

  auto buildData = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({10L, 10L})});
  populateShard(setId, 0, buildData, 1);

  auto probeData = makeRowVector(
      {"pk", "pval"},
      {makeFlatVector<int32_t>({1, 1, 2, 2}),
       makeFlatVector<int64_t>({5L, 50L, 5L, 50L})});

  auto pvalField = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(), "pval");
  auto vField = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(), "v");
  auto filterExpr = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{pvalField, vField},
      "lessthan");

  auto plan = makeJoinPlan(
      {probeData},
      core::JoinType::kInner,
      {"pk"},
      {"k"},
      asRowType(buildData->type()),
      setId,
      numShards,
      1024, 4,
      filterExpr);

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());
  ASSERT_EQ(results->size(), 2);

  auto pvalVec = results->childAt(1)->as<SimpleVector<int64_t>>();
  for (vector_size_t r = 0; r < results->size(); ++r) {
    EXPECT_EQ(pvalVec->valueAt(r), 5L)
        << "only pval=5 should pass filter (5 < 10)";
  }
}
