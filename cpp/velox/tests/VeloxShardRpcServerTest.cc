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

#include "shard/VeloxShardRpcServer.h"

#include <sstream>

#include <folly/io/IOBuf.h>
#include <grpcpp/grpcpp.h>

#include "compute/VeloxBackend.h"
#include "shard/BlockManagerBridge.h"
#include "shard/CompactRowWireFormat.h"
#include "shard/VeloxShardManager.h"
#include "shard/proto/shard_lookup.grpc.pb.h"
#include "velox/common/memory/StreamArena.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace gluten;
using namespace gluten::shard;

// ---------------------------------------------------------------------------
// MockBlockManagerBridge — stores shard data in memory for unit testing.
// (Same as in VeloxShardManagerTest.cc)
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
class VeloxShardRpcServerTest : public ::testing::Test,
                                public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    VeloxBackend::create({});
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    mockBridge_ = std::make_shared<MockBlockManagerBridge>();
    VeloxShardManager::getInstance()->initialize(mockBridge_);
    pool_ = memory::memoryManager()->addRootPool("rpcTest");
    leafPool_ = pool_->addLeafChild("rpcTestLeaf");
  }

  void TearDown() override {
    auto* rpcServer = VeloxShardRpcServer::getInstance();
    if (rpcServer->isRunning()) {
      rpcServer->stopServer();
    }
    VeloxShardManager::getInstance()->shutdown();
  }

  // Serialize a RowVector to Presto format bytes.
  // NOTE: Kept for build-side piece block storage (mockBridge_->putBlock),
  // which is the input format expected by VeloxShardManager::constructShardTable.
  // The probe-side RPC path uses serializeToCompactWire instead.
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

  // Serialize a RowVector to CompactRow wire format for the gRPC RPC path.
  // This matches what ShardLookupJoin produces when issuing remote lookups.
  std::string serializeToCompactWire(const RowVectorPtr& rowVector) {
    return shard::serializeCompactRowBatch(rowVector, leafPool_.get());
  }

  // Deserialize CompactRow wire-format bytes into a RowVector.
  // This matches what ShardLookupJoin consumes when receiving remote responses.
  RowVectorPtr deserializeFromCompactWire(
      const std::string& bytes,
      const RowTypePtr& rowType) {
    return shard::deserializeCompactRowBatch(
        std::string_view(bytes.data(), bytes.size()),
        rowType,
        leafPool_.get());
  }

  // Build a meta block.
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

  // Populate a shard in the mock bridge and construct the HashTable.
  void populateAndBuildShard(
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

  std::shared_ptr<MockBlockManagerBridge> mockBridge_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Verify gRPC server starts and stops correctly.
TEST_F(VeloxShardRpcServerTest, ServerStartStop) {
  auto* rpcServer = VeloxShardRpcServer::getInstance();
  ASSERT_FALSE(rpcServer->isRunning());

  // Start on port 0 to let OS assign a free port.
  int port = rpcServer->startServer("0.0.0.0", 0);
  ASSERT_TRUE(rpcServer->isRunning());
  ASSERT_GT(port, 0);
  ASSERT_EQ(rpcServer->port(), port);

  rpcServer->stopServer();
  ASSERT_FALSE(rpcServer->isRunning());
}

// End-to-end: start server, build HashTable, send gRPC lookup, verify result.
TEST_F(VeloxShardRpcServerTest, RemoteLookupEndToEnd) {
  int64_t setId = 300;
  int32_t shardId = 0;

  // Build side: key=int32, value=int64
  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  populateAndBuildShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  // Start gRPC server.
  auto* rpcServer = VeloxShardRpcServer::getInstance();
  int port = rpcServer->startServer("0.0.0.0", 0);
  ASSERT_GT(port, 0);

  // Create gRPC client stub.
  auto channel = grpc::CreateChannel(
      "localhost:" + std::to_string(port),
      grpc::InsecureChannelCredentials());
  auto stub = proto::ShardLookupService::NewStub(channel);

  // Build lookup request: probe keys {10, 30}
  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({10, 30})});
  auto probeBytes = serializeToCompactWire(probeKeys);

  proto::LookupRequest request;
  request.set_set_id(setId);
  request.set_shard_id(shardId);
  request.set_probe_keys_compact(probeBytes);
  request.add_input_indices(0);
  request.add_input_indices(1);

  proto::LookupResponse response;
  grpc::ClientContext context;
  auto status = stub->Lookup(&context, request, &response);

  ASSERT_TRUE(status.ok()) << "gRPC error: " << status.error_message();
  ASSERT_FALSE(response.output_compact().empty());
  ASSERT_EQ(response.input_hits_size(), 2);
  EXPECT_EQ(response.input_hits(0), 0);
  EXPECT_EQ(response.input_hits(1), 1);

  // Deserialize output and verify values.
  // Output type matches the build side schema: (int32, int64).
  auto outputType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});
  auto output = deserializeFromCompactWire(response.output_compact(), outputType);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), 2);
  ASSERT_EQ(output->childrenSize(), 2);

  auto outputKeys = output->childAt(0)->asFlatVector<int32_t>();
  EXPECT_EQ(outputKeys->valueAt(0), 10);
  EXPECT_EQ(outputKeys->valueAt(1), 30);

  auto outputValues = output->childAt(1)->asFlatVector<int64_t>();
  EXPECT_EQ(outputValues->valueAt(0), 100L);
  EXPECT_EQ(outputValues->valueAt(1), 300L);
}

// Remote lookup with no matches returns empty response.
TEST_F(VeloxShardRpcServerTest, RemoteLookupNoMatch) {
  int64_t setId = 301;
  int32_t shardId = 0;

  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({10, 20}),
      makeFlatVector<int32_t>({100, 200}),
  });
  populateAndBuildShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  auto* rpcServer = VeloxShardRpcServer::getInstance();
  int port = rpcServer->startServer("0.0.0.0", 0);

  auto channel = grpc::CreateChannel(
      "localhost:" + std::to_string(port),
      grpc::InsecureChannelCredentials());
  auto stub = proto::ShardLookupService::NewStub(channel);

  // Probe with keys that don't exist.
  auto probeKeys = makeRowVector({makeFlatVector<int32_t>({99, 999})});
  auto probeBytes = serializeToCompactWire(probeKeys);

  proto::LookupRequest request;
  request.set_set_id(setId);
  request.set_shard_id(shardId);
  request.set_probe_keys_compact(probeBytes);
  request.add_input_indices(0);
  request.add_input_indices(1);

  proto::LookupResponse response;
  grpc::ClientContext context;
  auto status = stub->Lookup(&context, request, &response);

  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(response.output_compact().empty());
  EXPECT_EQ(response.input_hits_size(), 0);
}

// Remote lookup on non-existent shard returns empty response.
TEST_F(VeloxShardRpcServerTest, RemoteLookupNonExistentShard) {
  auto* rpcServer = VeloxShardRpcServer::getInstance();
  int port = rpcServer->startServer("0.0.0.0", 0);

  auto channel = grpc::CreateChannel(
      "localhost:" + std::to_string(port),
      grpc::InsecureChannelCredentials());
  auto stub = proto::ShardLookupService::NewStub(channel);

  auto probeKeys =
      makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{1})});
  auto probeBytes = serializeToCompactWire(probeKeys);

  proto::LookupRequest request;
  request.set_set_id(999);
  request.set_shard_id(0);
  request.set_probe_keys_compact(probeBytes);
  request.add_input_indices(0);

  proto::LookupResponse response;
  grpc::ClientContext context;
  auto status = stub->Lookup(&context, request, &response);

  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(response.output_compact().empty());
  EXPECT_EQ(response.input_hits_size(), 0);
}

// Remote lookup with multi-key (composite key).
TEST_F(VeloxShardRpcServerTest, RemoteLookupMultiKey) {
  int64_t setId = 302;
  int32_t shardId = 0;

  // Build side: key1=int32, key2=int64, value=int32
  auto buildData = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2}),
      makeFlatVector<int64_t>({100L, 200L, 100L}),
      makeFlatVector<int32_t>({10, 20, 30}),
  });
  populateAndBuildShard(setId, shardId, buildData, /*numKeyColumns=*/2);

  auto* rpcServer = VeloxShardRpcServer::getInstance();
  int port = rpcServer->startServer("0.0.0.0", 0);

  auto channel = grpc::CreateChannel(
      "localhost:" + std::to_string(port),
      grpc::InsecureChannelCredentials());
  auto stub = proto::ShardLookupService::NewStub(channel);

  // Probe with composite key (1, 200) — should match row with value=20.
  auto probeKeys = makeRowVector({
      makeFlatVector<int32_t>(std::vector<int32_t>{1}),
      makeFlatVector<int64_t>(std::vector<int64_t>{200L}),
  });
  auto probeBytes = serializeToCompactWire(probeKeys);

  proto::LookupRequest request;
  request.set_set_id(setId);
  request.set_shard_id(shardId);
  request.set_probe_keys_compact(probeBytes);
  request.add_input_indices(0);

  proto::LookupResponse response;
  grpc::ClientContext context;
  auto status = stub->Lookup(&context, request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_FALSE(response.output_compact().empty());
  ASSERT_EQ(response.input_hits_size(), 1);
  EXPECT_EQ(response.input_hits(0), 0);

  // Output type matches the build side schema: (int32, int64, int32).
  auto outputType = ROW({"c0", "c1", "c2"}, {INTEGER(), BIGINT(), INTEGER()});
  auto output = deserializeFromCompactWire(response.output_compact(), outputType);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), 1);

  // Verify the matched value column (3rd column, index 2).
  auto outputValues = output->childAt(2)->asFlatVector<int32_t>();
  EXPECT_EQ(outputValues->valueAt(0), 20);
}

// Multiple concurrent lookups on the same server.
TEST_F(VeloxShardRpcServerTest, ConcurrentLookups) {
  int64_t setId = 303;

  // Build two shards.
  auto buildData0 = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int32_t>({10, 20}),
  });
  populateAndBuildShard(setId, 0, buildData0, /*numKeyColumns=*/1);

  auto buildData1 = makeRowVector({
      makeFlatVector<int32_t>({3, 4}),
      makeFlatVector<int32_t>({30, 40}),
  });
  populateAndBuildShard(setId, 1, buildData1, /*numKeyColumns=*/1);

  auto* rpcServer = VeloxShardRpcServer::getInstance();
  int port = rpcServer->startServer("0.0.0.0", 0);

  auto channel = grpc::CreateChannel(
      "localhost:" + std::to_string(port),
      grpc::InsecureChannelCredentials());

  // Issue two lookups concurrently using async API.
  grpc::CompletionQueue completionQueue;

  // Lookup 1: key=1 in shard 0
  auto probeKeys0 =
      makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{1})});
  auto probeBytes0 = serializeToCompactWire(probeKeys0);

  proto::LookupRequest request0;
  request0.set_set_id(setId);
  request0.set_shard_id(0);
  request0.set_probe_keys_compact(probeBytes0);
  request0.add_input_indices(0);

  proto::LookupResponse response0;
  grpc::ClientContext context0;
  grpc::Status status0;
  auto stub0 = proto::ShardLookupService::NewStub(channel);
  auto rpc0 = stub0->AsyncLookup(&context0, request0, &completionQueue);
  rpc0->Finish(&response0, &status0, reinterpret_cast<void*>(1));

  // Lookup 2: key=4 in shard 1
  auto probeKeys1 =
      makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{4})});
  auto probeBytes1 = serializeToCompactWire(probeKeys1);

  proto::LookupRequest request1;
  request1.set_set_id(setId);
  request1.set_shard_id(1);
  request1.set_probe_keys_compact(probeBytes1);
  request1.add_input_indices(0);

  proto::LookupResponse response1;
  grpc::ClientContext context1;
  grpc::Status status1;
  auto stub1 = proto::ShardLookupService::NewStub(channel);
  auto rpc1 = stub1->AsyncLookup(&context1, request1, &completionQueue);
  rpc1->Finish(&response1, &status1, reinterpret_cast<void*>(2));

  // Drain both results.
  for (int i = 0; i < 2; ++i) {
    void* tag;
    bool ok;
    completionQueue.Next(&tag, &ok);
    ASSERT_TRUE(ok);
  }
  completionQueue.Shutdown();

  // Verify both lookups succeeded.
  ASSERT_TRUE(status0.ok()) << status0.error_message();
  ASSERT_TRUE(status1.ok()) << status1.error_message();

  ASSERT_FALSE(response0.output_compact().empty());
  ASSERT_FALSE(response1.output_compact().empty());

  // Output type matches the build side schema: (int32, int32).
  auto outputType = ROW({"c0", "c1"}, {INTEGER(), INTEGER()});
  auto output0 = deserializeFromCompactWire(response0.output_compact(), outputType);
  ASSERT_EQ(output0->size(), 1);
  auto value0 = output0->childAt(1)->asFlatVector<int32_t>();
  EXPECT_EQ(value0->valueAt(0), 10);

  auto output1 = deserializeFromCompactWire(response1.output_compact(), outputType);
  ASSERT_EQ(output1->size(), 1);
  auto value1 = output1->childAt(1)->asFlatVector<int32_t>();
  EXPECT_EQ(value1->valueAt(0), 40);
}

// VARCHAR key tests are guarded by NDEBUG because Velox's Presto
// deserializer for StringView columns triggers a false-positive
// AlignedBuffer end-guard check in debug builds.
#ifdef NDEBUG

// Remote lookup with VARCHAR key.
TEST_F(VeloxShardRpcServerTest, RemoteLookupVarcharKey) {
  int64_t setId = 304;
  int32_t shardId = 0;

  // Build side: key=varchar, value=int64
  auto buildData = makeRowVector({
      makeFlatVector<StringView>({"alice"_sv, "bob"_sv, "charlie"_sv}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  populateAndBuildShard(setId, shardId, buildData, /*numKeyColumns=*/1);

  auto* rpcServer = VeloxShardRpcServer::getInstance();
  int port = rpcServer->startServer("0.0.0.0", 0);
  ASSERT_GT(port, 0);

  auto channel = grpc::CreateChannel(
      "localhost:" + std::to_string(port),
      grpc::InsecureChannelCredentials());
  auto stub = proto::ShardLookupService::NewStub(channel);

  // Probe with keys {"alice", "charlie"}.
  auto probeKeys = makeRowVector(
      {makeFlatVector<StringView>({"alice"_sv, "charlie"_sv})});
  auto probeBytes = serializeToCompactWire(probeKeys);

  proto::LookupRequest request;
  request.set_set_id(setId);
  request.set_shard_id(shardId);
  request.set_probe_keys_compact(probeBytes);
  request.add_input_indices(0);
  request.add_input_indices(1);

  proto::LookupResponse response;
  grpc::ClientContext context;
  auto status = stub->Lookup(&context, request, &response);

  ASSERT_TRUE(status.ok()) << "gRPC error: " << status.error_message();
  ASSERT_FALSE(response.output_compact().empty());
  ASSERT_EQ(response.input_hits_size(), 2);

  // Deserialize output: (varchar, int64).
  auto outputType = ROW({"c0", "c1"}, {VARCHAR(), BIGINT()});
  auto output = deserializeFromCompactWire(response.output_compact(), outputType);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), 2);

  auto outputKeys = output->childAt(0)->asFlatVector<StringView>();
  auto outputValues = output->childAt(1)->asFlatVector<int64_t>();

  std::map<std::string, int64_t> kvPairs;
  for (int i = 0; i < 2; ++i) {
    kvPairs[outputKeys->valueAt(i).str()] = outputValues->valueAt(i);
  }
  EXPECT_EQ(kvPairs["alice"], 100L);
  EXPECT_EQ(kvPairs["charlie"], 300L);
}

#endif // NDEBUG
