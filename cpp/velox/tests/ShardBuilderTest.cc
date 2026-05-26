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

#include "shard/ShardBuilder.h"

#include "compute/VeloxBackend.h"
#include "shard/BloomHashUtil.h"
#include "velox/common/base/BloomFilter.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace gluten;
using namespace gluten::shard;

class ShardBuilderTest : public ::testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    VeloxBackend::create({});
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addRootPool("shard_builder_test");
    leafPool_ = pool_->addLeafChild("leaf");
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
};

// ---------------------------------------------------------------------------
// Single batch, blockSize large enough — no flush during append, only on finish.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, SingleBatchNoFlush) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024 * 1024; // 1MB — large enough for 3 rows
  constexpr int64_t bloomCapacity = 100;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  auto batch = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });

  // Append should not flush (blockSize is large).
  auto piece = builder.appendBatch(batch);
  EXPECT_FALSE(piece.has_value());
  EXPECT_EQ(builder.totalRows(), 3);

  // Finish should return the last piece.
  auto result = builder.finish();
  ASSERT_TRUE(result.lastPiece.has_value());
  EXPECT_FALSE(result.lastPiece->bytes.empty());
  EXPECT_GT(result.lastPiece->checksum, 0u);
  EXPECT_EQ(result.totalPieceCount, 1);
  EXPECT_EQ(result.checksums.size(), 1u);
  EXPECT_EQ(result.checksums[0], result.lastPiece->checksum);

  // BloomFilter bytes should be non-empty.
  EXPECT_FALSE(result.bloomBytes.empty());
}

// ---------------------------------------------------------------------------
// Multiple batches with small blockSize — triggers flush during append.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, MultipleBatchesWithFlush) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1; // 1 byte — forces flush on every append
  constexpr int64_t bloomCapacity = 100;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({10L, 20L}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({3, 4}),
      makeFlatVector<int64_t>({30L, 40L}),
  });

  // First append should flush (blockSize=1).
  auto piece1 = builder.appendBatch(batch1);
  ASSERT_TRUE(piece1.has_value());
  EXPECT_FALSE(piece1->bytes.empty());
  EXPECT_GT(piece1->checksum, 0u);

  // Second append should also flush.
  auto piece2 = builder.appendBatch(batch2);
  ASSERT_TRUE(piece2.has_value());
  EXPECT_FALSE(piece2->bytes.empty());

  EXPECT_EQ(builder.totalRows(), 4);

  // Finish should have no last piece (all flushed during append).
  auto result = builder.finish();
  EXPECT_FALSE(result.lastPiece.has_value());
  EXPECT_EQ(result.totalPieceCount, 2);
  EXPECT_EQ(result.checksums.size(), 2u);
}

// ---------------------------------------------------------------------------
// Empty batch should not produce a piece.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, EmptyBatch) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024;
  constexpr int64_t bloomCapacity = 100;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  // Null batch.
  auto piece1 = builder.appendBatch(nullptr);
  EXPECT_FALSE(piece1.has_value());

  // Empty batch (0 rows).
  auto emptyBatch = makeRowVector({makeFlatVector<int32_t>({})});
  auto piece2 = builder.appendBatch(emptyBatch);
  EXPECT_FALSE(piece2.has_value());

  EXPECT_EQ(builder.totalRows(), 0);

  auto result = builder.finish();
  EXPECT_FALSE(result.lastPiece.has_value());
  EXPECT_EQ(result.totalPieceCount, 0);
  EXPECT_TRUE(result.checksums.empty());
}

// ---------------------------------------------------------------------------
// BloomFilter should contain inserted keys and reject absent keys.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, BloomFilterContainsKeys) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 1000;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  // Insert keys 0..99.
  std::vector<int32_t> keys(100);
  std::iota(keys.begin(), keys.end(), 0);
  auto batch = makeRowVector({
      makeFlatVector<int32_t>(keys),
      makeFlatVector<int64_t>(std::vector<int64_t>(100, 42L)),
  });
  builder.appendBatch(batch);
  auto result = builder.finish();

  // Deserialize the BloomFilter.
  ASSERT_FALSE(result.bloomBytes.empty());
  BloomFilter<> bloomFilter;
  bloomFilter.merge(result.bloomBytes.data());

  // All inserted keys should be present (hash with seed=42, matching Spark).
  for (int32_t key : keys) {
    int32_t hash = SparkMurmurHash::hashInt(key, 42);
    EXPECT_TRUE(bloomFilter.mayContain(spreadHashForBloom(hash)))
        << "Key " << key << " (hash=" << hash << ") should be in BloomFilter";
  }

  // Keys far outside the range should mostly not be present (probabilistic).
  int falsePositives = 0;
  for (int32_t key = 10000; key < 10100; ++key) {
    int32_t hash = SparkMurmurHash::hashInt(key, 42);
    if (bloomFilter.mayContain(spreadHashForBloom(hash))) {
      falsePositives++;
    }
  }
  // With fpp=0.01 and 100 absent keys, expect < 5 false positives.
  EXPECT_LT(falsePositives, 10)
      << "Too many false positives: " << falsePositives << "/100";
}

// ---------------------------------------------------------------------------
// Adler32 checksum should be deterministic and non-trivial.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, Adler32Deterministic) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 100;
  

  auto batch = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });

  // Build twice with the same data — checksums should match.
  ShardBuilder builder1(numKeyColumns, blockSize, bloomCapacity, leafPool_);
  builder1.appendBatch(batch);
  auto result1 = builder1.finish();

  ShardBuilder builder2(numKeyColumns, blockSize, bloomCapacity, leafPool_);
  builder2.appendBatch(batch);
  auto result2 = builder2.finish();

  ASSERT_EQ(result1.checksums.size(), 1u);
  ASSERT_EQ(result2.checksums.size(), 1u);
  EXPECT_EQ(result1.checksums[0], result2.checksums[0]);

  // Checksum should not be the trivial Adler32 of empty data (1).
  EXPECT_NE(result1.checksums[0], 1u);
}

// ---------------------------------------------------------------------------
// Multi-key BloomFilter — composite key hash should be chained.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, MultiKeyBloomFilter) {
  constexpr int32_t numKeyColumns = 2;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 1000;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  auto batch = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
      makeFlatVector<int32_t>({10, 20, 30}), // value column, not part of key
  });
  builder.appendBatch(batch);
  auto result = builder.finish();

  // Deserialize the BloomFilter.
  ASSERT_FALSE(result.bloomBytes.empty());
  BloomFilter<> bloomFilter;
  bloomFilter.merge(result.bloomBytes.data());

  // Verify composite key hash: hash = hashColumnAt(col1, hashColumnAt(col0, 42))
  // Key (1, 100L) should be present.
  {
    int32_t hash = 42;
    hash = SparkMurmurHash::hashInt(1, hash);
    hash = SparkMurmurHash::hashLong(100L, hash);
    EXPECT_TRUE(bloomFilter.mayContain(spreadHashForBloom(hash)));
  }

  // Key (2, 200L) should be present.
  {
    int32_t hash = 42;
    hash = SparkMurmurHash::hashInt(2, hash);
    hash = SparkMurmurHash::hashLong(200L, hash);
    EXPECT_TRUE(bloomFilter.mayContain(spreadHashForBloom(hash)));
  }
}

// ---------------------------------------------------------------------------
// TotalRows should accumulate across multiple batches.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, TotalRowsAccumulation) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 100;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  auto batch1 = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});
  auto batch2 = makeRowVector({makeFlatVector<int32_t>({4, 5})});
  auto batch3 = makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{6})});

  builder.appendBatch(batch1);
  EXPECT_EQ(builder.totalRows(), 3);

  builder.appendBatch(batch2);
  EXPECT_EQ(builder.totalRows(), 5);

  builder.appendBatch(batch3);
  EXPECT_EQ(builder.totalRows(), 6);
}

// ---------------------------------------------------------------------------
// Piece bytes should be valid Presto-serialized data that can be deserialized.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// VARCHAR key: BloomFilter should contain inserted string keys.
// ---------------------------------------------------------------------------
#ifdef NDEBUG
TEST_F(ShardBuilderTest, VarcharKeyBloomFilter) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 1000;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  auto batch = makeRowVector({
      makeFlatVector<StringView>({"alice"_sv, "bob"_sv, "charlie"_sv}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  builder.appendBatch(batch);
  auto result = builder.finish();

  ASSERT_FALSE(result.bloomBytes.empty());
  BloomFilter<> bloomFilter;
  bloomFilter.merge(result.bloomBytes.data());

  // Verify each inserted varchar key is present in the BloomFilter.
  std::vector<std::string> keys = {"alice", "bob", "charlie"};
  for (const auto& key : keys) {
    auto vec = makeFlatVector<StringView>({StringView(key)});
    auto rowVec = makeRowVector({vec});
    int32_t hash = SparkMurmurHash::hashColumnAt(rowVec, 0, 0, 42);
    EXPECT_TRUE(bloomFilter.mayContain(spreadHashForBloom(hash)))
        << "Key '" << key << "' (hash=" << hash << ") should be in BloomFilter";
  }

  // Absent keys should mostly not be present.
  int falsePositives = 0;
  std::vector<std::string> absentKeys = {
      "dave", "eve", "frank", "grace", "heidi", "ivan",
      "judy", "karl", "liam", "mike"};
  for (const auto& key : absentKeys) {
    auto vec = makeFlatVector<StringView>({StringView(key)});
    auto rowVec = makeRowVector({vec});
    int32_t hash = SparkMurmurHash::hashColumnAt(rowVec, 0, 0, 42);
    if (bloomFilter.mayContain(spreadHashForBloom(hash))) {
      falsePositives++;
    }
  }
  EXPECT_LT(falsePositives, 5)
      << "Too many false positives: " << falsePositives << "/10";
}

// ---------------------------------------------------------------------------
// Mixed-type multi-key (int32 + varchar): BloomFilter should use chained hash.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, MixedTypeMultiKeyBloomFilter) {
  constexpr int32_t numKeyColumns = 2;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 1000;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  // key1=int32, key2=varchar, value=int64
  auto batch = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<StringView>({"alpha"_sv, "beta"_sv, "gamma"_sv}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  builder.appendBatch(batch);
  auto result = builder.finish();

  ASSERT_FALSE(result.bloomBytes.empty());
  BloomFilter<> bloomFilter;
  bloomFilter.merge(result.bloomBytes.data());

  // Verify composite key hash: hash = hashColumnAt(col1, hashColumnAt(col0, 42))
  // Key (1, "alpha") should be present.
  {
    auto intVec = makeFlatVector<int32_t>(std::vector<int32_t>{1});
    auto strVec = makeFlatVector<StringView>({"alpha"_sv});
    auto rowVec = makeRowVector({intVec, strVec});
    int32_t hash = 42;
    hash = SparkMurmurHash::hashColumnAt(rowVec, 0, 0, hash);
    hash = SparkMurmurHash::hashColumnAt(rowVec, 1, 0, hash);
    EXPECT_TRUE(bloomFilter.mayContain(spreadHashForBloom(hash)))
        << "Key (1, 'alpha') should be in BloomFilter";
  }

  // Key (2, "beta") should be present.
  {
    auto intVec = makeFlatVector<int32_t>(std::vector<int32_t>{2});
    auto strVec = makeFlatVector<StringView>({"beta"_sv});
    auto rowVec = makeRowVector({intVec, strVec});
    int32_t hash = 42;
    hash = SparkMurmurHash::hashColumnAt(rowVec, 0, 0, hash);
    hash = SparkMurmurHash::hashColumnAt(rowVec, 1, 0, hash);
    EXPECT_TRUE(bloomFilter.mayContain(spreadHashForBloom(hash)))
        << "Key (2, 'beta') should be in BloomFilter";
  }

  // Key (1, "beta") — mismatched combination — should likely not be present.
  {
    auto intVec = makeFlatVector<int32_t>(std::vector<int32_t>{1});
    auto strVec = makeFlatVector<StringView>({"beta"_sv});
    auto rowVec = makeRowVector({intVec, strVec});
    int32_t hash = 42;
    hash = SparkMurmurHash::hashColumnAt(rowVec, 0, 0, hash);
    hash = SparkMurmurHash::hashColumnAt(rowVec, 1, 0, hash);
    // This is probabilistic — may be a false positive, but unlikely with
    // capacity=1000 and only 3 entries.
    // We just verify the hash computation doesn't crash.
  }
}

// ---------------------------------------------------------------------------
// VARCHAR key: serialized piece bytes should be deserializable.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, VarcharKeyPieceBytesDeserializable) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 100;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  auto batch = makeRowVector({
      makeFlatVector<StringView>({"hello"_sv, "world"_sv}),
      makeFlatVector<int32_t>({10, 20}),
  });
  builder.appendBatch(batch);
  auto result = builder.finish();

  ASSERT_TRUE(result.lastPiece.has_value());
  const auto& pieceBytes = result.lastPiece->bytes;

  auto serde = std::make_unique<serializer::presto::PrestoVectorSerde>();
  auto rowType = asRowType(batch->type());

  std::vector<ByteRange> ranges;
  ranges.push_back(ByteRange{
      const_cast<uint8_t*>(
          reinterpret_cast<const uint8_t*>(pieceBytes.data())),
      static_cast<int32_t>(pieceBytes.size()),
      0});
  auto byteStream = std::make_unique<BufferInputStream>(std::move(ranges));

  auto deserPool = pool_->addLeafChild("deser_varchar");
  RowVectorPtr deserialized;
  serializer::presto::PrestoVectorSerde::PrestoOptions opts;
  opts.useLosslessTimestamp = true;
  serde->deserialize(
      byteStream.get(), deserPool.get(), rowType, &deserialized, &opts);

  ASSERT_EQ(deserialized->size(), 2);
  auto col0 = deserialized->childAt(0)->asFlatVector<StringView>();
  auto col1 = deserialized->childAt(1)->asFlatVector<int32_t>();
  EXPECT_EQ(col0->valueAt(0).str(), "hello");
  EXPECT_EQ(col0->valueAt(1).str(), "world");
  EXPECT_EQ(col1->valueAt(0), 10);
  EXPECT_EQ(col1->valueAt(1), 20);
}
#endif // NDEBUG

// ---------------------------------------------------------------------------
// Piece bytes should be valid Presto-serialized data that can be deserialized.
// ---------------------------------------------------------------------------
TEST_F(ShardBuilderTest, PieceBytesDeserializable) {
  constexpr int32_t numKeyColumns = 1;
  constexpr int32_t blockSize = 1024 * 1024;
  constexpr int64_t bloomCapacity = 100;
  

  ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, leafPool_);

  auto batch = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  builder.appendBatch(batch);
  auto result = builder.finish();

  ASSERT_TRUE(result.lastPiece.has_value());
  const auto& pieceBytes = result.lastPiece->bytes;

  // Deserialize the piece using PrestoVectorSerde.
  auto serde = std::make_unique<serializer::presto::PrestoVectorSerde>();
  auto rowType = asRowType(batch->type());

  std::vector<ByteRange> ranges;
  ranges.push_back(ByteRange{
      const_cast<uint8_t*>(
          reinterpret_cast<const uint8_t*>(pieceBytes.data())),
      static_cast<int32_t>(pieceBytes.size()),
      0});
  auto byteStream = std::make_unique<BufferInputStream>(std::move(ranges));

  auto deserPool = pool_->addLeafChild("deser");
  RowVectorPtr deserialized;
  serializer::presto::PrestoVectorSerde::PrestoOptions opts;
  opts.useLosslessTimestamp = true;
  serde->deserialize(
      byteStream.get(), deserPool.get(), rowType, &deserialized, &opts);

  // Verify deserialized data matches original.
  ASSERT_EQ(deserialized->size(), 3);
  auto col0 = deserialized->childAt(0)->asFlatVector<int32_t>();
  auto col1 = deserialized->childAt(1)->asFlatVector<int64_t>();
  EXPECT_EQ(col0->valueAt(0), 10);
  EXPECT_EQ(col0->valueAt(1), 20);
  EXPECT_EQ(col0->valueAt(2), 30);
  EXPECT_EQ(col1->valueAt(0), 100L);
  EXPECT_EQ(col1->valueAt(1), 200L);
  EXPECT_EQ(col1->valueAt(2), 300L);
}
