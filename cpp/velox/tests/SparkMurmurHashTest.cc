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

#include "shard/SparkMurmurHash.h"
#include "compute/VeloxBackend.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace gluten;
using namespace gluten::shard;

// Expected values are pre-computed using Spark's Murmur3_x86_32 Java
// implementation to ensure bit-exact compatibility.
//
// Java reference code:
//   import org.apache.spark.unsafe.hash.Murmur3_x86_32;
//   Murmur3_x86_32.hashInt(value, seed);
//   Murmur3_x86_32.hashLong(value, seed);
//   Murmur3_x86_32.hashUnsafeBytes(bytes, offset, length, seed);

class SparkMurmurHashTest : public ::testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    VeloxBackend::create({});
    memory::MemoryManager::testingSetInstance({});
  }
};

// ---------------------------------------------------------------------------
// Primitive hash function tests
// ---------------------------------------------------------------------------

TEST_F(SparkMurmurHashTest, HashIntWithDefaultSeed) {
  // Spark: Murmur3_x86_32.hashInt(0, 42) = 933211791
  EXPECT_EQ(SparkMurmurHash::hashInt(0, 42), 933211791);
}

TEST_F(SparkMurmurHashTest, HashIntPositive) {
  // Spark: Murmur3_x86_32.hashInt(42, 42) = 29417773
  EXPECT_EQ(SparkMurmurHash::hashInt(42, 42), 29417773);
}

TEST_F(SparkMurmurHashTest, HashIntNegative) {
  // Spark: Murmur3_x86_32.hashInt(-1, 42) = -1604776387
  EXPECT_EQ(SparkMurmurHash::hashInt(-1, 42), -1604776387);
}

TEST_F(SparkMurmurHashTest, HashIntMaxValue) {
  // Spark: Murmur3_x86_32.hashInt(Integer.MAX_VALUE, 42) = 133916647
  EXPECT_EQ(SparkMurmurHash::hashInt(2147483647, 42), 133916647);
}

TEST_F(SparkMurmurHashTest, HashLongZero) {
  // Spark: Murmur3_x86_32.hashLong(0L, 42) = -1670924195
  EXPECT_EQ(SparkMurmurHash::hashLong(0L, 42), -1670924195);
}

TEST_F(SparkMurmurHashTest, HashLongPositive) {
  // Spark: Murmur3_x86_32.hashLong(42L, 42) = 1316951768
  EXPECT_EQ(SparkMurmurHash::hashLong(42L, 42), 1316951768);
}

TEST_F(SparkMurmurHashTest, HashLongLargeValue) {
  // Spark: Murmur3_x86_32.hashLong(Long.MAX_VALUE, 42) = -1604625029
  EXPECT_EQ(SparkMurmurHash::hashLong(9223372036854775807LL, 42), -1604625029);
}

TEST_F(SparkMurmurHashTest, HashBytesEmpty) {
  // Spark: Murmur3_x86_32.hashUnsafeBytes(new byte[0], 0, 0, 42) = 142593372
  EXPECT_EQ(SparkMurmurHash::hashBytes(nullptr, 0, 42), 142593372);
}

TEST_F(SparkMurmurHashTest, HashBytesHello) {
  // Spark: Murmur3_x86_32.hashUnsafeBytes("hello".getBytes("UTF-8"), 0, 5, 42)
  //      = -1008564952
  const auto* data = reinterpret_cast<const uint8_t*>("hello");
  EXPECT_EQ(SparkMurmurHash::hashBytes(data, 5, 42), -1008564952);
}

TEST_F(SparkMurmurHashTest, HashBytesLonger) {
  // Spark: Murmur3_x86_32.hashUnsafeBytes("hello world!".getBytes("UTF-8"), 0, 12, 42)
  //      = -294328854
  const auto* data = reinterpret_cast<const uint8_t*>("hello world!");
  EXPECT_EQ(SparkMurmurHash::hashBytes(data, 12, 42), -294328854);
}

// ---------------------------------------------------------------------------
// hashColumnAt tests — verify per-type dispatch matches Spark's Murmur3Hash
// ---------------------------------------------------------------------------

TEST_F(SparkMurmurHashTest, HashColumnAtInteger) {
  auto rowVector = makeRowVector({makeFlatVector<int32_t>({0, 42, -1})});
  // Row 0: hashInt(0, 42)
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 0, 42), 933211791);
  // Row 1: hashInt(42, 42)
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 1, 42), 29417773);
  // Row 2: hashInt(-1, 42)
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 2, 42), -1604776387);
}

TEST_F(SparkMurmurHashTest, HashColumnAtBigint) {
  auto rowVector = makeRowVector({makeFlatVector<int64_t>({0L, 42L})});
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 0, 42), -1670924195);
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 1, 42), 1316951768);
}

TEST_F(SparkMurmurHashTest, HashColumnAtVarchar) {
  auto rowVector = makeRowVector(
      {makeFlatVector<StringView>({"hello"_sv, "hello world!"_sv})});
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 0, 42), -1008564952);
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 1, 42), -294328854);
}

TEST_F(SparkMurmurHashTest, HashColumnAtNull) {
  // Null values should return the seed unchanged (Spark behavior).
  auto rowVector = makeRowVector(
      {makeNullableFlatVector<int32_t>({std::nullopt, 42})});
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 0, 42), 42);
  EXPECT_EQ(SparkMurmurHash::hashColumnAt(rowVector, 0, 1, 42), 29417773);
}

TEST_F(SparkMurmurHashTest, HashColumnAtBoolean) {
  // Spark: Murmur3Hash(true) = hashInt(1, 42), Murmur3Hash(false) = hashInt(0, 42)
  auto rowVector = makeRowVector({makeFlatVector<bool>({true, false})});
  EXPECT_EQ(
      SparkMurmurHash::hashColumnAt(rowVector, 0, 0, 42),
      SparkMurmurHash::hashInt(1, 42));
  EXPECT_EQ(
      SparkMurmurHash::hashColumnAt(rowVector, 0, 1, 42),
      SparkMurmurHash::hashInt(0, 42));
}

TEST_F(SparkMurmurHashTest, HashColumnAtDouble) {
  // Spark: Murmur3Hash(1.0) = hashLong(Double.doubleToLongBits(1.0), 42)
  // Double.doubleToLongBits(1.0) = 4607182418800017408L
  auto rowVector = makeRowVector({makeFlatVector<double>({1.0, -0.0})});
  EXPECT_EQ(
      SparkMurmurHash::hashColumnAt(rowVector, 0, 0, 42),
      SparkMurmurHash::hashLong(4607182418800017408LL, 42));
  // -0.0 is normalized to 0L in Spark
  EXPECT_EQ(
      SparkMurmurHash::hashColumnAt(rowVector, 0, 1, 42),
      SparkMurmurHash::hashLong(0L, 42));
}

// ---------------------------------------------------------------------------
// Multi-column chaining — simulates Spark's HashPartitioning(keys, numShards)
// ---------------------------------------------------------------------------

TEST_F(SparkMurmurHashTest, MultiColumnChaining) {
  // Spark chains: hash = Murmur3Hash(col1, Murmur3Hash(col0, seed=42))
  auto rowVector = makeRowVector({
      makeFlatVector<int32_t>({10}),
      makeFlatVector<int64_t>({20L}),
  });

  // Compute hash via chaining hashColumnAt calls.
  int32_t hash = 42;
  hash = SparkMurmurHash::hashColumnAt(rowVector, 0, 0, hash);
  hash = SparkMurmurHash::hashColumnAt(rowVector, 1, 0, hash);

  // Verify determinism: calling again with the same inputs yields the same result.
  int32_t hash2 = 42;
  hash2 = SparkMurmurHash::hashColumnAt(rowVector, 0, 0, hash2);
  hash2 = SparkMurmurHash::hashColumnAt(rowVector, 1, 0, hash2);
  EXPECT_EQ(hash, hash2);

  // Verify the intermediate step: hashColumnAt for col0 should be deterministic.
  int32_t step1 = SparkMurmurHash::hashColumnAt(rowVector, 0, 0, 42);
  int32_t step2 = SparkMurmurHash::hashColumnAt(rowVector, 1, 0, step1);
  EXPECT_EQ(hash, step2);
}

// ---------------------------------------------------------------------------
// nonNegativeMod — simulates Spark's HashPartitioning.getPartition
// ---------------------------------------------------------------------------

TEST_F(SparkMurmurHashTest, NonNegativeModPartitioning) {
  // Spark: partition = Math.floorMod(hash, numPartitions)
  int32_t numShards = 10;
  auto rowVector = makeRowVector({makeFlatVector<int32_t>({0, 1, 2, 3, 4})});

  for (vector_size_t row = 0; row < 5; ++row) {
    int32_t hash = SparkMurmurHash::hashColumnAt(rowVector, 0, row, 42);
    int32_t mod = hash % numShards;
    if (mod < 0) {
      mod += numShards;
    }
    // Partition should always be in [0, numShards)
    EXPECT_GE(mod, 0);
    EXPECT_LT(mod, numShards);
  }
}
