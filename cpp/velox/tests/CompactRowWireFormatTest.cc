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

#include "shard/CompactRowWireFormat.h"

#include <cstring>
#include <gtest/gtest.h>

#include "compute/VeloxBackend.h"
#include "velox/common/base/Exceptions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace gluten::shard;

// ---------------------------------------------------------------------------
// CompactRowWireFormatTest
//
// Exercises the gluten::shard::serializeCompactRowBatch /
// deserializeCompactRowBatch helpers that encapsulate the on-wire layout
// used by the ShardLookup gRPC service.  The test scope is intentionally
// narrow: the underlying CompactRow encoding itself is owned and tested by
// Velox; here we only validate the wrapper's header, framing, and error
// handling.
// ---------------------------------------------------------------------------
class CompactRowWireFormatTest : public ::testing::Test,
                                 public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    gluten::VeloxBackend::create({});
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addRootPool("compactWireTest");
    leafPool_ = pool_->addLeafChild("compactWireTestLeaf");
  }

  // Round-trip a RowVector through the wire format and assert that the
  // decoded vector is structurally and value-wise equal to the input.
  void roundTripAndAssertEqual(const RowVectorPtr& input) {
    auto wire = serializeCompactRowBatch(input, leafPool_.get());

    // Header sanity: magic + version + numRows must be at the front and
    // little-endian, even on an empty payload.
    ASSERT_GE(wire.size(), kCompactRowWireHeaderSize);
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t numRows = 0;
    std::memcpy(&magic, wire.data(), 4);
    std::memcpy(&version, wire.data() + 4, 4);
    std::memcpy(&numRows, wire.data() + 8, 4);
    EXPECT_EQ(magic, kCompactRowWireMagic);
    EXPECT_EQ(version, kCompactRowWireVersion);
    EXPECT_EQ(numRows, static_cast<uint32_t>(input->size()));

    auto decoded = deserializeCompactRowBatch(
        std::string_view(wire.data(), wire.size()),
        asRowType(input->type()),
        leafPool_.get());

    ASSERT_NE(decoded, nullptr);
    ASSERT_EQ(decoded->size(), input->size());
    ASSERT_EQ(decoded->childrenSize(), input->childrenSize());

    // Use Velox's value-equality utility (handles nulls, complex types).
    test::assertEqualVectors(input, decoded);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
};

// ---------------------------------------------------------------------------
// Round-trip: empty batch
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, EmptyBatchRoundTrip) {
  auto rowType = ROW({"c0"}, {INTEGER()});
  auto empty = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(rowType, 0, leafPool_.get()));

  auto wire = serializeCompactRowBatch(empty, leafPool_.get());
  // Empty payload is exactly the header.
  EXPECT_EQ(wire.size(), kCompactRowWireHeaderSize);

  auto decoded = deserializeCompactRowBatch(
      std::string_view(wire.data(), wire.size()),
      rowType,
      leafPool_.get());
  ASSERT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->size(), 0);
}

// ---------------------------------------------------------------------------
// Round-trip: nullptr input is treated as empty.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, NullVectorTreatedAsEmpty) {
  auto wire = serializeCompactRowBatch(nullptr, leafPool_.get());
  EXPECT_EQ(wire.size(), kCompactRowWireHeaderSize);

  uint32_t numRows = 0;
  std::memcpy(&numRows, wire.data() + 8, 4);
  EXPECT_EQ(numRows, 0u);
}

// ---------------------------------------------------------------------------
// Round-trip: single fixed-width column (fast path via fixedRowSize).
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, SingleIntegerColumnRoundTrip) {
  auto input = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
  });
  roundTripAndAssertEqual(input);
}

// ---------------------------------------------------------------------------
// Round-trip: multiple fixed-width columns covering the integer/float matrix.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, MultiFixedWidthColumnsRoundTrip) {
  auto input = makeRowVector({
      makeFlatVector<int8_t>({1, 2, 3}),
      makeFlatVector<int16_t>({10, 20, 30}),
      makeFlatVector<int32_t>({100, 200, 300}),
      makeFlatVector<int64_t>({1000L, 2000L, 3000L}),
      makeFlatVector<float>({1.5f, 2.5f, 3.5f}),
      makeFlatVector<double>({1.25, 2.25, 3.25}),
      makeFlatVector<bool>({true, false, true}),
  });
  roundTripAndAssertEqual(input);
}

// ---------------------------------------------------------------------------
// Round-trip: nullable fixed-width columns (variable-width fallback path
// because nulls disable the fixedRowSize fast path? Actually nulls do NOT
// disable it — fixedRowSize is purely type-based.  We still want explicit
// coverage that the per-row null bits are encoded/decoded correctly.)
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, NullableFixedWidthRoundTrip) {
  auto input = makeRowVector({
      makeNullableFlatVector<int32_t>({10, std::nullopt, 30, std::nullopt, 50}),
      makeNullableFlatVector<int64_t>(
          {std::nullopt, 200L, std::nullopt, 400L, 500L}),
  });
  roundTripAndAssertEqual(input);
}

// ---------------------------------------------------------------------------
// Round-trip: variable-width (VARCHAR) column — exercises the per-row
// rowSize() loop instead of the fixedRowSize fast path.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, VarcharColumnRoundTrip) {
  auto input = makeRowVector({
      makeFlatVector<StringView>(
          {"alice"_sv, "bob"_sv, "charlie"_sv, ""_sv, "x"_sv}),
      makeFlatVector<int64_t>({1L, 2L, 3L, 4L, 5L}),
  });
  roundTripAndAssertEqual(input);
}

// ---------------------------------------------------------------------------
// Round-trip: nullable VARCHAR — null + empty + non-empty mix.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, NullableVarcharRoundTrip) {
  auto input = makeRowVector({
      makeNullableFlatVector<StringView>({
          "hello"_sv,
          std::nullopt,
          ""_sv,
          "world"_sv,
          std::nullopt,
      }),
  });
  roundTripAndAssertEqual(input);
}

// ---------------------------------------------------------------------------
// Round-trip: TIMESTAMP column at microsecond precision.
//
// CompactRow encodes Timestamp as int64 microseconds (see Velox
// CompactRow.cpp::writeTimestamp).  This matches Spark's Timestamp type,
// which is also microsecond-precision.  Inputs whose nanosecond field is
// not a whole number of microseconds will be truncated by the encoder —
// see TimestampColumnTruncatesSubMicros below.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, TimestampColumnRoundTrip) {
  auto input = makeRowVector({
      makeFlatVector<Timestamp>({
          Timestamp(1700000000, 0),
          // 123456 us == 123'456'000 ns — round-trips cleanly.
          Timestamp(1700000001, 123'456'000),
          Timestamp(0, 0),
      }),
  });
  roundTripAndAssertEqual(input);
}

// ---------------------------------------------------------------------------
// CompactRow encodes Timestamp as int64 microseconds, so any sub-microsecond
// nanos in the input are silently dropped.  This is a Velox / Spark wire
// contract, not a wire-format bug; we lock it down explicitly so that any
// future "upgrade to nanos" change shows up as a deliberate test failure.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, TimestampColumnTruncatesSubMicros) {
  auto input = makeRowVector({
      makeFlatVector<Timestamp>({
          Timestamp(1700000001, 123'456'789),
      }),
  });

  auto wire = serializeCompactRowBatch(input, leafPool_.get());
  auto decoded = deserializeCompactRowBatch(
      std::string_view(wire.data(), wire.size()),
      asRowType(input->type()),
      leafPool_.get());

  ASSERT_NE(decoded, nullptr);
  ASSERT_EQ(decoded->size(), 1);
  auto decodedTs =
      decoded->childAt(0)->asFlatVector<Timestamp>()->valueAt(0);
  // Sub-microsecond nanos (789 ns) are dropped; the rest survives.
  EXPECT_EQ(decodedTs, Timestamp(1700000001, 123'456'000));
}

// ---------------------------------------------------------------------------
// Round-trip: composite key (multiple columns, mixed widths).
// Validates that per-row length prefixes correctly delimit rows of varying
// sizes.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, MixedWidthCompositeKeyRoundTrip) {
  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<StringView>({"short"_sv, "a much longer string"_sv, ""_sv}),
      makeFlatVector<int64_t>({100L, 200L, 300L}),
  });
  roundTripAndAssertEqual(input);
}

// ---------------------------------------------------------------------------
// Header validation: payload shorter than the header must fail cleanly.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, RejectTooShortPayload) {
  auto rowType = ROW({"c0"}, {INTEGER()});
  std::string tooShort(4, '\0'); // less than kCompactRowWireHeaderSize.
  EXPECT_THROW(
      deserializeCompactRowBatch(
          std::string_view(tooShort.data(), tooShort.size()),
          rowType,
          leafPool_.get()),
      VeloxException);
}

// ---------------------------------------------------------------------------
// Header validation: bad magic must fail with a descriptive error.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, RejectBadMagic) {
  auto input = makeRowVector({makeFlatVector<int32_t>({42})});
  auto wire = serializeCompactRowBatch(input, leafPool_.get());

  // Corrupt the magic.
  uint32_t badMagic = 0xDEADBEEFu;
  std::memcpy(wire.data(), &badMagic, sizeof(badMagic));

  EXPECT_THROW(
      deserializeCompactRowBatch(
          std::string_view(wire.data(), wire.size()),
          asRowType(input->type()),
          leafPool_.get()),
      VeloxException);
}

// ---------------------------------------------------------------------------
// Header validation: unsupported version must fail with a descriptive error.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, RejectBadVersion) {
  auto input = makeRowVector({makeFlatVector<int32_t>({42})});
  auto wire = serializeCompactRowBatch(input, leafPool_.get());

  // Corrupt the version (any value != kCompactRowWireVersion).
  uint32_t badVersion = kCompactRowWireVersion + 999u;
  std::memcpy(wire.data() + 4, &badVersion, sizeof(badVersion));

  EXPECT_THROW(
      deserializeCompactRowBatch(
          std::string_view(wire.data(), wire.size()),
          asRowType(input->type()),
          leafPool_.get()),
      VeloxException);
}

// ---------------------------------------------------------------------------
// Truncation: payload claims N rows but body is missing trailing rows.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, RejectTruncatedBody) {
  auto input = makeRowVector({makeFlatVector<int32_t>({10, 20, 30})});
  auto wire = serializeCompactRowBatch(input, leafPool_.get());

  // Drop the last 4 bytes — guaranteed to chop into a row body.
  ASSERT_GT(wire.size(), 4u);
  std::string truncated = wire.substr(0, wire.size() - 4);

  EXPECT_THROW(
      deserializeCompactRowBatch(
          std::string_view(truncated.data(), truncated.size()),
          asRowType(input->type()),
          leafPool_.get()),
      VeloxException);
}

// ---------------------------------------------------------------------------
// Truncation: per-row length prefix points past end-of-payload.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, RejectOverlongRowLength) {
  auto input = makeRowVector({makeFlatVector<int32_t>({42})});
  auto wire = serializeCompactRowBatch(input, leafPool_.get());

  // The first row's length prefix sits right after the 12-byte header.
  // Rewrite it to a value larger than the remaining wire bytes.
  uint32_t overlong = static_cast<uint32_t>(wire.size()) + 100u;
  std::memcpy(wire.data() + kCompactRowWireHeaderSize, &overlong, sizeof(overlong));

  EXPECT_THROW(
      deserializeCompactRowBatch(
          std::string_view(wire.data(), wire.size()),
          asRowType(input->type()),
          leafPool_.get()),
      VeloxException);
}

// ---------------------------------------------------------------------------
// Trailing garbage: payload contains extra bytes after the last row.
// We choose to be strict about this — silent truncation could mask bugs
// where the encoder over-allocated.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, RejectTrailingGarbage) {
  auto input = makeRowVector({makeFlatVector<int32_t>({1, 2})});
  auto wire = serializeCompactRowBatch(input, leafPool_.get());

  // Append 8 bytes of garbage.
  wire.append(8, '\xAB');

  EXPECT_THROW(
      deserializeCompactRowBatch(
          std::string_view(wire.data(), wire.size()),
          asRowType(input->type()),
          leafPool_.get()),
      VeloxException);
}

// ---------------------------------------------------------------------------
// Argument validation: null rowType and null pool must be rejected up-front.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, RejectNullArguments) {
  auto input = makeRowVector({makeFlatVector<int32_t>({1})});
  auto wire = serializeCompactRowBatch(input, leafPool_.get());
  std::string_view view(wire.data(), wire.size());

  EXPECT_THROW(
      deserializeCompactRowBatch(view, /*rowType=*/nullptr, leafPool_.get()),
      VeloxException);

  EXPECT_THROW(
      deserializeCompactRowBatch(
          view, asRowType(input->type()), /*pool=*/nullptr),
      VeloxException);
}

// ---------------------------------------------------------------------------
// Sanity: the on-wire encoding of magic / version is little-endian.
// This is a guardrail against future endian-conversion regressions.
// ---------------------------------------------------------------------------
TEST_F(CompactRowWireFormatTest, HeaderIsLittleEndian) {
  auto wire = serializeCompactRowBatch(nullptr, leafPool_.get());
  ASSERT_EQ(wire.size(), kCompactRowWireHeaderSize);

  // Magic 0x434F4D31 in little-endian on disk: 31 4D 4F 43.
  EXPECT_EQ(static_cast<uint8_t>(wire[0]), 0x31);
  EXPECT_EQ(static_cast<uint8_t>(wire[1]), 0x4D);
  EXPECT_EQ(static_cast<uint8_t>(wire[2]), 0x4F);
  EXPECT_EQ(static_cast<uint8_t>(wire[3]), 0x43);
}

