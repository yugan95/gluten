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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sstream>

#include "BloomFilter64.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/StreamArena.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/ComplexVector.h"

#include "BloomHashUtil.h"
#include "SparkMurmurHash.h"

namespace gluten {
namespace shard {

using namespace facebook::velox;

// ---------------------------------------------------------------------------
// ShardBuilder – streams ColumnarBatches into BlockManager pieces + BloomFilter.
//
// Mirrors vanilla Spark's ShardManager.writeShardBlock() pattern:
//   - Serializes batches to Presto format, flushing a "piece" every blockSize
//   - Simultaneously builds a BloomFilter from key columns
//   - Computes Adler32 checksum per piece
//
// Usage:
//   ShardBuilder builder(numKeyColumns, blockSize, bloomCapacity, fpp, pool);
//   for each batch:
//     auto piece = builder.appendBatch(rowVector);
//     if (piece) writePieceToBM(*piece);
//   auto result = builder.finish();
//   if (result.lastPiece) writePieceToBM(*result.lastPiece);
//   writeMetaToBM(result.pieceCount, result.checksums);
//   writeBloomToBM(result.bloomBytes);
// ---------------------------------------------------------------------------
class ShardBuilder {
 public:
  struct PieceData {
    std::string bytes;
    uint32_t checksum;
  };

  struct FinishResult {
    std::optional<PieceData> lastPiece;
    std::string bloomBytes;
    std::vector<uint32_t> checksums;
    int32_t totalPieceCount;
  };

  ShardBuilder(
      int32_t numKeyColumns,
      int32_t blockSize,
      int64_t bloomCapacity,
      std::shared_ptr<memory::MemoryPool> pool)
      : numKeyColumns_(numKeyColumns),
        blockSize_(blockSize),
        pool_(std::move(pool)) {
    // bloomCapacity is the desired number of entries for the BloomFilter.
    // BloomFilter::reset() internally sizes the bit array to ~2 bytes per
    // entry (8 bits/entry → ~2% FPR with 4 hash probes per word).
    // The caller should set bloomCapacity = expectedTotalKeys across all
    // shards so that after merging (bitwise OR), the FPR stays low.
    bloomFilter_.reset(static_cast<int32_t>(bloomCapacity));
  }

  // Append a batch.  If the internal buffer exceeds blockSize after appending,
  // returns a PieceData containing the flushed piece bytes and its checksum.
  // Otherwise returns std::nullopt.
  std::optional<PieceData> appendBatch(const RowVectorPtr& batch) {
    if (!batch || batch->size() == 0) {
      return std::nullopt;
    }

    // Serialize this batch to Presto format.
    auto serialized = serializeBatch(batch);
    buffer_.append(serialized);
    totalRows_ += batch->size();

    // Insert key columns into BloomFilter.
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      int32_t hash = 42; // Spark Murmur3 seed
      for (int32_t col = 0; col < numKeyColumns_; ++col) {
        hash = SparkMurmurHash::hashColumnAt(batch, col, row, hash);
      }
      bloomFilter_.insert(spreadHashForBloom(hash));
    }

    // Check if buffer exceeds blockSize – if so, flush a piece.
    // We flush the entire buffer as one piece (it contains complete batches).
    if (static_cast<int32_t>(buffer_.size()) >= blockSize_) {
      auto checksum = computeAdler32(buffer_);
      PieceData piece{std::move(buffer_), checksum};
      buffer_.clear();
      checksums_.push_back(checksum);
      pieceCount_++;
      return piece;
    }

    return std::nullopt;
  }

  // Finish building: flush remaining buffer and return bloom filter bytes.
  FinishResult finish() {
    FinishResult result;

    // Flush remaining buffer as the last piece.
    if (!buffer_.empty()) {
      auto checksum = computeAdler32(buffer_);
      checksums_.push_back(checksum);
      pieceCount_++;
      result.lastPiece = PieceData{std::move(buffer_), checksum};
      buffer_.clear();
    }

    // Serialize BloomFilter.
    if (bloomFilter_.isSet()) {
      auto serializedSize = bloomFilter_.serializedSize();
      result.bloomBytes.resize(serializedSize);
      bloomFilter_.serialize(result.bloomBytes.data());
    }

    result.checksums = checksums_;
    result.totalPieceCount = pieceCount_;
    return result;
  }

  int64_t totalRows() const {
    return totalRows_;
  }

 private:
  std::string serializeBatch(const RowVectorPtr& batch) {
    auto serde =
        std::make_unique<serializer::presto::PrestoVectorSerde>();
    auto numRows = batch->size();
    auto rowType = asRowType(batch->type());
    auto arena = std::make_unique<StreamArena>(pool_.get());
    serializer::presto::PrestoVectorSerde::PrestoOptions opts;
    opts.useLosslessTimestamp = true;

    auto serializer =
        serde->createIterativeSerializer(rowType, numRows, arena.get(), &opts);

    serializer->append(batch);

    std::ostringstream oss;
    auto outputStream = std::make_unique<OStreamOutputStream>(&oss);
    serializer->flush(outputStream.get());
    return oss.str();
  }

  static uint32_t computeAdler32(const std::string& data) {
    // Adler-32 implementation matching java.util.zip.Adler32
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < data.size(); ++i) {
      a = (a + static_cast<uint8_t>(data[i])) % 65521;
      b = (b + a) % 65521;
    }
    return (b << 16) | a;
  }

  int32_t numKeyColumns_;
  int32_t blockSize_;
  std::shared_ptr<memory::MemoryPool> pool_;

  BloomFilter64 bloomFilter_;
  std::string buffer_;
  std::vector<uint32_t> checksums_;
  int32_t pieceCount_{0};
  int64_t totalRows_{0};
};

} // namespace shard
} // namespace gluten
