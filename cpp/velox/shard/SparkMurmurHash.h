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
#include <cstring>

#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SimpleVector.h"

namespace gluten {
namespace shard {

/**
 * Spark-compatible Murmur3_x86_32 hash functions.
 *
 * These functions replicate the exact behavior of Spark's
 * org.apache.spark.unsafe.hash.Murmur3_x86_32 so that the hash values
 * produced here match those produced by Spark's HashPartitioning (seed=42).
 */
struct SparkMurmurHash {
  static uint32_t murmur3MixK1(uint32_t k1) {
    k1 *= 0xcc9e2d51u;
    k1 = (k1 << 15) | (k1 >> 17);
    k1 *= 0x1b873593u;
    return k1;
  }

  static uint32_t murmur3MixH1(uint32_t h1, uint32_t k1) {
    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >> 19);
    h1 = h1 * 5 + 0xe6546b64u;
    return h1;
  }

  static uint32_t murmur3Fmix(uint32_t h1, uint32_t length) {
    h1 ^= length;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6bu;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35u;
    h1 ^= h1 >> 16;
    return h1;
  }

  static int32_t hashInt(int32_t input, int32_t seed) {
    auto k1 = murmur3MixK1(static_cast<uint32_t>(input));
    auto h1 = murmur3MixH1(static_cast<uint32_t>(seed), k1);
    return static_cast<int32_t>(murmur3Fmix(h1, 4));
  }

  static int32_t hashLong(int64_t input, int32_t seed) {
    auto low = static_cast<uint32_t>(input);
    auto high = static_cast<uint32_t>(static_cast<uint64_t>(input) >> 32);
    auto k1 = murmur3MixK1(low);
    auto h1 = murmur3MixH1(static_cast<uint32_t>(seed), k1);
    k1 = murmur3MixK1(high);
    h1 = murmur3MixH1(h1, k1);
    return static_cast<int32_t>(murmur3Fmix(h1, 8));
  }

  static int32_t hashBytes(
      const uint8_t* data,
      int32_t length,
      int32_t seed) {
    auto h1 = static_cast<uint32_t>(seed);
    int32_t lengthAligned = length - (length % 4);
    for (int32_t i = 0; i < lengthAligned; i += 4) {
      uint32_t halfWord;
      std::memcpy(&halfWord, data + i, sizeof(uint32_t));
      h1 = murmur3MixH1(h1, murmur3MixK1(halfWord));
    }
    for (int32_t i = lengthAligned; i < length; ++i) {
      auto halfWord = static_cast<uint32_t>(
          static_cast<int32_t>(static_cast<int8_t>(data[i])));
      h1 = murmur3MixH1(h1, murmur3MixK1(halfWord));
    }
    return static_cast<int32_t>(murmur3Fmix(h1, static_cast<uint32_t>(length)));
  }

  /**
   * Compute Spark Murmur3Hash for a single column value at the given row,
   * chaining with the running seed.
   *
   * @param rowVector The RowVector containing the data
   * @param colIndex Column index within the RowVector
   * @param row Row index
   * @param seed Running hash seed
   * @return Updated hash value
   */
  static int32_t hashColumnAt(
      const facebook::velox::RowVectorPtr& rowVector,
      size_t colIndex,
      facebook::velox::vector_size_t row,
      int32_t seed) {
    using namespace facebook::velox;
    auto child = rowVector->childAt(colIndex);
    if (child->isNullAt(row)) {
      return seed;
    }
    auto typeKind = child->typeKind();
    switch (typeKind) {
      case TypeKind::BOOLEAN:
        return hashInt(
            child->as<SimpleVector<bool>>()->valueAt(row) ? 1 : 0, seed);
      case TypeKind::TINYINT:
        return hashInt(
            child->as<SimpleVector<int8_t>>()->valueAt(row), seed);
      case TypeKind::SMALLINT:
        return hashInt(
            child->as<SimpleVector<int16_t>>()->valueAt(row), seed);
      case TypeKind::INTEGER:
        return hashInt(
            child->as<SimpleVector<int32_t>>()->valueAt(row), seed);
      case TypeKind::BIGINT:
        return hashLong(
            child->as<SimpleVector<int64_t>>()->valueAt(row), seed);
      case TypeKind::REAL: {
        float f = child->as<SimpleVector<float>>()->valueAt(row);
        int32_t bits;
        if (f == -0.0f) {
          bits = 0;
        } else {
          std::memcpy(&bits, &f, sizeof(int32_t));
        }
        return hashInt(bits, seed);
      }
      case TypeKind::DOUBLE: {
        double d = child->as<SimpleVector<double>>()->valueAt(row);
        int64_t bits;
        if (d == -0.0) {
          bits = 0;
        } else {
          std::memcpy(&bits, &d, sizeof(int64_t));
        }
        return hashLong(bits, seed);
      }
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY: {
        auto sv = child->as<SimpleVector<StringView>>()->valueAt(row);
        return hashBytes(
            reinterpret_cast<const uint8_t*>(sv.data()),
            static_cast<int32_t>(sv.size()),
            seed);
      }
      default:
        VELOX_UNSUPPORTED(
            "Unsupported type for Spark Murmur3Hash: {}",
            mapTypeKindToName(typeKind));
    }
  }

  /**
   * Batch-compute Spark Murmur3Hash for a single column across a range of
   * rows.  Each row's hash is chained with its per-row seed from `hashes`.
   *
   * This amortizes virtual dispatch, type switch, and SimpleVector resolution
   * per column (once), then tight-loops over rows with direct value access.
   * ~2-5x faster than calling hashColumnAt per row.
   *
   * @param rowVector The RowVector containing the data
   * @param colIndex Column index within the RowVector
   * @param startRow First row index (inclusive)
   * @param endRow Last row index (exclusive)
   * @param hashes In/out: running hash seeds on entry, updated hashes on exit
   */
  static void hashColumnBatch(
      const facebook::velox::RowVectorPtr& rowVector,
      size_t colIndex,
      facebook::velox::vector_size_t startRow,
      facebook::velox::vector_size_t endRow,
      int32_t* hashes) {
    using namespace facebook::velox;
    auto child = rowVector->childAt(colIndex);
    auto typeKind = child->typeKind();

    // Resolve the typed vector once, then tight-loop over all rows.
    switch (typeKind) {
      case TypeKind::BOOLEAN: {
        auto* typed = child->as<SimpleVector<bool>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashInt(typed->valueAt(row) ? 1 : 0, hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::TINYINT: {
        auto* typed = child->as<SimpleVector<int8_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashInt(typed->valueAt(row), hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::SMALLINT: {
        auto* typed = child->as<SimpleVector<int16_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashInt(typed->valueAt(row), hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::INTEGER: {
        auto* typed = child->as<SimpleVector<int32_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashInt(typed->valueAt(row), hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::BIGINT: {
        auto* typed = child->as<SimpleVector<int64_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashLong(typed->valueAt(row), hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::REAL: {
        auto* typed = child->as<SimpleVector<float>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            float f = typed->valueAt(row);
            int32_t bits;
            if (f == -0.0f) {
              bits = 0;
            } else {
              std::memcpy(&bits, &f, sizeof(int32_t));
            }
            hashes[idx] = hashInt(bits, hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::DOUBLE: {
        auto* typed = child->as<SimpleVector<double>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            double d = typed->valueAt(row);
            int64_t bits;
            if (d == -0.0) {
              bits = 0;
            } else {
              std::memcpy(&bits, &d, sizeof(int64_t));
            }
            hashes[idx] = hashLong(bits, hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY: {
        auto* typed = child->as<SimpleVector<StringView>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            auto sv = typed->valueAt(row);
            hashes[idx] = hashBytes(
                reinterpret_cast<const uint8_t*>(sv.data()),
                static_cast<int32_t>(sv.size()),
                hashes[idx]);
          }
        }
        break;
      }
      default:
        VELOX_UNSUPPORTED(
            "Unsupported type for Spark Murmur3Hash batch: {}",
            mapTypeKindToName(typeKind));
    }
  }

  /**
   * Batch-compute Spark Murmur3Hash for a single column, casting to bigint
   * before hashing when `castToBigint` is true.  This matches the build-side
   * HashPartitioning behavior where buildBoundKeys may include
   * Cast(key, LongType).
   *
   * @param rowVector The RowVector containing the data
   * @param colIndex Column index within the RowVector
   * @param startRow First row index (inclusive)
   * @param endRow Last row index (exclusive)
   * @param castToBigint Whether to cast values to int64 before hashing
   * @param hashes In/out: running hash seeds on entry, updated hashes on exit
   */
  static void hashColumnBatchForShard(
      const facebook::velox::RowVectorPtr& rowVector,
      size_t colIndex,
      facebook::velox::vector_size_t startRow,
      facebook::velox::vector_size_t endRow,
      bool castToBigint,
      int32_t* hashes) {
    if (!castToBigint) {
      hashColumnBatch(rowVector, colIndex, startRow, endRow, hashes);
      return;
    }

    using namespace facebook::velox;
    auto child = rowVector->childAt(colIndex);
    auto typeKind = child->typeKind();

    // Cast to bigint (hashLong) for integer types smaller than BIGINT.
    switch (typeKind) {
      case TypeKind::BOOLEAN: {
        auto* typed = child->as<SimpleVector<bool>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashLong(
                typed->valueAt(row) ? 1L : 0L, hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::TINYINT: {
        auto* typed = child->as<SimpleVector<int8_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashLong(
                static_cast<int64_t>(typed->valueAt(row)), hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::SMALLINT: {
        auto* typed = child->as<SimpleVector<int16_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashLong(
                static_cast<int64_t>(typed->valueAt(row)), hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::INTEGER: {
        auto* typed = child->as<SimpleVector<int32_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashLong(
                static_cast<int64_t>(typed->valueAt(row)), hashes[idx]);
          }
        }
        break;
      }
      case TypeKind::BIGINT: {
        auto* typed = child->as<SimpleVector<int64_t>>();
        for (auto row = startRow; row < endRow; ++row) {
          auto idx = row - startRow;
          if (!typed->isNullAt(row)) {
            hashes[idx] = hashLong(typed->valueAt(row), hashes[idx]);
          }
        }
        break;
      }
      default:
        // For non-integer types, fall back to standard hash (no cast needed).
        hashColumnBatch(rowVector, colIndex, startRow, endRow, hashes);
        break;
    }
  }
};

} // namespace shard
} // namespace gluten
