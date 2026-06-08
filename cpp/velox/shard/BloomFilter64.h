#pragma once

#include <cstdint>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"

namespace gluten {
namespace shard {

// BloomFilter with 64-bit-safe merge.  Identical to Velox BloomFilter
// except merge() uses a direct loop instead of bits::orBits, avoiding
// int32 overflow when 64 * size >= 2^31 (capacity > 2^26).
// Serialization format is wire-compatible with Velox BloomFilter v1.
class BloomFilter64 {
 public:
  void reset(int64_t capacity) {
    bits_.clear();
    bits_.resize(std::max<int64_t>(
        4, facebook::velox::bits::nextPowerOfTwo(capacity) / 4));
  }

  bool isSet() const {
    return bits_.size() > 0;
  }

  void insert(uint64_t value) {
    set(bits_.data(), bits_.size(), value);
  }

  bool mayContain(uint64_t value) const {
    return test(bits_.data(), bits_.size(), value);
  }

  void merge(const char* serialized) {
    facebook::velox::common::InputByteStream stream(serialized);
    auto version = stream.read<int8_t>();
    VELOX_USER_CHECK_EQ(kBloomFilterV1, version);
    auto size = stream.read<int32_t>();
    bits_.resize(size);
    auto* src =
        reinterpret_cast<const uint64_t*>(serialized + stream.offset());
    if (bits_.size() == 0) {
      for (int32_t i = 0; i < size; i++) {
        bits_[i] = src[i];
      }
      return;
    } else if (size == 0) {
      return;
    }
    VELOX_DCHECK_EQ(bits_.size(), static_cast<size_t>(size));
    for (int32_t i = 0; i < size; i++) {
      bits_[i] |= src[i];
    }
  }

  uint32_t serializedSize() const {
    return 1 + 4 + bits_.size() * 8;
  }

  void serialize(char* output) const {
    facebook::velox::common::OutputByteStream stream(output);
    stream.appendOne(kBloomFilterV1);
    stream.appendOne(static_cast<int32_t>(bits_.size()));
    for (auto bit : bits_) {
      stream.appendOne(bit);
    }
  }

 private:
  inline static uint64_t bloomMask(uint64_t hashCode) {
    return (1ULL << (hashCode & 63)) | (1ULL << ((hashCode >> 6) & 63)) |
        (1ULL << ((hashCode >> 12) & 63)) | (1ULL << ((hashCode >> 18) & 63));
  }

  inline static size_t bloomIndex(size_t bloomSize, uint64_t hashCode) {
    return ((hashCode >> 24) & (bloomSize - 1));
  }

  inline static void
  set(uint64_t* bloom, size_t bloomSize, uint64_t hashCode) {
    bloom[bloomIndex(bloomSize, hashCode)] |= bloomMask(hashCode);
  }

  inline static bool
  test(const uint64_t* bloom, size_t bloomSize, uint64_t hashCode) {
    auto mask = bloomMask(hashCode);
    return mask == (bloom[bloomIndex(bloomSize, hashCode)] & mask);
  }

  static constexpr int8_t kBloomFilterV1 = 1;
  std::vector<uint64_t> bits_;
};

} // namespace shard
} // namespace gluten
