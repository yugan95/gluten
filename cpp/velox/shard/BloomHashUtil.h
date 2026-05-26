#pragma once

#include <cstdint>

namespace gluten {
namespace shard {

/// Spread a 32-bit Spark MurmurHash into a 64-bit value suitable for
/// Velox BloomFilter. The BloomFilter uses bits [0:23] for intra-word bit
/// selection and bits [24:24+N] for word index selection.  A raw int32 only
/// provides 8 bits for word index (bits 24-31), limiting addressable words
/// to 256 regardless of BF size.  This mixing step produces a well-distributed
/// 64-bit value so the full word space is utilized.
///
/// Uses MurmurHash3 fmix64-style avalanche mixing.
inline uint64_t spreadHashForBloom(int32_t hash) {
  uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(hash));
  h = (h ^ (h >> 16)) * 0xff51afd7ed558ccdULL;
  h = (h ^ (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
  h = h ^ (h >> 33);
  return h;
}

} // namespace shard
} // namespace gluten
