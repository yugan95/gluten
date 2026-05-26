#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/row/CompactRow.h"
#include "velox/vector/ComplexVector.h"

namespace gluten {
namespace shard {

// ---------------------------------------------------------------------------
// CompactRow wire format used for ShardLookup gRPC payloads.
//
// This format wraps Velox CompactRow byte buffers with a small header so that
// (a) endpoints can detect protocol mismatches early and (b) the decoder can
// split the payload back into per-row byte ranges (CompactRow::deserialize
// requires a vector<string_view> of complete row bytes).
//
// Layout (all integer fields are little-endian, matching x86_64 / aarch64-LE):
//
//   [magic:    uint32  = 0x434F4D31 ("COM1")]   ← 4 bytes
//   [version:  uint32  = kCompactRowWireVersion] ← 4 bytes
//   [numRows:  uint32]                            ← 4 bytes
//   For each row in [0, numRows):
//     [rowLength: uint32]                         ← 4 bytes
//     [rowBytes:  rowLength bytes]
//
// The schema (RowType) is NOT carried on the wire.  Callers must reconstruct
// it from out-of-band metadata (e.g., VeloxShardManager::getKeyType for
// probe keys, or the build-side output type known by ShardLookupJoin for
// lookup outputs).  This keeps the per-RPC payload tight at the cost of
// requiring the build/probe sides to share the same key/output schema
// definition.
//
// Version compatibility: strict.  Decoder rejects any payload whose magic
// or version does not match kCompactRowWireMagic / kCompactRowWireVersion
// with VELOX_USER_FAIL.  Bumping the version is a breaking change that
// requires synchronized rollout of build and probe binaries.
//
// Type fidelity caveat — TIMESTAMP:
//   CompactRow encodes Timestamp as int64 microseconds (matching Spark's
//   Timestamp type).  Sub-microsecond nanoseconds in the source vector are
//   silently truncated by the encoder.  This is a Velox CompactRow contract,
//   not a quirk of this wrapper — see velox/row/CompactRow.cpp::writeTimestamp.
//   Callers needing nanosecond fidelity must use a different wire format.
// ---------------------------------------------------------------------------

// "COM1" interpreted as little-endian: bytes 0x31 0x4D 0x4F 0x43.
// We use a fixed integer constant to avoid any endianness ambiguity in the
// magic check.
constexpr uint32_t kCompactRowWireMagic = 0x434F4D31u;

// Bump this when the wire layout changes in a way that breaks decoders.
constexpr uint32_t kCompactRowWireVersion = 1u;

// Header size: magic + version + numRows.
constexpr size_t kCompactRowWireHeaderSize = 3 * sizeof(uint32_t);

// Per-row prefix size: rowLength.
constexpr size_t kCompactRowWireRowPrefixSize = sizeof(uint32_t);

// Compile-time guard: the on-wire integer encoding above relies on host
// little-endian byte order so that memcpy of uint32_t round-trips through
// the wire correctly.  Any big-endian platform would silently corrupt the
// magic / version checks and row-length prefixes.
static_assert(
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
    "CompactRow wire format requires little-endian byte order");

namespace detail {

inline void writeUint32LE(char* dst, uint32_t value) {
  std::memcpy(dst, &value, sizeof(uint32_t));
}

inline uint32_t readUint32LE(const char* src) {
  uint32_t value;
  std::memcpy(&value, src, sizeof(uint32_t));
  return value;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Serialize a RowVector into the CompactRow wire format described above.
//
// Returns a freshly-allocated std::string owning the wire bytes.  The caller
// typically moves it into a Protobuf bytes field.
//
// Empty (numRows == 0) inputs produce a header-only payload (12 bytes).
// nullptr inputs are treated as empty.
// ---------------------------------------------------------------------------
inline std::string serializeCompactRowBatch(
    const facebook::velox::RowVectorPtr& vector,
    facebook::velox::memory::MemoryPool* /*pool*/) {
  const auto numRows = (vector == nullptr)
      ? 0
      : static_cast<uint32_t>(vector->size());

  // Empty fast path: header-only payload.
  if (numRows == 0) {
    std::string out(kCompactRowWireHeaderSize, '\0');
    detail::writeUint32LE(out.data(), kCompactRowWireMagic);
    detail::writeUint32LE(out.data() + 4, kCompactRowWireVersion);
    detail::writeUint32LE(out.data() + 8, 0u);
    return out;
  }

  facebook::velox::row::CompactRow compactRow(vector);
  const auto rowType = facebook::velox::asRowType(vector->type());

  // Compute total payload size and per-row sizes.
  // Use fixedRowSize fast path when all columns are fixed-width.
  std::vector<uint32_t> rowSizes(numRows);
  size_t bodyBytes = 0;
  if (auto fixed = facebook::velox::row::CompactRow::fixedRowSize(rowType)) {
    const auto fixedSize = static_cast<uint32_t>(fixed.value());
    for (uint32_t i = 0; i < numRows; ++i) {
      rowSizes[i] = fixedSize;
    }
    bodyBytes = static_cast<size_t>(numRows) *
        (kCompactRowWireRowPrefixSize + fixedSize);
  } else {
    for (uint32_t i = 0; i < numRows; ++i) {
      const auto sz = compactRow.rowSize(static_cast<int32_t>(i));
      VELOX_CHECK_GE(
          sz, 0, "CompactRow::rowSize returned negative size for row {}", i);
      rowSizes[i] = static_cast<uint32_t>(sz);
      bodyBytes += kCompactRowWireRowPrefixSize + rowSizes[i];
    }
  }

  // Allocate the output string up-front and write directly into it.
  // Initialize to zero so CompactRow::serialize can rely on the null-bit
  // bytes being zero (CompactRow contract: "buffer must be set to all zeros
  // for null-bits handling").
  std::string out(kCompactRowWireHeaderSize + bodyBytes, '\0');
  char* base = out.data();

  // Header.
  detail::writeUint32LE(base, kCompactRowWireMagic);
  detail::writeUint32LE(base + 4, kCompactRowWireVersion);
  detail::writeUint32LE(base + 8, numRows);

  // Body: write per-row length prefix then serialize the row in place.
  size_t cursor = kCompactRowWireHeaderSize;
  for (uint32_t i = 0; i < numRows; ++i) {
    const auto rowLen = rowSizes[i];
    detail::writeUint32LE(base + cursor, rowLen);
    cursor += kCompactRowWireRowPrefixSize;

    // CompactRow::serialize writes the row body and returns bytes written.
    // It must equal the pre-computed rowLen; otherwise the body is corrupt.
    const auto written = compactRow.serialize(
        static_cast<int32_t>(i), base + cursor);
    VELOX_CHECK_EQ(
        static_cast<uint32_t>(written),
        rowLen,
        "CompactRow::serialize wrote {} bytes but rowSize predicted {} for row {}",
        written,
        rowLen,
        i);
    cursor += rowLen;
  }
  VELOX_CHECK_EQ(
      cursor,
      out.size(),
      "CompactRow wire payload size mismatch: cursor={}, expected={}",
      cursor,
      out.size());
  return out;
}

// ---------------------------------------------------------------------------
// Assemble a CompactRow wire payload from per-row byte slices that were
// already produced by an external serializer (typically the dedup pass on
// the probe side, where each candidate row is serialized once for both
// hash-collision tie-break and final wire encoding).
//
// This avoids the second CompactRow::serialize pass that
// serializeCompactRowBatch would perform: callers that have already paid
// the per-row serialize cost can hand the bytes straight in.
//
// Each entry in `rowBytes` becomes one row in the wire payload, in input
// order.  An empty `rowBytes` produces a header-only payload (12 bytes).
// The row contents are NOT validated — callers are responsible for ensuring
// each slice is a complete CompactRow body for a consistent RowType (the
// decoder will fail loudly if not).
// ---------------------------------------------------------------------------
inline std::string assembleCompactRowWireFromBytes(
    const std::vector<std::string_view>& rowBytes) {
  const auto numRows = static_cast<uint32_t>(rowBytes.size());

  // Compute total size: header + sum(perRowPrefix + rowLen).
  size_t bodyBytes = 0;
  for (const auto& row : rowBytes) {
    bodyBytes += kCompactRowWireRowPrefixSize + row.size();
  }

  std::string out(kCompactRowWireHeaderSize + bodyBytes, '\0');
  char* base = out.data();

  // Header.
  detail::writeUint32LE(base, kCompactRowWireMagic);
  detail::writeUint32LE(base + 4, kCompactRowWireVersion);
  detail::writeUint32LE(base + 8, numRows);

  // Body: per-row length prefix + row bytes (memcpy from caller-owned data).
  size_t cursor = kCompactRowWireHeaderSize;
  for (uint32_t i = 0; i < numRows; ++i) {
    const auto rowLen = static_cast<uint32_t>(rowBytes[i].size());
    detail::writeUint32LE(base + cursor, rowLen);
    cursor += kCompactRowWireRowPrefixSize;
    if (rowLen > 0) {
      std::memcpy(base + cursor, rowBytes[i].data(), rowLen);
    }
    cursor += rowLen;
  }
  VELOX_CHECK_EQ(
      cursor,
      out.size(),
      "CompactRow wire payload size mismatch in assembleCompactRowWireFromBytes: "
      "cursor={}, expected={}",
      cursor,
      out.size());
  return out;
}

// ---------------------------------------------------------------------------
// Deserialize a CompactRow wire-format payload back into a RowVector.
//
// `rowType` must match the schema that was used on the encoder side.  Magic
// and version mismatches throw VELOX_USER_FAIL with a descriptive message.
// Truncated payloads are detected and rejected (no out-of-bounds reads).
//
// numRows == 0 returns an empty RowVector with the requested schema.
// ---------------------------------------------------------------------------
inline facebook::velox::RowVectorPtr deserializeCompactRowBatch(
    std::string_view wire,
    const facebook::velox::RowTypePtr& rowType,
    facebook::velox::memory::MemoryPool* pool) {
  VELOX_USER_CHECK_NOT_NULL(rowType, "CompactRow rowType must not be null");
  VELOX_USER_CHECK_NOT_NULL(pool, "CompactRow memory pool must not be null");

  VELOX_USER_CHECK_GE(
      wire.size(),
      kCompactRowWireHeaderSize,
      "CompactRow payload truncated: header requires {} bytes but got {}",
      kCompactRowWireHeaderSize,
      wire.size());

  const auto magic = detail::readUint32LE(wire.data());
  VELOX_USER_CHECK_EQ(
      magic,
      kCompactRowWireMagic,
      "CompactRow wire magic mismatch: expected 0x{:08x}, got 0x{:08x}",
      kCompactRowWireMagic,
      magic);

  const auto version = detail::readUint32LE(wire.data() + 4);
  VELOX_USER_CHECK_EQ(
      version,
      kCompactRowWireVersion,
      "CompactRow wire version mismatch: expected {}, got {}; "
      "build and probe binaries must be the same version",
      kCompactRowWireVersion,
      version);

  const auto numRows = detail::readUint32LE(wire.data() + 8);

  // Empty payload fast path.
  if (numRows == 0) {
    return std::dynamic_pointer_cast<facebook::velox::RowVector>(
        facebook::velox::BaseVector::create(rowType, 0, pool));
  }

  // Walk the body once to build the per-row string_view list, validating
  // bounds at every step so a malformed length prefix cannot cause an
  // out-of-bounds read in CompactRow::deserialize.
  std::vector<std::string_view> rowViews;
  rowViews.reserve(numRows);

  size_t cursor = kCompactRowWireHeaderSize;
  for (uint32_t i = 0; i < numRows; ++i) {
    VELOX_USER_CHECK_LE(
        cursor + kCompactRowWireRowPrefixSize,
        wire.size(),
        "CompactRow payload truncated reading length prefix for row {} "
        "(cursor={}, size={})",
        i,
        cursor,
        wire.size());
    const auto rowLen = detail::readUint32LE(wire.data() + cursor);
    cursor += kCompactRowWireRowPrefixSize;

    VELOX_USER_CHECK_LE(
        cursor + rowLen,
        wire.size(),
        "CompactRow payload truncated reading body for row {} "
        "(cursor={}, rowLen={}, size={})",
        i,
        cursor,
        rowLen,
        wire.size());

    rowViews.emplace_back(wire.data() + cursor, rowLen);
    cursor += rowLen;
  }
  VELOX_USER_CHECK_EQ(
      cursor,
      wire.size(),
      "CompactRow payload has {} trailing bytes after row {}",
      wire.size() - cursor,
      numRows - 1);

  return facebook::velox::row::CompactRow::deserialize(rowViews, rowType, pool);
}

} // namespace shard
} // namespace gluten
