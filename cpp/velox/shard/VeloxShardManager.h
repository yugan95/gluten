#pragma once

#include <jni.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <folly/Synchronized.h>
#include <folly/SharedMutex.h>

#include "BloomFilter64.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/HashTable.h"
#include "velox/expression/Expr.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

#include <grpcpp/grpcpp.h>

#include "BlockManagerBridge.h"

namespace gluten {
namespace shard {

struct ShardKey {
  int64_t setId;
  int32_t shardId;

  bool operator==(const ShardKey& other) const {
    return setId == other.setId && shardId == other.shardId;
  }
};

struct ShardKeyHash {
  size_t operator()(const ShardKey& key) const {
    return std::hash<int64_t>()(key.setId) ^
        (std::hash<int32_t>()(key.shardId) << 32);
  }
};

// ---------------------------------------------------------------------------
// VeloxShardManager – executor-level singleton that owns the in-process hash
// tables built from shard data.  Provides lookup for both local probe-side
// callers (ShardLookupJoin) and remote gRPC requests (via VeloxShardRpcServer).
//
// Analogous to vanilla Spark's ShardManager (data management part only).
// The gRPC serving is handled by VeloxShardRpcServer (see VeloxShardRpcServer.h).
// ---------------------------------------------------------------------------
class VeloxShardManager {
 public:
  static VeloxShardManager* getInstance() {
    static VeloxShardManager instance;
    return &instance;
  }

  // Initialize the root memory pool and set the BlockManagerBridge.
  // Called once during executor startup (after BlockManager is ready).
  // javaVm may be null in unit tests (disables JNI-based refreshShardLocations).
  void initialize(
      std::shared_ptr<BlockManagerBridge> bridge,
      JavaVM* javaVm = nullptr);

  // Tear down all HashTables and release memory pools.
  void shutdown();

  void constructShardTable(int64_t setId, int32_t shardId);

  void destroyShardTable(int64_t setId);

  // Fetch a BloomFilter from BlockManager.
  // shardId=-1 for set-level (merged) BF, shardId>=0 for shard-level BF.
  // Returns serialized BF bytes, or empty string if not found.
  std::string fetchBloomFilter(int64_t setId, int32_t shardId);

  // Returns a shared, executor-level BloomFilter for the given shard set.
  // The BF is loaded from BlockManager on first access and cached for
  // subsequent callers.  All ShardLookupJoin instances share the same
  // read-only BF, avoiding per-task memory duplication.
  // Returns nullptr if no BF is available for this set.
  std::shared_ptr<const BloomFilter64>
  getOrLoadBloomFilter(int64_t setId);

  struct LookupResult {
    // outputPool must be declared before output so that it is destroyed
    // after output (C++ destroys members in reverse declaration order).
    // The output RowVector references memory allocated from outputPool.
    std::shared_ptr<facebook::velox::memory::MemoryPool> outputPool;
    facebook::velox::RowVectorPtr output;
    std::vector<int32_t> inputHits;
  };

  struct SerializedLookupResult {
    // CompactRow wire-format bytes ready for gRPC response.
    // Empty string when there are no matches.
    std::string outputBytes;
    std::vector<int32_t> inputHits;
  };

  // Lookup probe keys and serialize matched rows in batches to bound peak
  // memory.  Each batch of up to kSerializeBatchSize rows is extracted from
  // the RowContainer, serialized to CompactRow wire format, then immediately
  // released.  The per-batch payloads are merged into a single wire payload
  // at the end.  Peak memory is O(kSerializeBatchSize × numCols) instead of
  // O(totalMatches × numCols) as with lookup() + serializeCompactRowBatch().
  //
  // Intended for the gRPC server path where the RowVector output is never
  // used directly — only the serialized bytes matter.
  SerializedLookupResult lookupAndSerialize(
      int64_t setId,
      int32_t shardId,
      const facebook::velox::RowVectorPtr& probeKeys,
      const std::vector<int32_t>& inputIndices,
      facebook::velox::memory::MemoryPool* rpcPool,
      const facebook::velox::RowVectorPtr& probeFilterColumns = nullptr,
      const std::string& filterExprJson = "",
      const facebook::velox::RowTypePtr& probeColumnsType = nullptr,
      const facebook::velox::RowTypePtr& filterInputType = nullptr);

  // Lookup probe keys in the hash table for the given shard.
  // If callerPool is non-null, output vectors are allocated from it directly
  // (avoiding a cross-pool copy when the caller merges results from multiple
  // shards).  If callerPool is null, a temporary leaf pool is created
  // internally and returned via LookupResult::outputPool.
  LookupResult lookup(
      int64_t setId,
      int32_t shardId,
      const facebook::velox::RowVectorPtr& probeKeys,
      const std::vector<int32_t>& inputIndices,
      facebook::velox::memory::MemoryPool* callerPool = nullptr,
      const facebook::velox::RowVectorPtr& probeFilterColumns = nullptr,
      const std::string& filterExprJson = "",
      const facebook::velox::RowTypePtr& probeColumnsType = nullptr,
      const facebook::velox::RowTypePtr& filterInputType = nullptr);

  // Returns the key RowType for the given shard, derived from the HashTable's
  // VectorHashers.  Used by VeloxShardRpcServer / ShardLookupJoin to
  // deserialize probe keys
  // with the correct type.  Returns nullptr if the shard is not found.
  facebook::velox::RowTypePtr getKeyType(int64_t setId, int32_t shardId);

  // Public accessor for the root memory pool (used by VeloxShardRpcServer
  // for serialization/deserialization of gRPC payloads).
  facebook::velox::memory::MemoryPool* pool() const { return pool_.get(); }

  // Returns an existing gRPC channel to `address` (host:port) or creates one.
  // Channels are cached at executor level (singleton lifetime) so all tasks
  // on the same executor share connections, avoiding per-task channel churn.
  // Thread-safe: guarded by channelsMutex_.
  std::shared_ptr<grpc::Channel> getOrCreateChannel(
      const std::string& address) {
    std::lock_guard<std::mutex> lock(channelsMutex_);
    auto it = grpcChannels_.find(address);
    if (it != grpcChannels_.end()) {
      return it->second;
    }
    grpc::ChannelArguments args;
    // Default 4MB is too small for large lookup responses (e.g. 7MB+).
    static constexpr int kMaxMessageSize = 512 * 1024 * 1024; // 512MB
    args.SetMaxReceiveMessageSize(kMaxMessageSize);
    args.SetMaxSendMessageSize(kMaxMessageSize);
    auto channel = grpc::CreateCustomChannel(
        address, grpc::InsecureChannelCredentials(), args);
    grpcChannels_[address] = channel;
    return channel;
  }

 private:
  VeloxShardManager() = default;
  ~VeloxShardManager() = default;
  VeloxShardManager(const VeloxShardManager&) = delete;
  VeloxShardManager& operator=(const VeloxShardManager&) = delete;

  /// Debug mode enabled by env GLUTEN_SHARD_DMJ_DEBUG=1 (checked in initialize).
  bool debugEnabled_{false};

  // --- Server-side join condition eval (filter pushdown) ---

  struct CachedFilter {
    // pool must be declared before queryCtx/execCtx/exprSet so it is
    // destroyed after them (C++ destroys members in reverse order).
    // execCtx holds pool.get() as a raw pointer.
    std::shared_ptr<facebook::velox::memory::MemoryPool> pool;
    std::shared_ptr<facebook::velox::core::QueryCtx> queryCtx;
    std::unique_ptr<facebook::velox::core::ExecCtx> execCtx;
    std::unique_ptr<facebook::velox::exec::ExprSet> exprSet;
    facebook::velox::RowTypePtr inputType;
    std::mutex evalMutex;
  };

  std::shared_ptr<CachedFilter> getOrCompileFilter(
      int64_t setId,
      const std::string& filterExprJson,
      const facebook::velox::RowTypePtr& filterInputType);

  void evaluateFilter(
      CachedFilter& filter,
      const facebook::velox::RowVectorPtr& probeFilterColumns,
      std::vector<int32_t>& inputHits,
      std::vector<char*>& matchedRows,
      facebook::velox::exec::RowContainer* rowContainer,
      facebook::velox::memory::MemoryPool* pool);

  std::mutex filterCacheMutex_;
  std::unordered_map<int64_t, std::shared_ptr<CachedFilter>> filterCache_;

  // --- End filter pushdown ---

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;

  folly::Synchronized<
      std::unordered_map<
          ShardKey,
          std::shared_ptr<facebook::velox::exec::HashTable<false>>,
          ShardKeyHash>,
      folly::SharedMutexWritePriority>
      shardTables_;

  // Per-shard leaf memory pools.  Each shard's HashTable allocates from its
  // own leaf pool.  The leaf pool must outlive the HashTable, so it is stored
  // here and erased together with the HashTable in destroyShardTable.
  folly::Synchronized<
      std::unordered_map<
          ShardKey,
          std::shared_ptr<facebook::velox::memory::MemoryPool>,
          ShardKeyHash>,
      folly::SharedMutexWritePriority>
      shardPools_;

  // Per-setId memory pool.  All shards within the same shard set share one
  // pool, which provides query-level memory accounting without the overhead
  // of per-shard pool management.  The pool must outlive all HashTables that
  // allocate from it, so destroyShardTable erases tables before pools.
  folly::Synchronized<
      std::unordered_map<
          int64_t,
          std::shared_ptr<facebook::velox::memory::MemoryPool>>,
      folly::SharedMutexWritePriority>
      setPools_;

  // Per-setId executor-level BloomFilter cache.  Each entry is a merged
  // set-level BF (shardId=-1), loaded once from BlockManager and shared
  // read-only by all ShardLookupJoin instances on this executor.
  folly::Synchronized<
      std::unordered_map<
          int64_t,
          std::shared_ptr<const BloomFilter64>>,
      folly::SharedMutexWritePriority>
      bloomFilters_;

  // Persistent BlockManagerBridge for reading shard data and BF.
  // Set once during initialize(), immutable thereafter.
  std::shared_ptr<BlockManagerBridge> blockManagerBridge_;

  // Cached JavaVM* for JNI callbacks (refreshShardLocations).
  // Set during initialize(), null in unit tests.
  JavaVM* javaVm_{nullptr};

  // Cached JNI handles for refreshShardLocations() – populated once during
  // initialize() so that the hot-path avoids repeated FindClass /
  // GetMethodID / GetStaticFieldID lookups.
  jclass jniGlobalClass_{nullptr};     // GlobalRef to GlutenShardManagerJni$
  jobject jniGlobalModule_{nullptr};   // GlobalRef to MODULE$ singleton
  jmethodID jniRefreshMethod_{nullptr};

  // Per-shard hash key type — the types used by build-side HashPartitioning
  // to compute shard assignment.  May differ from the data types stored in the
  // HashTable when buildBoundKeys contains cast expressions (e.g.
  // cast(k as bigint)).  Used by ShardLookupJoin on the probe side.
  folly::Synchronized<
      std::unordered_map<
          ShardKey,
          facebook::velox::RowTypePtr,
          ShardKeyHash>,
      folly::SharedMutexWritePriority>
      hashKeyTypes_;

 public:
  // Returns the hash key type for the given shard (used by ShardLookupJoin to
  // compute shard ids consistently with the build-side partitioning).
  facebook::velox::RowTypePtr getHashKeyType(int64_t setId, int32_t shardId) {
    ShardKey key{setId, shardId};
    auto types = hashKeyTypes_.rlock();
    auto it = types->find(key);
    if (it != types->end()) {
      return it->second;
    }
    return nullptr;
  }

  // Returns the hash key type for any shard in the given set.
  // Iterates through all registered hash key types for the set and returns
  // the first non-null one.  This is more robust than querying a specific
  // shard id (e.g. shard 0), because some shards may be empty and not have
  // a hash key type registered.
  facebook::velox::RowTypePtr getHashKeyTypeForSet(int64_t setId) {
    auto types = hashKeyTypes_.rlock();
    for (const auto& [key, type] : *types) {
      if (key.setId == setId && type != nullptr) {
        return type;
      }
    }
    return nullptr;
  }

  // Check whether a shard table (hash table) is available locally.
  // Used by ShardLookupJoin to decide whether to do local lookup or remote
  // gRPC.
  bool hasShardTable(int64_t setId, int32_t shardId) {
    ShardKey key{setId, shardId};
    auto tables = shardTables_.rlock();
    return tables->find(key) != tables->end();
  }

  // Refresh shard locations from ShardManagerMaster via JNI.
  // Invalidates the Spark-side location cache and fetches the latest
  // locations, including newly installed replicas after executor crash.
  // Mirrors Spark DMJ's `locations(refresh = true)`.
  // Returns a list of "host:port" strings, or empty if unavailable.
  // In unit tests (javaVm_ == nullptr), returns empty.
  std::vector<std::string> refreshShardLocations(
      int64_t setId, int32_t shardId);

  // Mark an address as dead for deadAddressTtl_ (default 10 min).
  // Thread-safe; called from SLJ drain threads on RPC failure.
  // All SLJ instances on this executor share the blacklist.
  void markAddressDead(const std::string& address);

  // Check if an address is currently blacklisted (not expired).
  // Thread-safe; called from SLJ driver threads during replica selection.
  bool isAddressDead(const std::string& address);

 private:
  // Guards initialize() so it runs at most once until shutdown() resets it.
  std::atomic<bool> initialized_{false};

  // Executor-level dead address blacklist with TTL.
  // Shared across all SLJ instances.  When an RPC fails, the target
  // address is added here so other tasks skip it immediately.
  // Entries expire after deadAddressTtl_ (lazy deletion on query).
  std::mutex deadAddressMutex_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      deadAddresses_;
  std::chrono::seconds deadAddressTtl_{600};

  // Executor-level gRPC channel cache keyed by "host:port".
  // Shared across all tasks on this executor.  Guarded by channelsMutex_.
  std::unordered_map<std::string, std::shared_ptr<grpc::Channel>> grpcChannels_;
  std::mutex channelsMutex_;
};

} // namespace shard
} // namespace gluten
