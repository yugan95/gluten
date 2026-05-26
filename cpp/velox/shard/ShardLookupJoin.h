#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "BloomFilter64.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"
#include "velox/vector/ComplexVector.h"

#include "BloomHashUtil.h"
#include "CompactRowWireFormat.h"
#include "ShardLookupJoinNode.h"
#include "SparkMurmurHash.h"
#include "VeloxShardManager.h"
#include "proto/shard_lookup.grpc.pb.h"

namespace gluten {
namespace shard {

using namespace facebook::velox;

/// ShardLookupJoin – a DistributedMapJoin Operator that buffers probe rows
/// per-shard and issues batched async RPCs, modeled after Spark's
/// LookupJoinIterator in DistributedMapJoinExec.
///
/// Key differences from IndexLookupJoin:
///   - Buffers probe rows per-shard across multiple addInput() calls
///   - Flushes a shard when its buffer reaches maxBatchSize (per-shard flush)
///   - Backpressure via needsInput() + maxInflightRpcs (no Velox DriverBlockingState)
///   - getOutput() synchronously waits on cv when RPCs are in-flight
///   - Outputs results in shard-completion order (not input order)
///   - Handles Left Join null-fill per-shard immediately on result arrival
class ShardLookupJoin : public exec::Operator {
 public:
  ShardLookupJoin(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const ShardLookupJoinNode>& joinNode);

  void initialize() override;

  bool needsInput() const override;

  void addInput(RowVectorPtr input) override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  RowVectorPtr getOutput() override;

  bool isFinished() override;

  void close() override;

 private:
  /// Whether input batch backpressure is active (too many retained batches).
  bool isInputBackpressureActive() const {
    return static_cast<int32_t>(inputBatches_.size()) > maxRetainedBatches_;
  }

  // ---------- Shard routing ----------

  /// Record a fatal error that should propagate to the driver thread.
  void setFatalError(const std::string& message);

  /// Helper called when retry is exhausted: sets fatal error, decrements
  /// numInFlight_, and notifies completedCv_.  Reduces duplication in
  /// retryRemoteLookup's multiple failure paths.
  void failRetry(const std::string& message);

  // ---------- Per-shard buffering ----------

  /// A buffered probe row: holds a reference to the input batch and the
  /// row index within it.  The input batch is kept alive via shared_ptr
  /// in inputBatches_.
  struct BufferedRow {
    /// Index into inputBatches_ vector.
    uint32_t batchIndex;
    /// Row index within that batch.
    vector_size_t rowIndex;
  };

  /// Per-shard buffer of probe rows.
  struct ShardBuffer {
    std::vector<BufferedRow> rows;
  };

  /// Add a probe row to the per-shard buffer; flush if threshold reached.
  void bufferRow(int32_t shardId, uint32_t batchIndex, vector_size_t row);

  /// Flush a shard's buffer: extract probe keys, issue async RPC.
  void flushShard(int32_t shardId);

  /// Flush all non-empty shard buffers (called on noMoreInput).
  void flushAllShards();

  // ---------- RPC ----------

  /// Forward declaration; defined in .cc.
  struct RpcContext;

  /// Re-issue a failed RPC on a (possibly different) replica.
  /// Called from the drain thread; `ctx` ownership is transferred in.
  void retryRemoteLookup(std::unique_ptr<RpcContext> ctx);

  /// A completed lookup result ready for join output.
  struct CompletedLookup {
    int32_t shardId;
    /// Probe rows that were sent in this RPC (references into inputBatches_).
    std::vector<BufferedRow> probeRows;
    /// Build-side output (deserialized from RPC response).
    RowVectorPtr buildOutput;
    /// For each build output row, the index into probeRows that matched.
    std::vector<int32_t> inputHits;
    /// Total output rows for this lookup (inner = buildOutput->size(),
    /// left = buildOutput->size() + unmatched probe rows).
    vector_size_t totalOutputRows{0};
    /// (Left Join only) Indices into probeRows that did NOT match any
    /// build row.  Pre-computed once in computeTotalOutputRows so that
    /// produceOutputForLeftJoin's phase 2 (null-fill) can iterate by
    /// index without re-building a hash set on every output slice.
    std::vector<int32_t> unmatchedProbeRows;
  };

  /// One entry in inputBatches_ — the batch itself plus a reference count
  /// of how many probe rows still reference it (across per-shard buffers,
  /// in-flight CompletedLookups, and BF-negative pending rows).  When
  /// refCount reaches 0 the entry is erased and the RowVectorPtr released,
  /// allowing upstream operators to reclaim the memory.
  struct BatchEntry {
    RowVectorPtr batch;
    int32_t refCount{0};
  };

  /// Dedup-expansion map: dedupMap[uniqueKeyIdx] = list of original
  /// probeRows indices that share that key.  When empty (or single-element
  /// per entry), no expansion is needed.
  using DedupMap = std::vector<std::vector<int32_t>>;

  /// Issue an async gRPC lookup for a shard.
  /// `dedupMap` maps each unique-key index to the original probeRows indices.
  void issueRemoteLookup(
      int32_t shardId,
      RowVectorPtr probeKeys,
      std::vector<BufferedRow> probeRows,
      DedupMap dedupMap,
      RowVectorPtr filterProbeColumns = nullptr);

  /// Issue a local lookup for a shard.
  void issueLocalLookup(
      int32_t shardId,
      RowVectorPtr probeKeys,
      std::vector<BufferedRow> probeRows,
      DedupMap dedupMap,
      RowVectorPtr filterProbeColumns = nullptr);

  /// Called when a gRPC async RPC finishes (from drain thread).
  void handleRpcCompletion(RpcContext* ctx, bool ok);

  /// Enqueue a completed lookup result and wake the Driver.
  void onRpcComplete(std::unique_ptr<CompletedLookup> result);

  /// Build a probe-side gathered column for output, using DictionaryVector
  /// wrapping wherever a contiguous run of output rows comes from the same
  /// input batch.  Falls back to per-row copy when needed.  Returns the
  /// gathered column ready to be assigned to output_->childAt().
  ///
  /// `inputChannel` is the channel within the source input batch.
  /// `outputType` is the column's output type (== input type for probe cols).
  /// `probeRowPerOut[i]` points to the BufferedRow to use for output row i.
  VectorPtr gatherProbeColumnAsDictionary(
      column_index_t inputChannel,
      const TypePtr& outputType,
      const std::vector<const BufferedRow*>& probeRowPerOut,
      vector_size_t numOutputRows);

  /// Build a probe-key column (no projection — just one of the leftKeys)
  /// across a per-shard rows buffer, using DictionaryVector wrapping for
  /// contiguous same-batch runs.  Used by flushShard.
  VectorPtr gatherProbeKeyColumnAsDictionary(
      column_index_t inputChannel,
      const TypePtr& keyType,
      const std::vector<BufferedRow>& rows);

  /// Compute totalOutputRows for a CompletedLookup (inner vs left).
  void computeTotalOutputRows(CompletedLookup& lookup) const;

  // ---------- Output production ----------

  /// Produce output rows from a completed lookup result.
  RowVectorPtr produceOutputForInnerJoin(const CompletedLookup& lookup);
  RowVectorPtr produceOutputForLeftJoin(CompletedLookup& lookup);


  // ---------- Output projection ----------

  void initOutputProjections();

  // ---------- Constants ----------

  const vector_size_t outputBatchSize_;
  const core::JoinType joinType_;
  const size_t numKeys_;
  const RowTypePtr probeType_;
  const RowTypePtr buildOutputType_;
  const int64_t shardSetId_;
  const int32_t numShards_;
  const int32_t maxInflightRpcs_;
  const int32_t maxBatchSize_;
  const RowTypePtr hashKeyType_;
  /// Post-join filter expression from the plan node (nullptr for pure equi-join).
  const core::TypedExprPtr filterExpr_;
  std::unordered_map<int32_t, std::vector<ShardServerLocation>>
      shardLocationMap_;

  // ---------- Derived constants (computed once in constructor) ----------

  /// Max retained input batches before backpressure kicks in.
  const int32_t maxRetainedBatches_;

  /// Column indices of leftKeys within probeType.
  std::vector<column_index_t> leftKeyChannels_;

  /// Pre-computed per-column castToBigint flags for shard hash computation.
  std::vector<bool> castToBigintFlags_;

  /// Cached probe key RowType (key columns only), built once.
  RowTypePtr probeKeyType_;

  // ---------- Post-join filter ----------

  /// Compiled filter expression set (nullptr when no filter).
  std::unique_ptr<exec::ExprSet> filterExprSet_;

  /// Input type for filter evaluation: concat(probeType, buildOutputType).
  RowTypePtr filterInputType_;

  // ---------- Server-side filter pushdown ----------

  /// Column indices within probeType_ that filterExpr_ references.
  std::vector<column_index_t> filterProbeChannels_;

  /// RowType of filter-referenced probe columns (subset of probeType_).
  RowTypePtr filterProbeType_;

  /// Serialized filter expression JSON (ISerializable, computed once).
  std::string filterExprJson_;

  /// Hive-format type strings (computed once in initialize).
  std::string filterProbeTypeStr_;
  std::string filterInputTypeStr_;

  /// Server-side filter eval input type: concat(filterProbeType, buildOutputType).
  /// Used by issueLocalLookup to pass to VeloxShardManager::lookup.
  RowTypePtr serverFilterInputType_;

  /// Cached dedup RowType when filter is active: concat(probeKeyType, filterProbeType).
  /// Built once in initialize() to avoid per-flushShard reconstruction.
  RowTypePtr dedupWithFilterType_;

  /// Decrement refCount for every BufferedRow in `rows` and erase any
  /// inputBatches_ entry whose refCount reaches zero.  Called from the
  /// Driver thread (getOutput / close) after consuming a CompletedLookup
  /// or the BF-negative buffer, so no locking is needed.
  void releaseProbeRefs(const std::vector<BufferedRow>& rows);

  // ---------- State ----------

  /// All input batches received via addInput(), kept alive only as long
  /// as some buffered probe row, in-flight CompletedLookup, or BF-negative
  /// row still references them.  Keyed by a monotonically increasing
  /// batchId (BufferedRow.batchIndex) — never reused so stale references
  /// can't accidentally hit a recycled slot.
  std::unordered_map<uint32_t, BatchEntry> inputBatches_;

  /// Monotonically increasing batch id; used as the key into inputBatches_
  /// and as BufferedRow.batchIndex.  Never reused.
  uint32_t nextBatchId_{0};

  /// Running total of refCount across all entries in inputBatches_.
  /// Maintained incrementally to avoid O(n) traversal per addInput call.
  int64_t totalRetainedRefCount_{0};

  /// Debug mode enabled by env GLUTEN_SHARD_DMJ_DEBUG=1.
  bool debugEnabled_{false};

  /// Lightweight counters for debug logging (no perf cost when debug off).
  uint64_t getOutputCalls_{0};
  uint64_t rpcIssuedCount_{0};
  uint64_t rpcCompletedCount_{0};

  /// Row-count stats (incremented unconditionally; only logged when debug on).
  uint64_t totalInputRows_{0};
  uint64_t bfNegativeRows_{0};
  uint64_t bfPositiveRows_{0};
  uint64_t totalBuildOutputRows_{0};
  uint64_t totalProducedRows_{0};
  uint64_t flushShardCount_{0};
  uint64_t localLookupCount_{0};

  /// Cumulative timing (nanoseconds, steady_clock). Only measured when
  /// debugEnabled_ is true to avoid rdtsc overhead on the hot path.
  uint64_t addInputNanos_{0};
  uint64_t flushShardNanos_{0};
  uint64_t getOutputWaitNanos_{0};
  uint64_t localLookupNanos_{0};

  /// Per-shard row buffers.
  std::unordered_map<int32_t, ShardBuffer> shardBuffers_;

  /// Per-shard round-robin index into shardLocationMap_[shardId].
  /// Each flushShard call advances the index so RPCs for the same shard
  /// rotate across replicas, spreading load (especially for skewed partitions).
  std::unordered_map<int32_t, size_t> shardReplicaRRIndex_;

  /// Set of shard IDs that are local (this executor holds the shard table).
  /// For these shards, issueLocalLookup is used instead of remote RPC.
  std::unordered_set<int32_t> localShards_;

  /// Mutex protecting shardLocationMap_, shardReplicaRRIndex_, and stubCache_
  /// against concurrent access from the driver thread and the drain thread.
  mutable std::mutex routeMutex_;

  /// Cached gRPC stubs per remote address.  Creating a Stub is cheap but
  /// not free (~50µs for method descriptor resolution + allocation).  With
  /// thousands of RPCs per second, caching avoids measurable overhead.
  std::unordered_map<std::string, std::shared_ptr<proto::ShardLookupService::Stub>>
      stubCache_;

  /// Number of in-flight async RPCs.
  std::atomic<int32_t> numInFlight_{0};

  /// Fatal error message set by the drain thread when retries are exhausted.
  /// Checked by the driver thread in needsInput() / getOutput().
  std::atomic<bool> hasFatalError_{false};
  std::mutex fatalErrorMutex_;
  std::string fatalErrorMessage_;

  /// Queue of completed lookup results (thread-safe).
  mutable std::mutex completedMutex_;
  std::queue<std::unique_ptr<CompletedLookup>> completedQueue_;

  /// Condition variable notified when an RPC completes (result pushed to
  /// completedQueue_ or numInFlight_ decremented on error).  Used by
  /// getOutput() to synchronously wait for results instead of going
  /// through Velox's DriverBlockingState promise/future mechanism, which
  /// has a race condition in Task::next()'s notReadyFutures collection
  /// that can cause permanent hangs when the RPC completes very quickly.
  std::condition_variable completedCv_;

  /// Current completed lookup being output (may span multiple getOutput calls).
  std::unique_ptr<CompletedLookup> currentLookup_;
  vector_size_t currentLookupOutputRow_{0};

  /// Whether tailing shard buffers have been flushed after noMoreInput.
  bool flushedTailing_{false};

  /// For Left Join: pre-materialized null-filled output batches for
  /// BF-negative rows.  Each batch is produced eagerly in addInput() so
  /// that BF-negative rows never pin inputBatches_ via refCount.
  std::queue<RowVectorPtr> pendingBfNegativeOutputs_;

  /// Output projections (probe → output, build → output).
  std::vector<exec::IdentityProjection> probeOutputProjections_;
  std::vector<exec::IdentityProjection> buildOutputProjections_;

  /// Reusable output vector.
  RowVectorPtr output_;

  /// Bloom filter for probe-side pre-filtering (shared across all SLJ
  /// instances on this executor, owned by VeloxShardManager).
  std::shared_ptr<const BloomFilter64> bloomFilter_;

  /// Reusable per-batch hash buffers for addInput's columnar batch hashing.
  /// `bfHashes_` stores one int32 per row of the current input (BF hash,
  /// no cast).  `shardHashes_` stores one int32 per row (hash with castToBigint
  /// applied per key).  Both are sized to the current input batch's row count
  /// in addInput() and reused across batches to avoid reallocation.
  std::vector<int32_t> bfHashes_;
  std::vector<int32_t> shardHashes_;

  /// gRPC completion queue for async RPCs.
  std::unique_ptr<grpc::CompletionQueue> completionQueue_;

  /// Background thread that drains the gRPC completion queue.
  std::thread drainThread_;

  /// gRPC stub kept alive for the lifetime of async RPCs.
  /// Each issueRemoteLookup stores its stub in the RpcContext to prevent
  /// premature destruction.
};

/// PlanNodeTranslator for ShardLookupJoinNode → ShardLookupJoin Operator.
class ShardLookupJoinTranslator
    : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;
};

} // namespace shard
} // namespace gluten
