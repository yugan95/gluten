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
#include "ShardLookupJoin.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <random>
#include <unordered_set>

#include <glog/logging.h>

#include <folly/json.h>

#include "velox/buffer/Buffer.h"
#include "velox/core/Expressions.h"
#include "velox/exec/Driver.h"
#include "velox/expression/Expr.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"

namespace gluten {
namespace shard {

using namespace facebook::velox;
using namespace facebook::velox::exec;

static void collectFieldAccesses(
    const core::TypedExprPtr& expr,
    std::unordered_set<std::string>& fields) {
  if (auto fa = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    fields.insert(fa->name());
  }
  for (const auto& input : expr->inputs()) {
    collectFieldAccesses(input, fields);
  }
}


// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
ShardLookupJoin::ShardLookupJoin(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const ShardLookupJoinNode>& joinNode)
    : Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "ShardLookupJoin"),
      outputBatchSize_{outputBatchRows()},
      joinType_{joinNode->joinType()},
      numKeys_{joinNode->leftKeys().size()},
      probeType_{joinNode->probeType()},
      buildOutputType_{joinNode->buildOutputType()},
      shardSetId_{joinNode->shardSetId()},
      numShards_{joinNode->numShards()},
      maxInflightRpcs_{joinNode->maxInflightRpcs()},
      maxBatchSize_{joinNode->maxBatchSize()},
      hashKeyType_{joinNode->hashKeyType()},
      filterExpr_{joinNode->filter()},
      shardLocationMap_{joinNode->shardLocationMap()},
      maxRetainedBatches_{2 * maxInflightRpcs_ * numShards_},
      debugEnabled_([] {
        const char* env = std::getenv("GLUTEN_SHARD_DMJ_DEBUG");
        return env != nullptr && std::string(env) == "1";
      }()) {
  // Resolve leftKey column indices within probeType.
  leftKeyChannels_.reserve(numKeys_);
  for (size_t k = 0; k < numKeys_; ++k) {
    const auto& keyName = joinNode->leftKeys()[k]->name();
    leftKeyChannels_.push_back(probeType_->getChildIdx(keyName));
  }

  // Pre-compute per-column castToBigint flags for shard hash.
  castToBigintFlags_.resize(numKeys_, false);
  for (size_t k = 0; k < numKeys_; ++k) {
    if (hashKeyType_ && k < hashKeyType_->size()) {
      if (hashKeyType_->childAt(k)->kind() == TypeKind::BIGINT &&
          probeType_->childAt(leftKeyChannels_[k])->kind() !=
              TypeKind::BIGINT) {
        castToBigintFlags_[k] = true;
      }
    }
  }

  // Cache probe key RowType (key columns only).
  {
    std::vector<std::string> keyNames;
    std::vector<TypePtr> keyTypes;
    keyNames.reserve(numKeys_);
    keyTypes.reserve(numKeys_);
    for (size_t k = 0; k < numKeys_; ++k) {
      auto channel = leftKeyChannels_[k];
      keyNames.push_back(probeType_->nameOf(channel));
      keyTypes.push_back(probeType_->childAt(channel));
    }
    probeKeyType_ = ROW(std::move(keyNames), std::move(keyTypes));
  }

  // Pre-compute per-shard routing info.  For each shard, if this executor
  // holds the shard table locally, mark it as local.  Otherwise initialise
  // a random round-robin index so that consecutive flushShard calls rotate
  // across replicas, spreading load even within a single task (important
  // for skewed partitions that generate thousands of RPCs to the same shard).
  {
    auto* shardManager = VeloxShardManager::getInstance();
    std::mt19937 rng(std::random_device{}());
    for (int32_t sid = 0; sid < numShards_; ++sid) {
      if (shardManager->hasShardTable(shardSetId_, sid)) {
        localShards_.insert(sid);
        continue;
      }
      auto it = shardLocationMap_.find(sid);
      if (it == shardLocationMap_.end() || it->second.empty()) {
        localShards_.insert(sid); // no locations known, fallback to local
        continue;
      }
      // Random start index so different tasks don't all begin at replica 0.
      shardReplicaRRIndex_[sid] =
          std::uniform_int_distribution<size_t>(0, it->second.size() - 1)(rng);
    }
  }
}

void ShardLookupJoin::initialize() {
  Operator::initialize();
  initOutputProjections();

  // Compile post-join filter if present.
  if (filterExpr_) {
    // Build the filter input type: concat(probeType, buildOutputType).
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (uint32_t i = 0; i < probeType_->size(); ++i) {
      names.push_back(probeType_->nameOf(i));
      types.push_back(probeType_->childAt(i));
    }
    for (uint32_t i = 0; i < buildOutputType_->size(); ++i) {
      names.push_back(buildOutputType_->nameOf(i));
      types.push_back(buildOutputType_->childAt(i));
    }
    filterInputType_ = ROW(std::move(names), std::move(types));

    std::vector<core::TypedExprPtr> filters = {filterExpr_};
    filterExprSet_ = std::make_unique<exec::ExprSet>(
        std::move(filters), operatorCtx_->execCtx());

    // --- Server-side filter pushdown preparation ---
    // Identify which probe columns the filter references, so we can
    // send only those columns to the server for filter evaluation.
    std::unordered_set<std::string> referencedFields;
    collectFieldAccesses(filterExpr_, referencedFields);

    std::vector<std::string> filterProbeNames;
    std::vector<TypePtr> filterProbeTypes;
    for (uint32_t i = 0; i < probeType_->size(); ++i) {
      if (referencedFields.count(probeType_->nameOf(i))) {
        filterProbeChannels_.push_back(static_cast<column_index_t>(i));
        filterProbeNames.push_back(probeType_->nameOf(i));
        filterProbeTypes.push_back(probeType_->childAt(i));
      }
    }
    if (!filterProbeChannels_.empty()) {
      filterProbeType_ = ROW(
          std::move(filterProbeNames), std::move(filterProbeTypes));
    }

    // Serialize filter expression as Velox ISerializable JSON.
    // TODO(perf): filter expression is identical across all RPCs for this
    // operator. A future optimization is to distribute it once during the
    // ColumnarShardExchange build phase and have the server pre-compile,
    // eliminating per-RPC overhead.
    Type::registerSerDe();
    core::ITypedExpr::registerSerDe();
    filterExprJson_ = folly::toJson(filterExpr_->serialize());

    // Build Hive-format type strings for the server.
    // filterInputType_ uses only the referenced probe columns (not full probeType)
    // as its probe-side prefix, matching how we send probe data.
    if (filterProbeType_) {
      filterProbeTypeStr_ =
          type::fbhive::HiveTypeSerializer::serialize(filterProbeType_);

      // Server-side filterInputType = concat(filterProbeType, buildOutputType)
      std::vector<std::string> serverNames;
      std::vector<TypePtr> serverTypes;
      for (uint32_t i = 0; i < filterProbeType_->size(); ++i) {
        serverNames.push_back(filterProbeType_->nameOf(i));
        serverTypes.push_back(filterProbeType_->childAt(i));
      }
      for (uint32_t i = 0; i < buildOutputType_->size(); ++i) {
        serverNames.push_back(buildOutputType_->nameOf(i));
        serverTypes.push_back(buildOutputType_->childAt(i));
      }
      auto serverFilterInputType = ROW(
          std::move(serverNames), std::move(serverTypes));
      serverFilterInputType_ = serverFilterInputType;
      filterInputTypeStr_ =
          type::fbhive::HiveTypeSerializer::serialize(serverFilterInputType);

      // Pre-build dedup type = concat(probeKeyType, filterProbeType).
      std::vector<std::string> dedupNames;
      std::vector<TypePtr> dedupTypes;
      for (size_t k = 0; k < numKeys_; ++k) {
        dedupNames.push_back(probeKeyType_->nameOf(k));
        dedupTypes.push_back(probeKeyType_->childAt(k));
      }
      for (uint32_t i = 0; i < filterProbeType_->size(); ++i) {
        dedupNames.push_back(filterProbeType_->nameOf(i));
        dedupTypes.push_back(filterProbeType_->childAt(i));
      }
      dedupWithFilterType_ = ROW(
          std::move(dedupNames), std::move(dedupTypes));
    }

    if (debugEnabled_) {
      std::string filterProbeColNames;
      for (size_t i = 0; i < filterProbeChannels_.size(); ++i) {
        if (i > 0) filterProbeColNames += ",";
        filterProbeColNames += probeType_->nameOf(filterProbeChannels_[i]);
      }
      LOG(INFO) << "[SLJ-dbg] filterPushdown enabled"
                << " filterProbeColumns=[" << filterProbeColNames << "]"
                << " filterExprJsonLen=" << filterExprJson_.size()
                << " filterProbeType=" << filterProbeTypeStr_
                << " filterInputType=" << filterInputTypeStr_;
    }
  }

  // Fetch shared BF from VeloxShardManager (executor-level cache).
  auto* shardManager = VeloxShardManager::getInstance();
  bloomFilter_ = shardManager->getOrLoadBloomFilter(shardSetId_);

  // Create gRPC completion queue and drain thread.
  completionQueue_ = std::make_unique<grpc::CompletionQueue>();
  drainThread_ = std::thread([this]() {
    void* tag;
    bool ok;
    // Use blocking Next() instead of AsyncNext(deadline) to eliminate
    // up to 100ms latency between RPC completion and pickup.  The thread
    // blocks until either a completion event arrives or the CQ is shut down.
    // Shutdown is triggered by close() which calls completionQueue_->Shutdown().
    while (completionQueue_->Next(&tag, &ok)) {
      auto* ctx = static_cast<RpcContext*>(tag);
      handleRpcCompletion(ctx, ok);
    }
    // CQ returned false → shutdown complete.
  });
}

// ---------------------------------------------------------------------------
// Operator interface
// ---------------------------------------------------------------------------
bool ShardLookupJoin::needsInput() const {
  if (hasFatalError_.load(std::memory_order_acquire)) {
    return false;
  }
  if (noMoreInput_) {
    return false;
  }
  // Don't accept more input if we have output pending.
  if (currentLookup_ != nullptr) {
    return false;
  }
  // Backpressure: don't accept more input if too many in-flight RPCs.
  if (numInFlight_.load(std::memory_order_relaxed) >= maxInflightRpcs_) {
    return false;
  }
  // Backpressure: don't accept more input if BF-negative output queue is
  // building up.  This prevents unbounded memory growth when BF filters
  // nearly all rows and getOutput() consumption lags behind addInput().
  if (!pendingBfNegativeOutputs_.empty()) {
    return false;
  }
  // Backpressure: don't accept more input if too many input batches are
  // retained (pinned by refCount from rows buffered in shardBuffers_ or
  // in-flight/completed RPCs).
  //
  // Each input batch's rows are scattered across numShards_ shard buffers.
  // A batch's refCount only reaches zero after ALL shards that received its
  // rows have been flushed, RPCs completed, and getOutput consumed them.
  // When BF filters most rows, each shard buffer accumulates very few rows
  // per batch, so a single RPC's probeRows span only a few batches — far
  // fewer than the rate at which new batches arrive.  Without this limit,
  // inputBatches_ grows unboundedly and causes OOM.
  //
  // Threshold: maxInflightRpcs * numShards batches ≈ the maximum number of
  // batches whose rows could be spread across all in-flight RPCs.  Beyond
  // this, pausing input lets getOutput drain completed lookups and release
  // batches before accepting more.
  if (isInputBackpressureActive()) {
    if (debugEnabled_) {
      LOG_EVERY_N(INFO, 200)
          << "[SLJ-dbg] needsInput backpressure retainedBatches="
          << inputBatches_.size()
          << " refCount=" << totalRetainedRefCount_
          << " inFlight=" << numInFlight_.load(std::memory_order_relaxed);
    }
    return false;
  }
  return true;
}

void ShardLookupJoin::addInput(RowVectorPtr input) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_GT(input->size(), 0);

  auto startNanos = debugEnabled_
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};

  auto batchIndex = nextBatchId_++;
  auto& entry = inputBatches_[batchIndex];
  entry.batch = input;

  if (debugEnabled_) {
    totalInputRows_ += input->size();
    if (batchIndex % 100 == 0) {
      LOG(INFO) << "[SLJ-dbg] addInput batch=" << batchIndex
                << " retained=" << inputBatches_.size()
                << " refCount=" << totalRetainedRefCount_
                << " inFlight=" << numInFlight_.load(std::memory_order_relaxed)
                << " rows=" << input->size()
                << " totalIn=" << totalInputRows_;
    }
  }

  // refCount is incremented per row that references this batch (routed to
  // a shard buffer).  BF-negative rows are materialized eagerly and do NOT
  // pin the batch.  If no row ends up referencing this batch, the entry is
  // erased at the end of this call.

  const auto numRows = input->size();
  const bool bfActive = bloomFilter_ && bloomFilter_->isSet();

  // Optimization 1: columnar batch hash.
  //
  // Spark's LookupJoinIterator computes BF hash and shard hash per row via
  // codegen UnsafeProjection, paying minimal per-row cost.  The naive port
  // here iterated row-by-row and called SparkMurmurHash::hashColumnAt per
  // (row, key) — each call resolved childAt() (virtual) and SimpleVector cast
  // dynamically, plus a TypeKind switch.  For a 4096-row batch with 4 keys,
  // that was ~16k virtual dispatches just for hashing.
  //
  // Instead, hoist the per-column work outside the per-row loop:
  //   1. For each key column, batch-resolve typed vector once, then
  //      tight-loop hash all rows in this batch into a stack-array.
  //   2. shard hash is identical to BF hash *except* for castToBigint columns,
  //      so when no key needs casting we share the buffer (only one pass).
  //   3. per-row, only do BF check + (mod numShards) + bufferRow — no
  //      virtual dispatch, no type switch, no per-key inner loop.
  bfHashes_.assign(numRows, 42);
  bool anyCast = false;
  for (size_t k = 0; k < numKeys_; ++k) {
    if (castToBigintFlags_[k]) {
      anyCast = true;
      break;
    }
  }

  for (size_t k = 0; k < numKeys_; ++k) {
    SparkMurmurHash::hashColumnBatch(
        input, leftKeyChannels_[k], 0, numRows, bfHashes_.data());
  }

  // Compute shard hashes.  When no key needs casting, shard hash == BF hash;
  // share the buffer to avoid duplicate work.
  int32_t* shardHashesPtr = nullptr;
  if (!anyCast) {
    shardHashesPtr = bfHashes_.data();
  } else {
    shardHashes_.assign(numRows, 42);
    for (size_t k = 0; k < numKeys_; ++k) {
      SparkMurmurHash::hashColumnBatchForShard(
          input,
          leftKeyChannels_[k],
          0,
          numRows,
          castToBigintFlags_[k],
          shardHashes_.data());
    }
    shardHashesPtr = shardHashes_.data();
  }

  // Per-row routing — the only loop that touches every row, kept lean.
  const bool isLeftJoin = (joinType_ == core::JoinType::kLeft);
  // Collect BF-negative row indices for this batch (Left Join only).
  // These are materialized eagerly into output RowVectors so they never
  // pin this batch via refCount.
  std::vector<vector_size_t> bfNegativeIndices;
  vector_size_t batchBfNeg = 0;
  for (vector_size_t row = 0; row < numRows; ++row) {
    if (bfActive &&
        !bloomFilter_->mayContain(spreadHashForBloom(bfHashes_[row]))) {
      ++batchBfNeg;
      if (isLeftJoin) {
        bfNegativeIndices.push_back(row);
      }
      continue;
    }

    int32_t mod = shardHashesPtr[row] % numShards_;
    if (mod < 0) {
      mod += numShards_;
    }
    bufferRow(mod, batchIndex, row);
    ++entry.refCount;
    ++totalRetainedRefCount_;
  }
  if (debugEnabled_) {
    bfNegativeRows_ += batchBfNeg;
    bfPositiveRows_ += (numRows - batchBfNeg);
  }

  // No row referenced this batch — release immediately.
  if (entry.refCount == 0) {
    inputBatches_.erase(batchIndex);
  }


  // Left Join: eagerly materialize BF-negative rows as null-filled output.
  // This avoids pinning inputBatches_ via refCount for BF-negative rows,
  // preventing OOM when BF filters >99% of rows.
  if (!bfNegativeIndices.empty()) {
    auto negCount = static_cast<vector_size_t>(bfNegativeIndices.size());

    // Build an indices buffer to wrap input columns via DictionaryVector.
    auto indexBuf = AlignedBuffer::allocate<vector_size_t>(negCount, pool());
    auto* rawIdx = indexBuf->asMutable<vector_size_t>();
    std::copy(bfNegativeIndices.begin(), bfNegativeIndices.end(), rawIdx);

    // Probe side: wrap each projected input column with DictionaryVector.
    std::vector<VectorPtr> outputCols(outputType_->size());
    for (const auto& proj : probeOutputProjections_) {
      outputCols[proj.outputChannel] = BaseVector::wrapInDictionary(
          nullptr, indexBuf, negCount, input->childAt(proj.inputChannel));
    }
    // Build side: all-null columns.
    for (const auto& proj : buildOutputProjections_) {
      auto nullCol = BaseVector::create(
          buildOutputType_->childAt(proj.inputChannel), negCount, pool());
      for (vector_size_t i = 0; i < negCount; ++i) {
        nullCol->setNull(i, true);
      }
      outputCols[proj.outputChannel] = std::move(nullCol);
    }

    auto result = std::make_shared<RowVector>(
        pool(), outputType_, nullptr, negCount, std::move(outputCols));
    pendingBfNegativeOutputs_.push(std::move(result));
  }

  if (debugEnabled_) {
    addInputNanos_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - startNanos).count();
  }
}

BlockingReason ShardLookupJoin::isBlocked(ContinueFuture* future) {
  // Never report blocked via Velox's promise/future mechanism.
  //
  // Background: Velox Task::next() has a race condition in its
  // notReadyFutures collection loop — if the RPC drain thread fulfills
  // the promise very quickly (< 10 µs), the DriverBlockingState callback
  // chain can make the blockFuture ready before Task::next() collects it
  // into notReadyFutures, causing it to be skipped.  collectAny() then
  // only waits on the forever-blocked HashBuild drivers → permanent hang.
  //
  // Instead, getOutput() synchronously waits on completedCv_ when there
  // are in-flight RPCs but no completed results.  This completely bypasses
  // the DriverBlockingState layer and its race.
  return BlockingReason::kNotBlocked;
}

RowVectorPtr ShardLookupJoin::getOutput() {
  ++getOutputCalls_;

  if (debugEnabled_ && getOutputCalls_ % 200 == 0) {
    size_t queueSize = 0;
    {
      std::lock_guard<std::mutex> lock(completedMutex_);
      queueSize = completedQueue_.size();
    }
    LOG(INFO) << "[SLJ-dbg] getOutput#" << getOutputCalls_
              << " retained=" << inputBatches_.size()
              << " refCount=" << totalRetainedRefCount_
              << " inFlight=" << numInFlight_.load(std::memory_order_relaxed)
              << " completedQ=" << queueSize
              << " totalBatchIn=" << nextBatchId_
              << " rpcIssued=" << rpcIssuedCount_
              << " rpcDone=" << rpcCompletedCount_;
  }

  // Propagate fatal RPC errors to the driver so Spark can retry the task.
  if (hasFatalError_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(fatalErrorMutex_);
    VELOX_FAIL(fatalErrorMessage_);
  }

  // Try to get a completed lookup.  If there are in-flight RPCs but no
  // completed results, synchronously wait on completedCv_ instead of
  // returning nullptr and relying on Velox's DriverBlockingState
  // promise/future mechanism (which has a race in Task::next()).
  if (currentLookup_ == nullptr) {
    std::unique_lock<std::mutex> lock(completedMutex_);
    if (completedQueue_.empty() && numInFlight_ > 0) {
      auto waitStart = debugEnabled_
          ? std::chrono::steady_clock::now()
          : std::chrono::steady_clock::time_point{};
      completedCv_.wait(lock, [this] {
        return !completedQueue_.empty() || numInFlight_ == 0 ||
            hasFatalError_.load(std::memory_order_acquire);
      });
      if (debugEnabled_) {
        getOutputWaitNanos_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - waitStart).count();
      }
    }
    // Re-check fatal error after waking up.
    if (hasFatalError_.load(std::memory_order_acquire)) {
      lock.unlock();
      std::lock_guard<std::mutex> errLock(fatalErrorMutex_);
      VELOX_FAIL(fatalErrorMessage_);
    }
    // Drain all completed lookups that produce no output rows (e.g. when
    // BF filters most rows and the server returns empty results).  This
    // releases their probeRows' refCounts immediately instead of waiting
    // for one-per-getOutput-call consumption, which is critical when BF
    // filter rates are high and most RPCs return zero rows — without this,
    // inputBatches_ accumulates because releaseProbeRefs is called too
    // infrequently relative to the addInput rate.
    while (!completedQueue_.empty()) {
      auto lookup = std::move(completedQueue_.front());
      completedQueue_.pop();
      if (lookup->totalOutputRows == 0) {
        // No output to produce — release probe refs immediately.
        lock.unlock();
        releaseProbeRefs(lookup->probeRows);
        lock.lock();
        continue;
      }
      currentLookup_ = std::move(lookup);
      currentLookupOutputRow_ = 0;
      if (currentLookup_->totalOutputRows == 0) {
        releaseProbeRefs(currentLookup_->probeRows);
        currentLookup_ = nullptr;
        continue;
      }
      break;
    }
  }

  if (currentLookup_ == nullptr) {
    // Left Join: emit pre-materialized BF-negative output batches.
    if (!pendingBfNegativeOutputs_.empty()) {
      auto result = std::move(pendingBfNegativeOutputs_.front());
      pendingBfNegativeOutputs_.pop();
      return result;
    }

    // If noMoreInput and all RPCs done, flush tailing shard buffers.
    if (noMoreInput_ && numInFlight_ == 0 && !flushedTailing_) {
      flushAllShards();
      flushedTailing_ = true;
    }

    // Proactive flush: only when backpressure is blocking addInput, flush
    // buffered rows to make progress.  Without this, the driver would spin:
    // needsInput()=false → getOutput() returns nullptr → repeat.  By
    // flushing, we create in-flight RPCs so the next getOutput() call will
    // wait on completedCv_ instead of busy-looping.
    //
    // When backpressure is NOT active, the driver will call addInput() next,
    // allowing shard buffers to fill naturally to maxBatchSize before
    // flushing — producing large-batch RPCs for better throughput.
    bool backpressureActive = isInputBackpressureActive();
    if (!noMoreInput_ && numInFlight_ == 0 && backpressureActive) {
      bool hasBufferedRows = false;
      for (const auto& [_, buf] : shardBuffers_) {
        if (!buf.rows.empty()) {
          hasBufferedRows = true;
          break;
        }
      }
      if (hasBufferedRows) {
        if (debugEnabled_) {
          LOG_EVERY_N(INFO, 50)
              << "[SLJ-dbg] proactiveFlush retained="
              << inputBatches_.size()
              << " refCount=" << totalRetainedRefCount_;
        }
        flushAllShards();
      }
    }

    return nullptr;
  }

  // Produce output from current lookup.
  RowVectorPtr result;
  if (joinType_ == core::JoinType::kInner) {
    result = produceOutputForInnerJoin(*currentLookup_);
  } else {
    result = produceOutputForLeftJoin(*currentLookup_);
  }

  // If we've consumed all rows from this lookup, release it AND drop
  // refCounts on the input batches that backed its probe rows.  Once an
  // entry's refCount hits zero the batch is erased from inputBatches_,
  // releasing the upstream RowVectorPtr early.
  if (currentLookup_ &&
      currentLookupOutputRow_ >= currentLookup_->totalOutputRows) {
    releaseProbeRefs(currentLookup_->probeRows);
    currentLookup_ = nullptr;
    currentLookupOutputRow_ = 0;
  }

  if (result && result->size() > 0) {
    if (debugEnabled_) {
      totalProducedRows_ += result->size();
    }
    return result;
  }
  return nullptr;
}

bool ShardLookupJoin::isFinished() {
  if (!noMoreInput_) {
    return false;
  }
  if (numInFlight_ > 0) {
    return false;
  }
  if (currentLookup_ != nullptr) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(completedMutex_);
    if (!completedQueue_.empty()) {
      return false;
    }
  }
  if (!flushedTailing_) {
    return false;
  }
  // Left Join: pending BF-negative outputs must be emitted.
  if (!pendingBfNegativeOutputs_.empty()) {
    return false;
  }
  return true;
}

void ShardLookupJoin::close() {
  if (debugEnabled_) {
    LOG(INFO) << "[SLJ-dbg] close() cumulative stats:"
              << " totalInputRows=" << totalInputRows_
              << " bfNeg=" << bfNegativeRows_
              << " bfPos=" << bfPositiveRows_
              << " rpcIssued=" << rpcIssuedCount_
              << " rpcDone=" << rpcCompletedCount_
              << " localLookups=" << localLookupCount_
              << " flushes=" << flushShardCount_
              << " totalBuildRows=" << totalBuildOutputRows_
              << " totalProducedRows=" << totalProducedRows_
              << " getOutputCalls=" << getOutputCalls_;
    LOG(INFO) << "[SLJ-dbg] close() timing (wall-clock, includes waits):"
              << " addInputMs=" << (addInputNanos_ / 1000000)
              << " flushShardMs=" << (flushShardNanos_ / 1000000)
              << " getOutputWaitMs=" << (getOutputWaitNanos_ / 1000000)
              << " localLookupMs=" << (localLookupNanos_ / 1000000);
    auto* driver = operatorCtx_->driverCtx()->driver;
    if (driver) {
      for (auto* op : driver->operators()) {
        auto opStats = op->stats(false);
        LOG(INFO) << "[SLJ-dbg] pipeline op[" << opStats.operatorId << "] "
                  << opStats.operatorType
                  << ": getOutputMs=" << (opStats.getOutputTiming.wallNanos / 1000000)
                  << " getOutputCpu=" << (opStats.getOutputTiming.cpuNanos / 1000000)
                  << " addInputMs=" << (opStats.addInputTiming.wallNanos / 1000000)
                  << " addInputCpu=" << (opStats.addInputTiming.cpuNanos / 1000000)
                  << " outputRows=" << opStats.outputPositions
                  << " outputBatches=" << opStats.outputVectors
                  << " inputRows=" << opStats.inputPositions;
      }
    }
  }

  Operator::close();
  if (completionQueue_) {
    completionQueue_->Shutdown();
  }
  if (drainThread_.joinable()) {
    drainThread_.join();
  }
  completionQueue_.reset();

  // Release currentLookup_ if we were in the middle of outputting.
  currentLookup_.reset();
  currentLookupOutputRow_ = 0;

  // Drain and release all unconsumed CompletedLookups in the queue.
  // Each holds a buildOutput RowVectorPtr allocated from pool() — if not
  // released here, that memory leaks until the operator is destructed.
  {
    std::lock_guard<std::mutex> lock(completedMutex_);
    while (!completedQueue_.empty()) {
      completedQueue_.pop();
    }
  }

  // Drop refCounts contributed by all still-buffered rows so any leftover
  // inputBatches_ entries are released here (in the unlikely case that
  // close() is called before all RPCs have been consumed).
  for (auto& [sid, buf] : shardBuffers_) {
    releaseProbeRefs(buf.rows);
  }
  // Whatever remains (entries still referenced by an in-flight RPC's
  // CompletedLookup that the Driver never picked up) is force-released.
  inputBatches_.clear();
  shardBuffers_.clear();
  // Drain pending BF-negative outputs.
  while (!pendingBfNegativeOutputs_.empty()) {
    pendingBfNegativeOutputs_.pop();
  }

  // Release reusable output vector and stub cache.
  output_.reset();
  stubCache_.clear();
}


void ShardLookupJoin::setFatalError(const std::string& message) {
  std::lock_guard<std::mutex> lock(fatalErrorMutex_);
  if (!hasFatalError_.load(std::memory_order_relaxed)) {
    fatalErrorMessage_ = message;
    hasFatalError_.store(true, std::memory_order_release);
  }
}

void ShardLookupJoin::failRetry(const std::string& message) {
  setFatalError(message);
  {
    std::lock_guard<std::mutex> lock(completedMutex_);
    numInFlight_--;
  }
  completedCv_.notify_all();
}

// ---------------------------------------------------------------------------
// Per-shard buffering
// ---------------------------------------------------------------------------
void ShardLookupJoin::bufferRow(
    int32_t shardId,
    uint32_t batchIndex,
    vector_size_t row) {
  auto& buffer = shardBuffers_[shardId];
  buffer.rows.push_back(BufferedRow{batchIndex, row});

  if (static_cast<int32_t>(buffer.rows.size()) >= maxBatchSize_) {
    flushShard(shardId);
  }
}

void ShardLookupJoin::flushShard(int32_t shardId) {
  auto it = shardBuffers_.find(shardId);
  if (it == shardBuffers_.end() || it->second.rows.empty()) {
    return;
  }
  auto startNanos = debugEnabled_
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
  if (debugEnabled_) {
    ++flushShardCount_;
  }
  auto rows = std::move(it->second.rows);
  it->second.rows.clear();

  // Extract probe keys via DictionaryVector wrapping (zero-copy for
  // single-batch case, copyRanges per segment for multi-batch).
  auto numRows = static_cast<vector_size_t>(rows.size());
  std::vector<VectorPtr> keyColumns(numKeys_);
  for (size_t k = 0; k < numKeys_; ++k) {
    auto channel = leftKeyChannels_[k];
    auto keyType = probeType_->childAt(channel);
    keyColumns[k] = gatherProbeKeyColumnAsDictionary(channel, keyType, rows);
  }

  auto allKeys = std::make_shared<RowVector>(
      pool(), probeKeyType_, nullptr, numRows, std::move(keyColumns));

  // Gather filter probe columns if server-side filter pushdown is active.
  RowVectorPtr filterProbeColumns;
  if (filterExpr_ && !filterProbeChannels_.empty()) {
    std::vector<VectorPtr> filterCols(filterProbeChannels_.size());
    for (size_t i = 0; i < filterProbeChannels_.size(); ++i) {
      auto channel = filterProbeChannels_[i];
      auto colType = probeType_->childAt(channel);
      filterCols[i] = gatherProbeKeyColumnAsDictionary(channel, colType, rows);
    }
    filterProbeColumns = std::make_shared<RowVector>(
        pool(), filterProbeType_, nullptr, numRows, std::move(filterCols));
  }

  // --- Probe key dedup (mirrors Spark DMJ's KeyValueBatch.wrapKeysBuffer) ---
  //
  // Serialize each probe key to CompactRow bytes and dedup by content.
  // Only unique keys are sent to the server; the dedupMap records which
  // original probeRow indices share each unique key so results can be
  // expanded after the RPC returns.
  //
  // When server-side filter is active, the dedup key includes both probe
  // keys and filter probe columns. This prevents rows with same key but
  // different filter column values from being merged (which would cause
  // incorrect server-side filter evaluation).
  //
  // For a batch with K unique keys out of N total rows, this reduces:
  //   - RPC payload size by (N-K)/N
  //   - Server-side joinProbe calls from N to K
  //   - Server-side extractColumn rows from N*fanout to K*fanout
  //   - Response size from N*fanout to K*fanout

  // Build the RowVector used for dedup serialization.
  // Without filter: dedup by probe keys only.
  // With filter: dedup by concat(probeKeys, filterProbeColumns).
  RowVectorPtr dedupInput;
  if (filterProbeColumns) {
    std::vector<VectorPtr> combinedCols;
    combinedCols.reserve(numKeys_ + filterProbeChannels_.size());
    for (size_t k = 0; k < numKeys_; ++k) {
      combinedCols.push_back(allKeys->childAt(k));
    }
    for (size_t i = 0; i < filterProbeChannels_.size(); ++i) {
      combinedCols.push_back(filterProbeColumns->childAt(i));
    }
    dedupInput = std::make_shared<RowVector>(
        pool(), dedupWithFilterType_, nullptr, numRows,
        std::move(combinedCols));
  } else {
    dedupInput = allKeys;
  }

  facebook::velox::row::CompactRow compactRow(dedupInput);

  // Two-pass serialize into a single contiguous buffer to avoid N heap
  // allocations (one std::string per row).  Pass 1 computes sizes/offsets;
  // pass 2 serializes into the pre-allocated arena.
  std::vector<size_t> rowSizes(numRows);
  std::vector<size_t> rowOffsets(numRows);
  size_t totalBytes = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    auto sz = compactRow.rowSize(i);
    rowOffsets[i] = totalBytes;
    rowSizes[i] = sz;
    totalBytes += sz;
  }
  std::string serializeBuf(totalBytes, '\0');
  for (vector_size_t i = 0; i < numRows; ++i) {
    compactRow.serialize(i, serializeBuf.data() + rowOffsets[i]);
  }

  // Build unique key index: keyMap[serializedKey] = uniqueIndex.
  // dedupMap[uniqueIndex] = list of original row indices sharing this key.
  // string_views point into serializeBuf which is stable for the
  // lifetime of this function.
  std::unordered_map<std::string_view, int32_t> keyMap;
  keyMap.reserve(numRows);
  DedupMap dedupMap;
  dedupMap.reserve(numRows); // upper bound
  std::vector<vector_size_t> uniqueRowIndices; // row index of representative
  uniqueRowIndices.reserve(numRows);

  for (vector_size_t i = 0; i < numRows; ++i) {
    std::string_view sv(serializeBuf.data() + rowOffsets[i], rowSizes[i]);
    auto [mapIt, inserted] = keyMap.emplace(sv, static_cast<int32_t>(dedupMap.size()));
    if (inserted) {
      dedupMap.push_back({i});
      uniqueRowIndices.push_back(i);
    } else {
      dedupMap[mapIt->second].push_back(i);
    }
  }

  auto numUniqueKeys = static_cast<vector_size_t>(uniqueRowIndices.size());

  // Build the unique-keys RowVector by slicing allKeys with an indices buffer.
  RowVectorPtr probeKeys;
  if (numUniqueKeys == numRows) {
    // No duplicates — skip the DictionaryVector overhead.
    probeKeys = allKeys;
    dedupMap.clear(); // empty = no expansion needed
  } else {
    auto indexBuf = AlignedBuffer::allocate<vector_size_t>(
        numUniqueKeys, pool());
    auto* rawIdx = indexBuf->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numUniqueKeys; ++i) {
      rawIdx[i] = uniqueRowIndices[i];
    }
    std::vector<VectorPtr> uniqueKeyColumns(numKeys_);
    for (size_t k = 0; k < numKeys_; ++k) {
      uniqueKeyColumns[k] = BaseVector::wrapInDictionary(
          /*nulls=*/nullptr, indexBuf, numUniqueKeys,
          allKeys->childAt(k));
    }
    probeKeys = std::make_shared<RowVector>(
        pool(), probeKeyType_, nullptr, numUniqueKeys,
        std::move(uniqueKeyColumns));

    // Slice filter probe columns to unique rows as well.
    if (filterProbeColumns) {
      std::vector<VectorPtr> uniqueFilterCols(filterProbeChannels_.size());
      for (size_t i = 0; i < filterProbeChannels_.size(); ++i) {
        uniqueFilterCols[i] = BaseVector::wrapInDictionary(
            /*nulls=*/nullptr, indexBuf, numUniqueKeys,
            filterProbeColumns->childAt(i));
      }
      filterProbeColumns = std::make_shared<RowVector>(
          pool(), filterProbeType_, nullptr, numUniqueKeys,
          std::move(uniqueFilterCols));
    }
  }

  if (localShards_.count(shardId) > 0) {
    issueLocalLookup(
        shardId, probeKeys, std::move(rows), std::move(dedupMap),
        filterProbeColumns);
  } else {
    issueRemoteLookup(
        shardId, probeKeys, std::move(rows), std::move(dedupMap),
        filterProbeColumns);
  }

  if (debugEnabled_) {
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - startNanos).count();
    flushShardNanos_ += elapsed;
    if (flushShardCount_ % 50 == 0) {
      LOG(INFO) << "[SLJ-dbg] flushShard#" << flushShardCount_
                << " shard=" << shardId
                << " rows=" << numRows
                << " uniqueKeys=" << numUniqueKeys
                << " thisMs=" << (elapsed / 1000000)
                << " cumulMs=" << (flushShardNanos_ / 1000000)
                << " local=" << (localShards_.count(shardId) > 0);
    }
  }
}

// ---------------------------------------------------------------------------
// expandDedupResult – expand dedup'd server result back to original probe rows.
//
// Server returns (buildOutput, inputHits) where inputHits[i] indexes into
// the unique-key list.  For each build row i with uniqueKeyIdx = inputHits[i],
// dedupMap[uniqueKeyIdx] lists all original probeRow indices sharing that key.
// We replicate each build row K times (K = dedupMap[uniqueKeyIdx].size()) and
// map inputHits to original probeRow indices.
//
// Build-side replication uses DictionaryVector wrapping (zero physical copy).
// The only new memory is indexBuf (totalExpanded * 4B) + expandedHits vector.
// produceOutput already emits rows in outputBatchSize_ slices, so there is
// no need to split the expansion into chunks.
// ---------------------------------------------------------------------------
static void expandDedupResult(
    const RowVectorPtr& serverBuildOutput,
    const std::vector<int32_t>& serverInputHits,
    const std::vector<std::vector<int32_t>>& dedupMap,
    facebook::velox::memory::MemoryPool* memPool,
    RowVectorPtr& expandedBuildOutput,
    std::vector<int32_t>& expandedInputHits) {
  auto serverRows = serverBuildOutput->size();
  VELOX_CHECK_EQ(
      serverRows, serverInputHits.size(),
      "buildOutput size must match inputHits size");

  // Compute total expanded size (with bounds check).
  auto dedupMapSize = static_cast<int32_t>(dedupMap.size());
  size_t totalExpanded = 0;
  for (size_t i = 0; i < serverRows; ++i) {
    auto uniqueIdx = serverInputHits[i];
    VELOX_CHECK_GE(uniqueIdx, 0, "inputHits contains negative index");
    VELOX_CHECK_LT(
        uniqueIdx, dedupMapSize,
        "inputHits[{}]={} out of dedupMap range [0, {})",
        i, uniqueIdx, dedupMapSize);
    totalExpanded += dedupMap[uniqueIdx].size();
  }

  auto indexBuf = AlignedBuffer::allocate<vector_size_t>(totalExpanded, memPool);
  auto* rawIdx = indexBuf->asMutable<vector_size_t>();
  expandedInputHits.resize(totalExpanded);
  size_t pos = 0;
  for (size_t i = 0; i < serverRows; ++i) {
    auto uniqueIdx = serverInputHits[i];
    for (int32_t origIdx : dedupMap[uniqueIdx]) {
      rawIdx[pos] = static_cast<vector_size_t>(i);
      expandedInputHits[pos] = origIdx;
      ++pos;
    }
  }

  auto numCols = serverBuildOutput->childrenSize();
  std::vector<VectorPtr> cols(numCols);
  for (size_t c = 0; c < numCols; ++c) {
    cols[c] = BaseVector::wrapInDictionary(
        nullptr, indexBuf,
        static_cast<vector_size_t>(totalExpanded),
        serverBuildOutput->childAt(c));
  }
  expandedBuildOutput = std::make_shared<RowVector>(
      memPool, serverBuildOutput->type(), nullptr,
      static_cast<vector_size_t>(totalExpanded), std::move(cols));
}

void ShardLookupJoin::flushAllShards() {
  // Collect shard ids first to avoid modifying map during iteration.
  std::vector<int32_t> shardIds;
  for (auto& [sid, buf] : shardBuffers_) {
    if (!buf.rows.empty()) {
      shardIds.push_back(sid);
    }
  }
  for (auto sid : shardIds) {
    flushShard(sid);
  }
}

// ---------------------------------------------------------------------------
// RPC — local lookup
// ---------------------------------------------------------------------------
void ShardLookupJoin::issueLocalLookup(
    int32_t shardId,
    RowVectorPtr probeKeys,
    std::vector<BufferedRow> probeRows,
    DedupMap dedupMap,
    RowVectorPtr filterProbeColumns) {
  auto startNanos = debugEnabled_
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
  if (debugEnabled_) {
    ++localLookupCount_;
  }
  auto* shardManager = VeloxShardManager::getInstance();
  auto numRows = static_cast<int32_t>(probeKeys->size());
  std::vector<int32_t> inputIndices(numRows);
  std::iota(inputIndices.begin(), inputIndices.end(), 0);

  auto result = shardManager->lookup(
      shardSetId_, shardId, probeKeys, inputIndices, pool(),
      filterProbeColumns, filterExprJson_, nullptr, serverFilterInputType_);

  if (debugEnabled_) {
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - startNanos).count();
    localLookupNanos_ += elapsed;
    if (localLookupCount_ % 50 == 0) {
      auto buildRows = (result.output ? result.output->size() : 0);
      LOG(INFO) << "[SLJ-dbg] localLookup#" << localLookupCount_
                << " shard=" << shardId
                << " probeKeys=" << numRows
                << " buildRows=" << buildRows
                << " thisMs=" << (elapsed / 1000000)
                << " cumulMs=" << (localLookupNanos_ / 1000000);
    }
  }

  if (result.output && result.output->size() > 0 && !dedupMap.empty()) {
    // Expand dedup'd result back to original probe row indices.
    RowVectorPtr expandedBuild;
    std::vector<int32_t> expandedHits;
    expandDedupResult(
        result.output, result.inputHits, dedupMap, pool(),
        expandedBuild, expandedHits);

    auto completed = std::make_unique<CompletedLookup>();
    completed->shardId = shardId;
    completed->probeRows = std::move(probeRows);
    completed->buildOutput = std::move(expandedBuild);
    completed->inputHits = std::move(expandedHits);
    computeTotalOutputRows(*completed);
    if (joinType_ == core::JoinType::kInner &&
        completed->totalOutputRows == 0) {
      releaseProbeRefs(completed->probeRows);
      return;
    }
    onRpcComplete(std::move(completed));
    return;
  }

  auto completed = std::make_unique<CompletedLookup>();
  completed->shardId = shardId;
  completed->probeRows = std::move(probeRows);
  if (result.output && result.output->size() > 0) {
    completed->buildOutput = result.output;
    completed->inputHits = std::move(result.inputHits);
  } else {
    completed->buildOutput = BaseVector::create<RowVector>(
        buildOutputType_, 0, pool());
  }
  computeTotalOutputRows(*completed);
  if (joinType_ == core::JoinType::kInner &&
      completed->totalOutputRows == 0) {
    releaseProbeRefs(completed->probeRows);
    return;
  }
  onRpcComplete(std::move(completed));
}

// ---------------------------------------------------------------------------
// RPC — remote lookup (async gRPC)
// ---------------------------------------------------------------------------

/// Internal context for a single async gRPC call.
/// The stub must outlive the async RPC, so it is stored here.
struct ShardLookupJoin::RpcContext {
  ShardLookupJoin* owner;
  int32_t shardId;
  std::vector<BufferedRow> probeRows;
  grpc::ClientContext clientCtx;
  proto::LookupRequest request;
  proto::LookupResponse response;
  grpc::Status status;
  std::shared_ptr<proto::ShardLookupService::Stub> stub;
  std::unique_ptr<grpc::ClientAsyncResponseReader<proto::LookupResponse>> rpc;
  std::unordered_set<std::string> triedTargets;
  std::string lastErrorMsg;
  /// The target address ("host:port") used for the current/last RPC attempt.
  /// Used to remove the failed location from shardLocationMap_ on failure.
  std::string lastTarget;
  /// Dedup expansion map: dedupMap[uniqueKeyIdx] = original probeRows indices.
  /// Empty when no dedup was applied (all keys unique).
  DedupMap dedupMap;
};

void ShardLookupJoin::issueRemoteLookup(
    int32_t shardId,
    RowVectorPtr probeKeys,
    std::vector<BufferedRow> probeRows,
    DedupMap dedupMap,
    RowVectorPtr filterProbeColumns) {
  numInFlight_++;
  ++rpcIssuedCount_;

  if (debugEnabled_ && rpcIssuedCount_ % 50 == 0) {
    LOG(INFO) << "[SLJ-dbg] issueRpc#" << rpcIssuedCount_
              << " shard=" << shardId
              << " probeRows=" << probeRows.size()
              << " probeKeys=" << probeKeys->size()
              << " filterCols=" << (filterProbeColumns ? filterProbeColumns->size() : 0)
              << " inFlight=" << numInFlight_.load(std::memory_order_relaxed);
  }


  auto* ctx = new RpcContext();
  ctx->owner = this;
  ctx->shardId = shardId;
  ctx->probeRows = std::move(probeRows);
  ctx->dedupMap = std::move(dedupMap);
  // Serialize probe keys (already dedup'd by flushShard).
  ctx->request.set_set_id(shardSetId_);
  ctx->request.set_shard_id(shardId);
  auto serialized = serializeCompactRowBatch(probeKeys, pool());
  ctx->request.set_probe_keys_compact(std::move(serialized));
  for (int32_t i = 0; i < static_cast<int32_t>(probeKeys->size()); ++i) {
    ctx->request.add_input_indices(i);
  }

  // Set server-side filter pushdown fields.
  // TODO(perf): filter_expression / probe_columns_type / filter_input_type
  // are identical across all RPCs. Future optimization: distribute once
  // during ColumnarShardExchange build phase.
  if (filterProbeColumns && filterProbeColumns->size() > 0 &&
      !filterExprJson_.empty()) {
    ctx->request.set_probe_filter_columns_compact(
        serializeCompactRowBatch(filterProbeColumns, pool()));
    ctx->request.set_filter_expression(filterExprJson_);
    ctx->request.set_probe_columns_type(filterProbeTypeStr_);
    ctx->request.set_filter_input_type(filterInputTypeStr_);
  }

  ctx->clientCtx.set_deadline(
      std::chrono::system_clock::now() + std::chrono::seconds(120));

  std::string target;
  auto* shardManager = VeloxShardManager::getInstance();
  {
    std::lock_guard<std::mutex> lock(routeMutex_);
    // Round-robin across replicas for this shard, skipping blacklisted
    // addresses (executor-level dead address list).
    auto locIt = shardLocationMap_.find(shardId);
    bool needRefresh = (locIt == shardLocationMap_.end() ||
                        locIt->second.empty());
    if (!needRefresh) {
      auto& locs = locIt->second;
      auto& idx = shardReplicaRRIndex_[shardId];
      auto numReplicas = locs.size();
      for (size_t attempt = 0; attempt < numReplicas; ++attempt) {
        const auto& loc = locs[(idx + attempt) % numReplicas];
        auto addr = loc.host + ":" + std::to_string(loc.port);
        if (!shardManager->isAddressDead(addr)) {
          target = addr;
          idx += attempt + 1;
          break;
        }
      }
      if (target.empty()) {
        needRefresh = true;
      }
    }
  }

  // If no locations available (e.g. all replicas removed by failover or
  // blacklisted), refresh from master before giving up.
  if (target.empty()) {
    LOG(WARNING) << "ShardLookupJoin::issueRemoteLookup: no locations for "
                 << "shard " << shardId << ", refreshing from master";
    auto freshLocations =
        shardManager->refreshShardLocations(shardSetId_, shardId);
    {
      std::lock_guard<std::mutex> lock(routeMutex_);
      std::vector<ShardServerLocation> newLocs;
      newLocs.reserve(freshLocations.size());
      for (const auto& addr : freshLocations) {
        auto colonPos = addr.rfind(':');
        if (colonPos != std::string::npos) {
          newLocs.emplace_back(
              addr.substr(0, colonPos),
              std::stoi(addr.substr(colonPos + 1)));
        }
      }
      shardLocationMap_[shardId] = std::move(newLocs);
      auto& updatedLocs = shardLocationMap_[shardId];
      if (!updatedLocs.empty()) {
        auto& idx = shardReplicaRRIndex_[shardId];
        idx = static_cast<size_t>(std::rand() % updatedLocs.size());
        for (size_t i = 0; i < updatedLocs.size(); ++i) {
          const auto& loc = updatedLocs[(idx + i) % updatedLocs.size()];
          auto addr = loc.host + ":" + std::to_string(loc.port);
          if (!shardManager->isAddressDead(addr)) {
            target = addr;
            idx += i + 1;
            break;
          }
        }
      }
    }
    VELOX_CHECK(
        !target.empty(),
        "issueRemoteLookup: no live locations for shard {} even after refresh"
        " (all blacklisted or empty)",
        shardId);
  }

  {
    std::lock_guard<std::mutex> lock(routeMutex_);
    auto& stub = stubCache_[target];
    if (!stub) {
      auto* shardManager = VeloxShardManager::getInstance();
      auto channel = shardManager->getOrCreateChannel(target);
      stub = proto::ShardLookupService::NewStub(channel);
    }
    ctx->stub = stub;
  }
  ctx->triedTargets.insert(target);
  ctx->lastTarget = target;
  ctx->rpc = ctx->stub->AsyncLookup(
      &ctx->clientCtx, ctx->request, completionQueue_.get());
  ctx->rpc->Finish(&ctx->response, &ctx->status, static_cast<void*>(ctx));
}

void ShardLookupJoin::retryRemoteLookup(std::unique_ptr<RpcContext> oldCtx) {
  const auto shardId = oldCtx->shardId;
  auto triedTargets = std::move(oldCtx->triedTargets);
  auto lastError = std::move(oldCtx->lastErrorMsg);
  auto* shardManager = VeloxShardManager::getInstance();

  // --- Determine next target ---
  // Find the first untried, non-blacklisted replica from the local
  // shardLocationMap_.  This is per-RpcContext (via triedTargets), so
  // concurrent RPCs for the same shard don't interfere.
  // If all local replicas are exhausted, refresh from master.
  std::string newTarget;
  {
    std::lock_guard<std::mutex> lock(routeMutex_);
    auto locIt = shardLocationMap_.find(shardId);
    if (locIt != shardLocationMap_.end()) {
      for (const auto& loc : locIt->second) {
        auto addr = loc.host + ":" + std::to_string(loc.port);
        if (triedTargets.count(addr) == 0 &&
            !shardManager->isAddressDead(addr)) {
          newTarget = addr;
          break;
        }
      }
    }
  }

  if (newTarget.empty()) {
    // All local replicas tried or blacklisted → refresh from master.
    LOG(WARNING) << "ShardLookupJoin: all local replicas tried for shard "
                 << shardId << ", refreshing locations from master";
    auto freshLocations =
        shardManager->refreshShardLocations(shardSetId_, shardId);

    // Log what refresh returned for troubleshooting.
    {
      std::string refreshedStr;
      for (const auto& loc : freshLocations) {
        if (!refreshedStr.empty()) refreshedStr += ", ";
        refreshedStr += loc;
      }
      std::string triedStr;
      for (const auto& t : triedTargets) {
        if (!triedStr.empty()) triedStr += ", ";
        triedStr += t;
      }
      LOG(WARNING) << "ShardLookupJoin: refreshShardLocations returned ["
                   << refreshedStr << "] for shard " << shardId
                   << ", already tried [" << triedStr << "]";
    }

    // Update shardLocationMap_ with refreshed locations so subsequent
    // probe batches and failover calls use the fresh list.
    // Also reset the round-robin index to start fresh with the new replicas.
    {
      std::lock_guard<std::mutex> lock(routeMutex_);
      std::vector<ShardServerLocation> newLocs;
      newLocs.reserve(freshLocations.size());
      for (const auto& addr : freshLocations) {
        auto colonPos = addr.rfind(':');
        if (colonPos != std::string::npos) {
          newLocs.emplace_back(
              addr.substr(0, colonPos),
              std::stoi(addr.substr(colonPos + 1)));
        }
      }
      shardLocationMap_[shardId] = std::move(newLocs);
      // Use a random start index so concurrent tasks that refresh
      // the same shard don't all converge on replica 0.
      auto& updatedLocs = shardLocationMap_[shardId];
      if (!updatedLocs.empty()) {
        shardReplicaRRIndex_[shardId] =
            static_cast<size_t>(std::rand() % updatedLocs.size());
      }
    }

    // Find the first untried, non-blacklisted location from the refreshed list.
    newTarget.clear();
    for (const auto& loc : freshLocations) {
      if (triedTargets.count(loc) == 0 &&
          !shardManager->isAddressDead(loc)) {
        newTarget = loc;
        break;
      }
    }
    if (newTarget.empty()) {
      std::string fatalMsg =
          "ShardLookupJoin: no live untried replicas for shard " +
          std::to_string(shardId) + " after refreshShardLocations " +
          "(tried " + std::to_string(triedTargets.size()) +
          " replicas, remaining blacklisted), last RPC error: " + lastError;
      LOG(ERROR) << fatalMsg;
      failRetry(fatalMsg);
      return;
    }
    // No need to update a global route cache — per-RPC round-robin picks
    // the next replica each time.  The retry just uses newTarget directly.
  }

  // Build a fresh RpcContext, reusing the serialized request and probe rows.
  auto* ctx = new RpcContext();
  ctx->owner = this;
  ctx->shardId = shardId;
  ctx->probeRows = std::move(oldCtx->probeRows);
  ctx->dedupMap = std::move(oldCtx->dedupMap);
  ctx->triedTargets = std::move(triedTargets);
  ctx->triedTargets.insert(newTarget);
  ctx->lastTarget = newTarget;
  ctx->lastErrorMsg = std::move(lastError);

  // Copy the already-serialized request (avoids re-serialization).
  ctx->request = oldCtx->request;

  ctx->clientCtx.set_deadline(
      std::chrono::system_clock::now() + std::chrono::seconds(120));

  {
    std::lock_guard<std::mutex> lock(routeMutex_);
    auto& stub = stubCache_[newTarget];
    if (!stub) {
      auto* shardManager = VeloxShardManager::getInstance();
      auto channel = shardManager->getOrCreateChannel(newTarget);
      stub = proto::ShardLookupService::NewStub(channel);
    }
    ctx->stub = stub;
  }

  LOG(WARNING) << "ShardLookupJoin retrying shard " << shardId
               << " on " << newTarget
               << " (tried " << ctx->triedTargets.size() << " replicas)";

  ctx->rpc = ctx->stub->AsyncLookup(
      &ctx->clientCtx, ctx->request, completionQueue_.get());
  ctx->rpc->Finish(&ctx->response, &ctx->status, static_cast<void*>(ctx));
  // numInFlight_ is NOT incremented — the original issueRemoteLookup already
  // counted this RPC, and retries inherit that count.
}

void ShardLookupJoin::handleRpcCompletion(RpcContext* ctx, bool ok) {
  std::unique_ptr<RpcContext> owned(ctx);

  if (!ok || !ctx->status.ok()) {
    const auto errorMsg = ok ? ctx->status.error_message() : "CQ shutdown";
    LOG(WARNING) << "ShardLookupJoin RPC failed for shard " << ctx->shardId
                 << " (tried " << ctx->triedTargets.size()
                 << " replicas), will retry: " << errorMsg;

    // Carry forward the error message so the final fatal error includes it.
    ctx->lastErrorMsg = errorMsg;

    // Remove the failed location from shardLocationMap_ so that subsequent
    // probe batches (and other concurrent RPCs) will not be routed to this
    // unreachable server.  Also evict the cached stub for the same reason.
    if (!ctx->lastTarget.empty()) {
      std::lock_guard<std::mutex> lock(routeMutex_);
      auto locIt = shardLocationMap_.find(ctx->shardId);
      if (locIt != shardLocationMap_.end()) {
        auto& locs = locIt->second;
        locs.erase(
            std::remove_if(
                locs.begin(),
                locs.end(),
                [&](const ShardServerLocation& loc) {
                  return (loc.host + ":" + std::to_string(loc.port)) ==
                      ctx->lastTarget;
                }),
            locs.end());
      }
      stubCache_.erase(ctx->lastTarget);
    }

    // Mark at executor level so other SLJ instances also skip this address.
    if (!ctx->lastTarget.empty()) {
      VeloxShardManager::getInstance()->markAddressDead(ctx->lastTarget);
    }

    // Mirrors Spark fetchRemoteBatch: immediately try next replica (no backoff).
    // retryRemoteLookup handles failover → refresh → fatal-if-exhausted.
    retryRemoteLookup(std::move(owned));
    return;
  }

  auto probeRows = std::move(ctx->probeRows);

  if (!ctx->response.output_compact().empty()) {
    const auto& outputBytes = ctx->response.output_compact();
    auto serverBuildOutput = deserializeCompactRowBatch(
        std::string_view(outputBytes.data(), outputBytes.size()),
        buildOutputType_,
        pool());
    std::vector<int32_t> serverInputHits(
        ctx->response.input_hits().begin(),
        ctx->response.input_hits().end());

    if (!ctx->dedupMap.empty()) {
      // Expand dedup'd result back to original probe row indices.
      // Use separate output variables to avoid aliasing (input refs
      // and output refs must not be the same object).
      RowVectorPtr expandedBuild;
      std::vector<int32_t> expandedHits;
      expandDedupResult(
          serverBuildOutput, serverInputHits, ctx->dedupMap, pool(),
          expandedBuild, expandedHits);
      serverBuildOutput = std::move(expandedBuild);
      serverInputHits = std::move(expandedHits);
    }

    auto completed = std::make_unique<CompletedLookup>();
    completed->shardId = ctx->shardId;
    completed->probeRows = std::move(probeRows);
    auto outputRows = static_cast<int64_t>(serverBuildOutput->size());
    completed->buildOutput = std::move(serverBuildOutput);
    completed->inputHits = std::move(serverInputHits);
    computeTotalOutputRows(*completed);
    {
      std::lock_guard<std::mutex> lock(completedMutex_);
      numInFlight_--;
      completedQueue_.push(std::move(completed));
    }
    ++rpcCompletedCount_;
    if (debugEnabled_) {
      totalBuildOutputRows_ += outputRows;
    }
    if (debugEnabled_ && rpcCompletedCount_ % 50 == 0) {
      LOG(INFO) << "[SLJ-dbg] rpcDone#" << rpcCompletedCount_
                << " shard=" << ctx->shardId
                << " outputRows=" << outputRows
                << " inFlight=" << numInFlight_.load(std::memory_order_relaxed);
    }
    completedCv_.notify_all();
  } else {
    // Empty response.
    auto completed = std::make_unique<CompletedLookup>();
    completed->shardId = ctx->shardId;
    completed->probeRows = std::move(probeRows);
    completed->buildOutput = BaseVector::create<RowVector>(
        buildOutputType_, 0, pool());
    computeTotalOutputRows(*completed);
    {
      std::lock_guard<std::mutex> lock(completedMutex_);
      numInFlight_--;
      completedQueue_.push(std::move(completed));
    }
    ++rpcCompletedCount_;
    if (debugEnabled_ && rpcCompletedCount_ % 50 == 0) {
      LOG(INFO) << "[SLJ-dbg] rpcDone#" << rpcCompletedCount_
                << " shard=" << ctx->shardId << " (empty)"
                << " inFlight=" << numInFlight_.load(std::memory_order_relaxed);
    }
    completedCv_.notify_all();
  }
}

void ShardLookupJoin::computeTotalOutputRows(CompletedLookup& lookup) const {
  auto buildHits = lookup.buildOutput->size();
  if (joinType_ == core::JoinType::kInner) {
    lookup.totalOutputRows = buildHits;
    return;
  }
  // Left join: total = matched rows + unmatched probe rows (null-filled).
  auto numProbeRows = static_cast<vector_size_t>(lookup.probeRows.size());
  std::vector<bool> matched(numProbeRows, false);
  for (auto pi : lookup.inputHits) {
    if (pi >= 0 && pi < numProbeRows) {
      matched[pi] = true;
    }
  }
  lookup.unmatchedProbeRows.reserve(numProbeRows);
  for (vector_size_t pi = 0; pi < numProbeRows; ++pi) {
    if (!matched[pi]) {
      lookup.unmatchedProbeRows.push_back(pi);
    }
  }
  lookup.totalOutputRows = buildHits +
      static_cast<vector_size_t>(lookup.unmatchedProbeRows.size());
}

void ShardLookupJoin::onRpcComplete(
    std::unique_ptr<CompletedLookup> result) {
  {
    std::lock_guard<std::mutex> lock(completedMutex_);
    // NOTE: numInFlight_ is NOT decremented here.  This method is called
    // from the local-lookup path (issueLocalLookup) which never increments
    // numInFlight_.  The remote path (handleRpcCompletion) decrements
    // numInFlight_ in its own lock-protected section before enqueueing.
    completedQueue_.push(std::move(result));
  }
  completedCv_.notify_all();
}

// ---------------------------------------------------------------------------
// Output projection
// ---------------------------------------------------------------------------
void ShardLookupJoin::initOutputProjections() {
  for (auto i = 0; i < probeType_->size(); ++i) {
    const auto& name = probeType_->nameOf(i);
    auto outputIdx = outputType_->getChildIdxIfExists(name);
    if (outputIdx.has_value()) {
      probeOutputProjections_.emplace_back(i, outputIdx.value());
    }
  }
  for (auto i = 0; i < buildOutputType_->size(); ++i) {
    const auto& name = buildOutputType_->nameOf(i);
    auto outputIdx = outputType_->getChildIdxIfExists(name);
    if (outputIdx.has_value()) {
      buildOutputProjections_.emplace_back(i, outputIdx.value());
    }
  }
}

// ---------------------------------------------------------------------------
// Output production helpers
// ---------------------------------------------------------------------------

/// Prepare the reusable output_ RowVector for numOutputRows.
static RowVectorPtr prepareOutputVector(
    RowVectorPtr& output,
    const RowTypePtr& outputType,
    vector_size_t numOutputRows,
    memory::MemoryPool* memPool) {
  if (output == nullptr) {
    output = BaseVector::create<RowVector>(outputType, numOutputRows, memPool);
  } else {
    VectorPtr tmp = std::move(output);
    BaseVector::prepareForReuse(tmp, numOutputRows);
    output = std::static_pointer_cast<RowVector>(tmp);
  }
  return output;
}


/// Optimization 2/3: gather a probe-side column for output by wrapping each
/// contiguous run of same-batch rows in a DictionaryVector (zero-copy), then
/// concatenating the segments.  When all rows come from a single input batch
/// (the common case after a recent flush), this produces exactly one
/// DictionaryVector — no physical copy at all.
///
/// The base column is pinned alive by inputBatches_[batchIndex].batch
/// (refCount > 0 for as long as any BufferedRow references it), so the
/// dictionary wrap is safe to outlive this gather call.
VectorPtr ShardLookupJoin::gatherProbeColumnAsDictionary(
    column_index_t inputChannel,
    const TypePtr& outputType,
    const std::vector<const BufferedRow*>& probeRowPerOut,
    vector_size_t numOutputRows) {
  if (numOutputRows == 0) {
    return BaseVector::create(outputType, 0, pool());
  }

  // Single-batch fast path: scan once, if all rows share the same batchIndex,
  // wrap directly.
  uint32_t firstBatch = probeRowPerOut[0]->batchIndex;
  bool singleBatch = true;
  for (vector_size_t i = 1; i < numOutputRows; ++i) {
    if (probeRowPerOut[i]->batchIndex != firstBatch) {
      singleBatch = false;
      break;
    }
  }

  if (singleBatch) {
    auto indexBuf = AlignedBuffer::allocate<vector_size_t>(
        numOutputRows, pool());
    auto* rawIdx = indexBuf->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numOutputRows; ++i) {
      rawIdx[i] = probeRowPerOut[i]->rowIndex;
    }
    const auto& base =
        inputBatches_.at(firstBatch).batch->childAt(inputChannel);
    return BaseVector::wrapInDictionary(
        /*nulls=*/nullptr, std::move(indexBuf), numOutputRows, base);
  }

  // Multi-batch: build per-segment Dictionary vectors and concat.
  //
  // INVARIANT: segIndices is always populated for [0, i] before the segment-
  // boundary check at row i+1.  We pre-fill all rowIndex values up front so
  // that flushSegment(segEnd) sees a complete [segStart, segEnd) range —
  // earlier versions push_back'd inside the loop AFTER calling flushSegment,
  // which left the boundary row missing from the previous segment and caused
  // RPC payloads to be silently truncated.
  std::vector<vector_size_t> segIndices(numOutputRows);
  for (vector_size_t i = 0; i < numOutputRows; ++i) {
    segIndices[i] = probeRowPerOut[i]->rowIndex;
  }

  std::vector<VectorPtr> segments;
  segments.reserve(4);
  vector_size_t segStart = 0;
  uint32_t segBatch = probeRowPerOut[0]->batchIndex;

  auto flushSegment = [&](vector_size_t segEnd) {
    auto segLen = segEnd - segStart;
    if (segLen == 0) {
      return;
    }
    auto indexBuf =
        AlignedBuffer::allocate<vector_size_t>(segLen, pool());
    std::memcpy(
        indexBuf->asMutable<vector_size_t>(),
        segIndices.data() + segStart,
        sizeof(vector_size_t) * segLen);
    const auto& base =
        inputBatches_.at(segBatch).batch->childAt(inputChannel);
    segments.push_back(BaseVector::wrapInDictionary(
        /*nulls=*/nullptr, std::move(indexBuf), segLen, base));
  };

  for (vector_size_t i = 1; i < numOutputRows; ++i) {
    auto curBatch = probeRowPerOut[i]->batchIndex;
    if (curBatch != segBatch) {
      flushSegment(/*segEnd=*/i);
      segStart = i;
      segBatch = curBatch;
    }
  }
  flushSegment(/*segEnd=*/numOutputRows);

  if (segments.size() == 1) {
    return std::move(segments[0]);
  }

  // Concatenate segments into a single flat vector.  BaseVector::create +
  // copyRanges per source minimizes virtual dispatch (one flatten per segment
  // instead of per row).
  auto result = BaseVector::create(outputType, numOutputRows, pool());
  vector_size_t offset = 0;
  for (auto& seg : segments) {
    auto segLen = seg->size();
    std::vector<BaseVector::CopyRange> ranges{
        {/*sourceIndex=*/0, /*targetIndex=*/offset, /*count=*/segLen}};
    result->copyRanges(seg.get(), ranges);
    offset += segLen;
  }
  return result;
}

/// Optimization 3: gather a probe-key column for the per-shard RPC by
/// wrapping contiguous same-batch runs as DictionaryVector.  Zero-copy in
/// the common single-batch case.  Multi-batch falls back to a single
/// flat-vector materialization with copyRanges (one virtual dispatch per
/// segment, not per row).
VectorPtr ShardLookupJoin::gatherProbeKeyColumnAsDictionary(
    column_index_t inputChannel,
    const TypePtr& keyType,
    const std::vector<BufferedRow>& rows) {
  auto numRows = static_cast<vector_size_t>(rows.size());
  if (numRows == 0) {
    return BaseVector::create(keyType, 0, pool());
  }

  uint32_t firstBatch = rows[0].batchIndex;
  bool singleBatch = true;
  for (vector_size_t i = 1; i < numRows; ++i) {
    if (rows[i].batchIndex != firstBatch) {
      singleBatch = false;
      break;
    }
  }

  if (singleBatch) {
    auto indexBuf =
        AlignedBuffer::allocate<vector_size_t>(numRows, pool());
    auto* rawIdx = indexBuf->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numRows; ++i) {
      rawIdx[i] = rows[i].rowIndex;
    }
    const auto& base =
        inputBatches_.at(firstBatch).batch->childAt(inputChannel);
    return BaseVector::wrapInDictionary(
        /*nulls=*/nullptr, std::move(indexBuf), numRows, base);
  }

  // Multi-batch: build segments and concat into a flat result.
  //
  // INVARIANT: indices is fully populated up front so flushSegment(segEnd)
  // always sees a complete [segStart, segEnd) range.  An earlier
  // implementation interleaved push_back with the boundary check, leaving
  // the boundary row missing from the previous segment — that silently
  // truncated probe-key RPC payloads (server saw fewer rows than the
  // shard buffer actually contained, then inputHits indices on the client
  // side referenced the wrong probe rows, leading to mis-routed output and
  // pipeline stalls).
  std::vector<vector_size_t> indices(numRows);
  for (vector_size_t i = 0; i < numRows; ++i) {
    indices[i] = rows[i].rowIndex;
  }

  std::vector<VectorPtr> segments;
  vector_size_t segStart = 0;
  uint32_t segBatch = rows[0].batchIndex;

  auto flushSegment = [&](vector_size_t segEnd) {
    auto segLen = segEnd - segStart;
    if (segLen == 0) {
      return;
    }
    auto indexBuf =
        AlignedBuffer::allocate<vector_size_t>(segLen, pool());
    std::memcpy(
        indexBuf->asMutable<vector_size_t>(),
        indices.data() + segStart,
        sizeof(vector_size_t) * segLen);
    const auto& base =
        inputBatches_.at(segBatch).batch->childAt(inputChannel);
    segments.push_back(BaseVector::wrapInDictionary(
        /*nulls=*/nullptr, std::move(indexBuf), segLen, base));
  };

  for (vector_size_t i = 1; i < numRows; ++i) {
    auto curBatch = rows[i].batchIndex;
    if (curBatch != segBatch) {
      flushSegment(/*segEnd=*/i);
      segStart = i;
      segBatch = curBatch;
    }
  }
  flushSegment(/*segEnd=*/numRows);

  if (segments.size() == 1) {
    return std::move(segments[0]);
  }

  auto result = BaseVector::create(keyType, numRows, pool());
  vector_size_t offset = 0;
  for (auto& seg : segments) {
    auto segLen = seg->size();
    std::vector<BaseVector::CopyRange> ranges{
        {/*sourceIndex=*/0, /*targetIndex=*/offset, /*count=*/segLen}};
    result->copyRanges(seg.get(), ranges);
    offset += segLen;
  }
  return result;
}

/// Decrement refCounts for all probe-row references and erase any
/// inputBatches_ entry that drops to zero.  Driver-thread only.
void ShardLookupJoin::releaseProbeRefs(
    const std::vector<BufferedRow>& rows) {
  size_t erasedCount = 0;
  for (const auto& br : rows) {
    auto it = inputBatches_.find(br.batchIndex);
    if (it == inputBatches_.end()) {
      continue;
    }
    --totalRetainedRefCount_;
    if (--it->second.refCount <= 0) {
      inputBatches_.erase(it);
      ++erasedCount;
    }
  }
  if (debugEnabled_ && erasedCount > 0 && inputBatches_.size() >= 50) {
    static std::atomic<uint64_t> releaseRefsLogCount{0};
    auto cnt = ++releaseRefsLogCount;
    if (cnt <= 5 || cnt % 100 == 0) {
      LOG(INFO) << "[SLJ-dbg] releaseRefs#" << cnt
                << " erased=" << erasedCount
                << " retained=" << inputBatches_.size()
                << " refCount=" << totalRetainedRefCount_;
    }
  }
}

// ---------------------------------------------------------------------------
// Output production — Inner Join
// ---------------------------------------------------------------------------
RowVectorPtr ShardLookupJoin::produceOutputForInnerJoin(
    const CompletedLookup& lookup) {
  auto totalHits =
      static_cast<vector_size_t>(lookup.buildOutput->size());
  if (totalHits == 0) {
    currentLookup_ = nullptr;
    return nullptr;
  }

  auto numOutputRows = std::min<vector_size_t>(
      totalHits - currentLookupOutputRow_, outputBatchSize_);

  prepareOutputVector(output_, outputType_, numOutputRows, pool());

  // Optimization 2: probe-side gather via DictionaryVector.  Build a per-out
  // pointer table once, then let gatherProbeColumnAsDictionary wrap the
  // contiguous same-batch runs (typically one wrap covers the whole slice
  // since per-shard buffering keeps probe rows from the same batch grouped).
  std::vector<const BufferedRow*> probeRowPerOut(numOutputRows, nullptr);
  for (vector_size_t i = 0; i < numOutputRows; ++i) {
    auto hitIdx = currentLookupOutputRow_ + i;
    auto probeRowIdx = lookup.inputHits[hitIdx];
    probeRowPerOut[i] = &lookup.probeRows[probeRowIdx];
  }
  for (const auto& proj : probeOutputProjections_) {
    output_->childAt(proj.outputChannel) = gatherProbeColumnAsDictionary(
        proj.inputChannel,
        probeType_->childAt(proj.inputChannel),
        probeRowPerOut,
        numOutputRows);
  }

  // Build side: slice from buildOutput.
  for (const auto& proj : buildOutputProjections_) {
    if (currentLookupOutputRow_ == 0 && numOutputRows == totalHits) {
      output_->childAt(proj.outputChannel) =
          lookup.buildOutput->childAt(proj.inputChannel);
    } else {
      output_->childAt(proj.outputChannel) =
          lookup.buildOutput->childAt(proj.inputChannel)
              ->slice(currentLookupOutputRow_, numOutputRows);
    }
  }

  currentLookupOutputRow_ += numOutputRows;
  return output_;
}

// ---------------------------------------------------------------------------
// Output production — Left Join
// ---------------------------------------------------------------------------
RowVectorPtr ShardLookupJoin::produceOutputForLeftJoin(
    CompletedLookup& lookup) {
  auto totalOutput = lookup.totalOutputRows;
  if (totalOutput == 0) {
    currentLookup_ = nullptr;
    return nullptr;
  }

  auto numBuildHits =
      static_cast<vector_size_t>(lookup.buildOutput->size());
  auto numProbeRows =
      static_cast<vector_size_t>(lookup.probeRows.size());

  auto numOutputRows = std::min<vector_size_t>(
      totalOutput - currentLookupOutputRow_, outputBatchSize_);

  prepareOutputVector(output_, outputType_, numOutputRows, pool());
  (void)numProbeRows;

  // Build a per-output-row "plan": for each output position [0, numOutputRows)
  // record (probeBufferedRow, optional buildSourceRowIdx).  We need a single
  // pass to figure out the boundary between phase-1 (matched) and phase-2
  // (unmatched), which the caller's currentLookupOutputRow_ may bisect.
  std::vector<const BufferedRow*> probeRowPerOut(numOutputRows, nullptr);
  std::vector<vector_size_t> buildSrcPerOut(
      numOutputRows, /*sentinel=*/-1);
  vector_size_t outIdx = 0;

  // Phase 1: rows [currentLookupOutputRow_, numBuildHits) → matched build.
  for (vector_size_t globalIdx = currentLookupOutputRow_;
       globalIdx < numBuildHits && outIdx < numOutputRows;
       ++globalIdx, ++outIdx) {
    auto probeRowIdx = lookup.inputHits[globalIdx];
    probeRowPerOut[outIdx] = &lookup.probeRows[probeRowIdx];
    buildSrcPerOut[outIdx] = globalIdx;
  }

  // Phase 2: pre-computed unmatchedProbeRows by index — no per-call hash-set
  // rebuild, O(1) start-position seek.
  if (outIdx < numOutputRows) {
    auto numUnmatched =
        static_cast<vector_size_t>(lookup.unmatchedProbeRows.size());
    vector_size_t unmatchedStart = 0;
    if (currentLookupOutputRow_ + outIdx > numBuildHits) {
      unmatchedStart = (currentLookupOutputRow_ + outIdx) - numBuildHits;
    }
    for (vector_size_t u = unmatchedStart;
         u < numUnmatched && outIdx < numOutputRows;
         ++u, ++outIdx) {
      auto pi = lookup.unmatchedProbeRows[u];
      probeRowPerOut[outIdx] = &lookup.probeRows[pi];
      // buildSrcPerOut stays at -1 sentinel → null-fill.
    }
  }

  // Optimization 2: probe-side gather via DictionaryVector.  probeRowPerOut
  // already contains pointers for both phase-1 (matched) and phase-2
  // (unmatched) rows; both phases reference rows in lookup.probeRows so the
  // base columns live in inputBatches_ and Dictionary wrap is safe.
  for (const auto& proj : probeOutputProjections_) {
    output_->childAt(proj.outputChannel) = gatherProbeColumnAsDictionary(
        proj.inputChannel,
        probeType_->childAt(proj.inputChannel),
        probeRowPerOut,
        outIdx);
  }

  // Build side: fixed-size column, copy from buildOutput for matched rows
  // and setNull for unmatched.
  for (const auto& proj : buildOutputProjections_) {
    auto outputCol = BaseVector::create(
        buildOutputType_->childAt(proj.inputChannel), outIdx, pool());
    auto* sourceCol = lookup.buildOutput->childAt(proj.inputChannel).get();
    for (vector_size_t i = 0; i < outIdx; ++i) {
      auto buildSrc = buildSrcPerOut[i];
      if (buildSrc < 0) {
        outputCol->setNull(i, true);
      } else {
        outputCol->copy(sourceCol, i, buildSrc, 1);
      }
    }
    output_->childAt(proj.outputChannel) = outputCol;
  }

  output_->resize(outIdx);
  currentLookupOutputRow_ += outIdx;
  return output_;
}


// ---------------------------------------------------------------------------
// PlanNodeTranslator
// ---------------------------------------------------------------------------
std::unique_ptr<Operator> ShardLookupJoinTranslator::toOperator(
    DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto joinNode =
          std::dynamic_pointer_cast<const ShardLookupJoinNode>(node)) {
    return std::make_unique<ShardLookupJoin>(id, ctx, joinNode);
  }
  return nullptr;
}

} // namespace shard
} // namespace gluten
