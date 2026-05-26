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
#include "VeloxShardRpcServer.h"

#include <queue>

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/type/fbhive/HiveTypeParser.h"

#include "CompactRowWireFormat.h"

namespace gluten {
namespace shard {

using namespace facebook::velox;

// ---------------------------------------------------------------------------
// LookupThreadPool implementation
// ---------------------------------------------------------------------------
LookupThreadPool::LookupThreadPool(size_t numThreads) {
  workers_.reserve(numThreads);
  for (size_t i = 0; i < numThreads; ++i) {
    workers_.emplace_back([this]() { workerLoop(); });
  }
}

LookupThreadPool::~LookupThreadPool() {
  shutdown();
}

bool LookupThreadPool::submit(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
      return false;
    }
    tasks_.push(std::move(task));
  }
  cv_.notify_one();
  return true;
}

void LookupThreadPool::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
      return;
    }
    stopped_ = true;
  }
  cv_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void LookupThreadPool::workerLoop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return stopped_ || !tasks_.empty(); });
      if (stopped_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

// ---------------------------------------------------------------------------
// ShardLookupCallbackServiceImpl – async callback gRPC service.
//
// Each RPC creates a ServerUnaryReactor immediately and offloads the actual
// lookup work to a LookupThreadPool.  The gRPC callback thread returns
// immediately, freeing the event loop to accept more RPCs.  The worker
// thread calls reactor->Finish() when done.
// ---------------------------------------------------------------------------

grpc::ServerUnaryReactor* ShardLookupCallbackServiceImpl::Lookup(
    grpc::CallbackServerContext* context,
    const proto::LookupRequest* request,
    proto::LookupResponse* response) {
  auto* reactor = context->DefaultReactor();
  if (!threadPool_->submit([this, request, response, reactor]() {
        doLookup(request, response, reactor);
      })) {
    // Pool is shutting down — finish immediately so the client gets an
    // error instead of hanging until deadline.
    reactor->Finish(grpc::Status(
        grpc::StatusCode::UNAVAILABLE, "Server is shutting down"));
  }
  return reactor;
}

grpc::ServerUnaryReactor* ShardLookupCallbackServiceImpl::BatchLookup(
    grpc::CallbackServerContext* context,
    const proto::BatchLookupRequest* request,
    proto::BatchLookupResponse* response) {
  auto* reactor = context->DefaultReactor();
  if (!threadPool_->submit([this, request, response, reactor]() {
        doBatchLookup(request, response, reactor);
      })) {
    reactor->Finish(grpc::Status(
        grpc::StatusCode::UNAVAILABLE, "Server is shutting down"));
  }
  return reactor;
}

void ShardLookupCallbackServiceImpl::doLookup(
    const proto::LookupRequest* request,
    proto::LookupResponse* response,
    grpc::ServerUnaryReactor* reactor) {
  try {
    auto setId = static_cast<int64_t>(request->set_id());
    auto shardId = static_cast<int32_t>(request->shard_id());

    auto* shardManager = VeloxShardManager::getInstance();

    auto keyType = shardManager->getKeyType(setId, shardId);
    if (!keyType) {
      reactor->Finish(grpc::Status::OK);
      return;
    }

    if (!rpcPool_) {
      reactor->Finish(grpc::Status(
          grpc::StatusCode::INTERNAL,
          "RPC memory pool not initialised"));
      return;
    }

    const std::string& probeBytes = request->probe_keys_compact();
    auto probeKeys = deserializeCompactRowBatch(
        std::string_view(probeBytes.data(), probeBytes.size()),
        keyType,
        rpcPool_);

    std::vector<int32_t> inputIndices(
        request->input_indices().begin(), request->input_indices().end());

    // Parse server-side filter pushdown fields.
    RowVectorPtr probeFilterColumns;
    std::string filterExprJson;
    RowTypePtr probeColumnsType;
    RowTypePtr filterInputType;

    if (!request->filter_expression().empty()) {
      filterExprJson = request->filter_expression();
      type::fbhive::HiveTypeParser parser;
      probeColumnsType = std::dynamic_pointer_cast<const RowType>(
          parser.parse(request->probe_columns_type()));
      filterInputType = std::dynamic_pointer_cast<const RowType>(
          parser.parse(request->filter_input_type()));
      if (!request->probe_filter_columns_compact().empty() &&
          probeColumnsType) {
        const auto& filterBytes = request->probe_filter_columns_compact();
        probeFilterColumns = deserializeCompactRowBatch(
            std::string_view(filterBytes.data(), filterBytes.size()),
            probeColumnsType,
            rpcPool_);
      }
    }

    auto result = shardManager->lookupAndSerialize(
        setId, shardId, probeKeys, inputIndices, rpcPool_,
        probeFilterColumns, filterExprJson, probeColumnsType, filterInputType);

    if (!result.outputBytes.empty()) {
      response->set_output_compact(std::move(result.outputBytes));

      for (int32_t hit : result.inputHits) {
        response->add_input_hits(hit);
      }
    }

    reactor->Finish(grpc::Status::OK);
  } catch (const std::exception& ex) {
    reactor->Finish(
        grpc::Status(grpc::StatusCode::INTERNAL, ex.what()));
  }
}

void ShardLookupCallbackServiceImpl::doBatchLookup(
    const proto::BatchLookupRequest* request,
    proto::BatchLookupResponse* response,
    grpc::ServerUnaryReactor* reactor) {
  try {
    auto setId = static_cast<int64_t>(request->set_id());
    auto* shardManager = VeloxShardManager::getInstance();

    if (!rpcPool_) {
      reactor->Finish(grpc::Status(
          grpc::StatusCode::INTERNAL,
          "RPC memory pool not initialised"));
      return;
    }

    for (const auto& entry : request->entries()) {
      auto shardId = static_cast<int32_t>(entry.shard_id());

      auto keyType = shardManager->getKeyType(setId, shardId);
      if (!keyType) {
        continue;
      }

      const std::string& probeBytes = entry.probe_keys_compact();
      if (probeBytes.empty()) {
        continue;
      }

      auto probeKeys = deserializeCompactRowBatch(
          std::string_view(probeBytes.data(), probeBytes.size()),
          keyType,
          rpcPool_);

      std::vector<int32_t> inputIndices(
          entry.input_indices().begin(), entry.input_indices().end());

      // Parse server-side filter pushdown fields.
      RowVectorPtr probeFilterColumns;
      std::string filterExprJson;
      RowTypePtr probeColumnsType;
      RowTypePtr filterInputType;

      if (!entry.filter_expression().empty()) {
        filterExprJson = entry.filter_expression();
        type::fbhive::HiveTypeParser parser;
        probeColumnsType = std::dynamic_pointer_cast<const RowType>(
            parser.parse(entry.probe_columns_type()));
        filterInputType = std::dynamic_pointer_cast<const RowType>(
            parser.parse(entry.filter_input_type()));
        if (!entry.probe_filter_columns_compact().empty() &&
            probeColumnsType) {
          const auto& filterBytes = entry.probe_filter_columns_compact();
          probeFilterColumns = deserializeCompactRowBatch(
              std::string_view(filterBytes.data(), filterBytes.size()),
              probeColumnsType,
              rpcPool_);
        }
      }

      auto result = shardManager->lookupAndSerialize(
          setId, shardId, probeKeys, inputIndices, rpcPool_,
          probeFilterColumns, filterExprJson, probeColumnsType,
          filterInputType);

      if (!result.outputBytes.empty()) {
        auto* resultEntry = response->add_results();
        resultEntry->set_shard_id(shardId);
        resultEntry->set_output_compact(std::move(result.outputBytes));

        for (int32_t hit : result.inputHits) {
          resultEntry->add_input_hits(hit);
        }
      }
    }

    reactor->Finish(grpc::Status::OK);
  } catch (const std::exception& ex) {
    reactor->Finish(
        grpc::Status(grpc::StatusCode::INTERNAL, ex.what()));
  }
}

// ---------------------------------------------------------------------------
// VeloxShardRpcServer lifecycle
// ---------------------------------------------------------------------------
int VeloxShardRpcServer::startServer(
    const std::string& host,
    int requestedPort) {
  host_ = host;

  // Create a global thread-safe leaf pool for all RPC serde operations.
  auto* shardManager = VeloxShardManager::getInstance();
  rpcPool_ = shardManager->pool()->addLeafChild(
      "rpc_lookup_pool",
      true /* threadSafe */);

  // Create lookup thread pool.  The pool size should be large enough to
  // saturate CPU on lookup-heavy workloads, but not so large that context
  // switching dominates.  32 threads matches the previous MAX_POLLERS
  // config and provides good throughput for concurrent shard lookups.
  constexpr size_t kLookupPoolSize = 32;
  threadPool_ = std::make_unique<LookupThreadPool>(kLookupPoolSize);

  grpcService_ = std::make_unique<ShardLookupCallbackServiceImpl>(
      shardManager, rpcPool_.get(), threadPool_.get());

  std::string listenAddress =
      host + ":" + std::to_string(requestedPort);

  grpc::ServerBuilder builder;
  // Default 4MB is too small for large lookup responses.
  static constexpr int kMaxMessageSize = 512 * 1024 * 1024; // 512MB
  builder.SetMaxReceiveMessageSize(kMaxMessageSize);
  builder.SetMaxSendMessageSize(kMaxMessageSize);
  int selectedPort = requestedPort;
  builder.AddListeningPort(
      listenAddress, grpc::InsecureServerCredentials(), &selectedPort);
  builder.RegisterService(grpcService_.get());

  // CallbackService uses gRPC's internal event-loop threads for RPC
  // dispatch; SyncServerOption settings have no effect and are omitted.

  grpcServer_ = builder.BuildAndStart();
  if (!grpcServer_) {
    throw std::runtime_error(
        "Failed to start gRPC VeloxShardRpcServer on " + listenAddress);
  }

  port_ = selectedPort;
  running_ = true;

  grpcThread_ = std::thread([this]() { grpcServer_->Wait(); });

  return port_;
}

void VeloxShardRpcServer::stopServer() {
  running_ = false;

  // Shutdown order matters:
  // 1. Shutdown gRPC server — stops accepting new RPCs and drains in-flight.
  // 2. Join serving thread — ensures all RPC callbacks have returned.
  // 3. Shutdown thread pool — drains pending tasks and joins workers.
  // 4. Destroy service — safe because no lookup is running.
  // 5. Release rpcPool_ — safe because no serde operation is in progress.
  if (grpcServer_) {
    grpcServer_->Shutdown();
  }
  if (grpcThread_.joinable()) {
    grpcThread_.join();
  }
  if (threadPool_) {
    threadPool_->shutdown();
    threadPool_.reset();
  }
  grpcServer_.reset();
  grpcService_.reset();
  rpcPool_.reset();
}

} // namespace shard
} // namespace gluten
