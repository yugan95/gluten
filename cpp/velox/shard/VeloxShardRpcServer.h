#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "VeloxShardManager.h"
#include "proto/shard_lookup.grpc.pb.h"

namespace gluten {
namespace shard {

// ---------------------------------------------------------------------------
// Simple fixed-size thread pool for offloading lookup work from gRPC
// callback threads.  gRPC CallbackService delivers RPCs on a small set
// of internal threads; doing heavy HashTable lookups + serde there would
// block new RPC arrivals.  This pool decouples RPC dispatch from compute.
// ---------------------------------------------------------------------------
class LookupThreadPool {
 public:
  explicit LookupThreadPool(size_t numThreads);
  ~LookupThreadPool();

  // Submit a task to be executed by one of the worker threads.
  // Returns false if the pool is already stopped (task is NOT enqueued).
  bool submit(std::function<void()> task);

  // Drain pending tasks and join all worker threads.
  void shutdown();

 private:
  void workerLoop();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopped_{false};
};

// ---------------------------------------------------------------------------
// gRPC Callback service implementation – serves remote shard lookup requests
// using the async CallbackService model.  Each RPC is dispatched to a
// LookupThreadPool so that gRPC's internal event-loop threads are never
// blocked by HashTable lookup or serde work.
//
// Compared to the previous sync Service:
//   - No per-RPC thread allocation (sync server spawns 1 thread per RPC)
//   - gRPC can accept new RPCs while previous ones are being processed
//   - Throughput scales with pool size rather than MAX_POLLERS
// ---------------------------------------------------------------------------
class ShardLookupCallbackServiceImpl final
    : public proto::ShardLookupService::CallbackService {
 public:
  ShardLookupCallbackServiceImpl(
      VeloxShardManager* manager,
      facebook::velox::memory::MemoryPool* rpcPool,
      LookupThreadPool* threadPool)
      : manager_(manager), rpcPool_(rpcPool), threadPool_(threadPool) {}

  grpc::ServerUnaryReactor* Lookup(
      grpc::CallbackServerContext* context,
      const proto::LookupRequest* request,
      proto::LookupResponse* response) override;

  grpc::ServerUnaryReactor* BatchLookup(
      grpc::CallbackServerContext* context,
      const proto::BatchLookupRequest* request,
      proto::BatchLookupResponse* response) override;

 private:
  // Internal: perform lookup work and finish the reactor.
  void doLookup(
      const proto::LookupRequest* request,
      proto::LookupResponse* response,
      grpc::ServerUnaryReactor* reactor);

  void doBatchLookup(
      const proto::BatchLookupRequest* request,
      proto::BatchLookupResponse* response,
      grpc::ServerUnaryReactor* reactor);

  VeloxShardManager* manager_;
  facebook::velox::memory::MemoryPool* rpcPool_;
  LookupThreadPool* threadPool_;
};

// ---------------------------------------------------------------------------
// VeloxShardRpcServer – gRPC server that exposes shard lookup to remote
// executors.  Uses CallbackService + LookupThreadPool for high-concurrency
// async RPC processing.
//
// Owns the gRPC server lifecycle; delegates lookup requests to
// VeloxShardManager which manages the in-process HashTables.
// ---------------------------------------------------------------------------
class VeloxShardRpcServer {
 public:
  static VeloxShardRpcServer* getInstance() {
    static VeloxShardRpcServer instance;
    return &instance;
  }

  // Start the gRPC server on the given host/port.
  // Pass port=0 to let the OS assign a free port; the actual port is returned.
  int startServer(const std::string& host, int requestedPort);

  // Stop the gRPC server and join the background serving thread.
  void stopServer();

  bool isRunning() const { return running_; }
  int port() const { return port_; }
  const std::string& host() const { return host_; }

 private:
  VeloxShardRpcServer() = default;
  ~VeloxShardRpcServer() = default;
  VeloxShardRpcServer(const VeloxShardRpcServer&) = delete;
  VeloxShardRpcServer& operator=(const VeloxShardRpcServer&) = delete;

  std::atomic<bool> running_{false};
  int port_{-1};
  std::string host_;

  std::unique_ptr<grpc::Server> grpcServer_;
  std::unique_ptr<ShardLookupCallbackServiceImpl> grpcService_;
  std::unique_ptr<LookupThreadPool> threadPool_;
  std::thread grpcThread_;

  // Global thread-safe memory pool for RPC serde operations.
  // Created in startServer(), released in stopServer() after gRPC shutdown.
  std::shared_ptr<facebook::velox::memory::MemoryPool> rpcPool_;
};

} // namespace shard
} // namespace gluten
