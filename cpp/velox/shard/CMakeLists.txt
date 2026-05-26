# Shard lookup components for DistributedMapJoin native backend.
#
# Includes:
#   - VeloxShardManager: in-process hash table storage + gRPC lookup service
#   - VeloxShardRpcServer: gRPC server hosting per-executor shard lookups
#   - ShardLookupJoin: Velox Operator that buffers probe rows per-shard and
#       issues batched async RPCs (replaces the older ShardSource +
#       IndexLookupJoin path)
#   - ShardManagerJni: JNI bridge to Scala org.apache.gluten.shard.ShardManagerJni
#   - proto/shard_lookup.proto: gRPC service definition
#
# Proto-generated sources (proto/shard_lookup.pb.cc, proto/shard_lookup.grpc.pb.cc)
# are pre-generated and committed to the repository.  To regenerate after
# editing shard_lookup.proto, run from the proto/ directory:
#
#   cd proto && protoc --grpc_out=. --cpp_out=. -I . \
#     --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
#     shard_lookup.proto
#
# gRPC dependency note:
#   Shard is currently the only Gluten module that uses gRPC (v1.51.1,
#   installed at /usr/local).  Velox has an optional gRPC dependency
#   (v1.48.1) for its Spark fuzzer tests, but that is not activated in
#   Gluten builds.  If a future Velox feature (e.g. Remote Functions)
#   enables gRPC at the Velox level, the versions should be aligned to
#   avoid symbol conflicts in libvelox.so.

# ---------------------------------------------------------------------------
# All shard source files (hand-written + pre-generated proto) are compiled
# directly into libvelox.so, following Gluten's single-library model.
# ---------------------------------------------------------------------------
set(SHARD_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/VeloxShardManager.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/VeloxShardRpcServer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ShardManagerJni.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ShardLookupJoin.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/proto/shard_lookup.pb.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/proto/shard_lookup.grpc.pb.cc
)
set(VELOX_SRCS ${VELOX_SRCS} ${SHARD_SRCS} PARENT_SCOPE)

# Export include dirs for proto-generated headers and shard headers.
set(SHARD_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/proto"
  PARENT_SCOPE)
