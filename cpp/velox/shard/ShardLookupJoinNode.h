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

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/type/Type.h"

namespace gluten {
namespace shard {

using namespace facebook::velox;

/// gRPC server location for one shard replica.  Mirrors Spark's
/// ShardManager.getLocations(setId, shardId) entry.
struct ShardServerLocation {
  std::string host;
  int32_t port;

  ShardServerLocation(std::string host, int32_t port)
      : host(std::move(host)), port(port) {}
};

/// PlanNode for ShardLookupJoin – a DistributedMapJoin operator that buffers
/// probe rows per-shard and issues batched RPCs, breaking the per-input-batch
/// limitation of Velox's IndexLookupJoin.
///
/// Unlike IndexLookupJoinNode, this node does NOT go through the Connector /
/// IndexSource path.  The ShardLookupJoin Operator directly manages shard
/// routing, gRPC communication, and join output.
///
/// sources()[0] = probe (left) child.
/// The build side is served by remote shard servers and is NOT a plan child.
class ShardLookupJoinNode : public core::PlanNode {
 public:
  ShardLookupJoinNode(
      const core::PlanNodeId& id,
      core::JoinType joinType,
      std::vector<core::FieldAccessTypedExprPtr> leftKeys,
      std::vector<core::FieldAccessTypedExprPtr> rightKeys,
      core::PlanNodePtr probeChild,
      RowTypePtr buildOutputType,
      RowTypePtr outputType,
      int64_t shardSetId,
      int32_t numShards,
      std::unordered_map<int32_t, std::vector<ShardServerLocation>>
          shardLocationMap,
      int32_t maxInflightRpcs,
      int32_t maxBatchSize,
      RowTypePtr hashKeyType,
      core::TypedExprPtr filter = nullptr)
      : PlanNode(id),
        joinType_(joinType),
        leftKeys_(std::move(leftKeys)),
        rightKeys_(std::move(rightKeys)),
        probeChild_(std::move(probeChild)),
        buildOutputType_(std::move(buildOutputType)),
        outputType_(std::move(outputType)),
        shardSetId_(shardSetId),
        numShards_(numShards),
        shardLocationMap_(std::move(shardLocationMap)),
        maxInflightRpcs_(maxInflightRpcs),
        maxBatchSize_(maxBatchSize),
        hashKeyType_(std::move(hashKeyType)),
        filter_(std::move(filter)),
        sources_{probeChild_} {
    VELOX_USER_CHECK(
        !leftKeys_.empty(),
        "ShardLookupJoin requires at least one join key");
    VELOX_USER_CHECK_EQ(
        leftKeys_.size(),
        rightKeys_.size(),
        "ShardLookupJoin requires same number of join keys on both sides");
    VELOX_USER_CHECK(
        joinType_ == core::JoinType::kInner ||
            joinType_ == core::JoinType::kLeft,
        "ShardLookupJoin only supports INNER and LEFT join, got {}",
        core::joinTypeName(joinType_));
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "ShardLookupJoin";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("ShardLookupJoinNode is not serializable");
  }

  core::JoinType joinType() const {
    return joinType_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& leftKeys() const {
    return leftKeys_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& rightKeys() const {
    return rightKeys_;
  }

  const RowTypePtr& probeType() const {
    return probeChild_->outputType();
  }

  const RowTypePtr& buildOutputType() const {
    return buildOutputType_;
  }

  int64_t shardSetId() const {
    return shardSetId_;
  }

  int32_t numShards() const {
    return numShards_;
  }

  const std::unordered_map<int32_t, std::vector<ShardServerLocation>>&
  shardLocationMap() const {
    return shardLocationMap_;
  }

  int32_t maxInflightRpcs() const {
    return maxInflightRpcs_;
  }

  int32_t maxBatchSize() const {
    return maxBatchSize_;
  }

  const RowTypePtr& hashKeyType() const {
    return hashKeyType_;
  }

  /// Optional post-join filter expression (non-equi condition).
  /// nullptr when no filter is needed (pure equi-join).
  const core::TypedExprPtr& filter() const {
    return filter_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "ShardLookupJoin["
           << "type=" << core::joinTypeName(joinType_)
           << ", setId=" << shardSetId_
           << ", shards=" << numShards_
           << ", maxBatch=" << maxBatchSize_
           << ", maxInFlight=" << maxInflightRpcs_
           << "]";
  }

  const core::JoinType joinType_;
  const std::vector<core::FieldAccessTypedExprPtr> leftKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> rightKeys_;
  const core::PlanNodePtr probeChild_;
  const RowTypePtr buildOutputType_;
  const RowTypePtr outputType_;
  const int64_t shardSetId_;
  const int32_t numShards_;
  const std::unordered_map<int32_t, std::vector<ShardServerLocation>>
      shardLocationMap_;
  const int32_t maxInflightRpcs_;
  const int32_t maxBatchSize_;
  const RowTypePtr hashKeyType_;
  /// Optional post-join filter (non-equi condition). nullptr for pure equi-join.
  const core::TypedExprPtr filter_;
  // sources_ keeps a reference for PlanNode::sources().
  const std::vector<core::PlanNodePtr> sources_;
};

} // namespace shard
} // namespace gluten
