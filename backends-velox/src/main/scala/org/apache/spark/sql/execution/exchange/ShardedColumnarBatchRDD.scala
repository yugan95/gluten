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
package org.apache.spark.sql.execution.exchange

import org.apache.spark.{OneToOneDependency, Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * A simple RDD wrapper that assigns preferred host locations to each partition of a columnar
 * shuffle output. This is the columnar equivalent of Spark's ShardedRowRDD, used by
 * ColumnarShardExchangeExec to co-locate shard data with specific executors.
 */
private[spark] class ShardedColumnarBatchRDD(
    prev: RDD[ColumnarBatch],
    preferredHosts: Array[Seq[String]])
  extends RDD[ColumnarBatch](prev.sparkContext, Seq(new OneToOneDependency(prev))) {

  override protected def getPartitions: Array[Partition] = prev.partitions

  override protected def getPreferredLocations(split: Partition): Seq[String] = {
    preferredHosts(split.index % preferredHosts.length)
  }

  override def compute(split: Partition, ctx: TaskContext): Iterator[ColumnarBatch] = {
    prev.iterator(split, ctx)
  }
}
