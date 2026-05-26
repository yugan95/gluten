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
package org.apache.gluten.execution

import org.apache.spark.sql.{DataFrame, QueryTest, SparkSession}
import org.apache.spark.sql.catalyst.plans.SQLHelper
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.execution.exchange.{ColumnarShardExchangeExec, EnsureRequirements, ShardExchangeExec}
import org.apache.spark.sql.execution.joins.{DistributedMapJoinExec, DistributedMapJoinExecTransformer}
import org.apache.spark.sql.internal.SQLConf

/**
 * End-to-end correctness tests for DistributedMapJoin native offload.
 *
 * Mirrors Spark's DistributedMapJoinSuiteBase to verify that the Gluten/Velox native implementation
 * produces identical results to vanilla Spark joins. Runs in local-cluster mode with multiple
 * executors so that ShardExchange actually distributes data.
 *
 * Additionally validates plan structure:
 *   1. DistributedMapJoinExec is offloaded to DistributedMapJoinExecTransformer 2.
 *      ShardExchangeExec is offloaded to ColumnarShardExchangeExec 3. Unsupported join types fall
 *      back to Spark's DistributedMapJoinExec
 */
class VeloxDistributedMapJoinSuite extends QueryTest with SQLHelper with AdaptiveSparkPlanHelper {

  protected var _spark: SparkSession = _
  override protected def spark: SparkSession = _spark

  private def sql(sqlText: String): DataFrame = spark.sql(sqlText)

  private val ensureRequirements = EnsureRequirements()

  override def beforeAll(): Unit = {
    super.beforeAll()
    // spark.test.home is required for local-cluster mode (Spark forks worker
    // JVMs and needs to locate its own jars).  Pass it via:
    //   mvn test -DargLine="-Dspark.test.home=/path/to/spark"
    // or set the SPARK_HOME environment variable.
    val sparkHome = sys.env.getOrElse(
      "SPARK_HOME",
      sys.props.getOrElse(
        "spark.test.home",
        throw new IllegalStateException(
          "SPARK_HOME environment variable or spark.test.home system property must be set. " +
            """E.g.: mvn test -DargLine="-Dspark.test.home=/path/to/spark"""")
      )
    )
    sys.props += ("spark.test.home" -> sparkHome)

    // Native library loading: GlutenPlugin (via JniLibLoader) automatically
    // loads libvelox.so from the classpath.  Shard JNI symbols are compiled
    // into libvelox.so, so no separate library or extraLibraryPath is needed.

    val localWarehouse = new java.io.File(
      System.getProperty("java.io.tmpdir"),
      s"dmj-test-warehouse-${System.nanoTime()}").getAbsolutePath

    _spark = SparkSession
      .builder()
      .master("local-cluster[2,1,1024]")
      .appName("distmj-test")
      .config("spark.test.home", sparkHome)
      .config("spark.plugins", "org.apache.gluten.GlutenPlugin")
      .config("spark.gluten.enabled", "true")
      .config("spark.driver.memory", "1G")
      .config("spark.executor.memory", "1024m")
      .config("spark.memory.offHeap.enabled", "true")
      .config("spark.memory.offHeap.size", "1024MB")
      .config("spark.ui.enabled", "false")
      .config("spark.gluten.ui.enabled", "false")
      .config("spark.unsafe.exceptionOnMemoryLeak", "true")
      .config("spark.gluten.sql.columnar.forceShuffledHashJoin", "false")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.sort.ColumnarShuffleManager")
      .config(SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key, "-1")
      .config("spark.sql.warehouse.dir", localWarehouse)
      .getOrCreate()
    prepareTables()
  }

  override def afterAll(): Unit = {
    try {
      if (_spark != null) {
        _spark.stop()
        _spark = null
      }
    } finally {
      super.afterAll()
    }
  }

  override def beforeEach(): Unit = {
    super.beforeEach()
    System.gc()
  }

  // -------------------------------------------------------------------------
  // Test data -- mirrors Spark's DistributedMapJoinSuiteBase
  // -------------------------------------------------------------------------

  private def prepareTables(): Unit = {
    val sparkSession = spark
    import sparkSession.implicits._

    val dim: DataFrame = Seq[(Option[Int], String)](
      (Some(1), "A"),
      (Some(2), "B"),
      (Some(3), "C"),
      (Some(4), "D"),
      (Some(5), "E"),
      (Some(6), "F"),
      (Some(7), "G"),
      (None, "NULLK")).toDF("k", "v")

    val fact: DataFrame = Seq
      .tabulate(1000) {
        i =>
          val k: Option[Int] = if (i % 20 == 0) None else Some(i % 9)
          (k, s"r$i")
      }
      .toDF("k", "payload")

    dim.createOrReplaceTempView("dim")
    fact.createOrReplaceTempView("fact")

    val dim2: DataFrame = Seq[(Option[Int], String, String)](
      (Some(1), "X", "AX"),
      (Some(1), "Y", "AY"),
      (Some(2), "X", "BX"),
      (Some(3), "Z", "CZ"),
      (None, "X", "NX")).toDF("k", "cat", "val")

    val fact2: DataFrame = Seq(
      (Some(1), "X", "fx1"),
      (Some(1), "Y", "fy1"),
      (Some(1), null.asInstanceOf[String], "fnull1"),
      (Some(2), "Y", "fy2"),
      (Some(2), "X", "fx2"),
      (Some(3), "Z", "fz3"),
      (Some(4), "W", "fw4")
    ).toDF("k", "cat", "payload")

    dim2.createOrReplaceTempView("dim2")
    fact2.createOrReplaceTempView("fact2")

    val dim3: DataFrame = Seq((1, "P"), (2, "Q"), (3, "R")).toDF("k", "tag")
    dim3.createOrReplaceTempView("dim3")

    val t: DataFrame = Seq.tabulate(20)(i => (i % 5, s"s$i")).toDF("k", "s")
    t.createOrReplaceTempView("t")

    // TPC-DS style tables for multi-table join tests
    val storeSales = Seq.tabulate(200) { i =>
      (i, i % 10, i % 20, i % 30, (i * 1.5).toDouble)
    }.toDF("ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", "ss_store_sk", "ss_ext_sales_price")
    storeSales.createOrReplaceTempView("store_sales")

    val item = Seq.tabulate(10) { i =>
      val cats = Seq("Women", "Men", "Electronics", "Home", "Sports")
      (i, i * 100, s"Brand_$i", cats(i % cats.length))
    }.toDF("i_item_sk", "i_brand_id", "i_brand", "i_category")
    item.createOrReplaceTempView("item")

    val customer = Seq.tabulate(20) { i =>
      val countries = Seq("US", "CN", "UK", "JP", "DE")
      (i, countries(i % countries.length))
    }.toDF("c_customer_sk", "c_birth_country")
    customer.createOrReplaceTempView("customer")

    val dateDim = Seq.tabulate(30) { i =>
      (i, 2000 + (i % 3), 1 + (i % 12))
    }.toDF("d_date_sk", "d_year", "d_moy")
    dateDim.createOrReplaceTempView("date_dim")
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private def assertDMJPlan(plan: SparkPlan, expectBuildRight: Boolean = true): Unit = {
    val applied = ensureRequirements.apply(plan)
    val hasShard = applied.collect { case _: ShardExchangeExec => 1 }.nonEmpty
    val dmjOption = applied.collect { case j: DistributedMapJoinExec => j }.headOption
    assert(hasShard, s"Plan should contain ShardExchangeExec:\n$applied")
    assert(dmjOption.isDefined, s"Plan should contain DistributedMapJoinExec:\n$applied")
    if (expectBuildRight) {
      assert(dmjOption.get.buildSide == org.apache.spark.sql.catalyst.optimizer.BuildRight)
    } else {
      assert(dmjOption.get.buildSide == org.apache.spark.sql.catalyst.optimizer.BuildLeft)
    }
  }

  private def assertDMJPlanCount(plan: SparkPlan, expectedCount: Int): Unit = {
    val applied = ensureRequirements.apply(plan)
    val dmjCount = applied.collect { case _: DistributedMapJoinExec => 1 }.sum
    val shardCount = applied.collect { case _: ShardExchangeExec => 1 }.sum
    assert(
      dmjCount == expectedCount,
      s"Expected $expectedCount DMJ, but found $dmjCount:\n$applied")
    assert(
      shardCount == expectedCount,
      s"Expected $expectedCount ShardExchange, but found $shardCount:\n$applied")
  }

  /** Verify DMJ plan structure and data correctness against a baseline query. */
  private def checkDMJEquals(
      dmjSql: String,
      baselineSql: String,
      expectBuildRight: Boolean = true): Unit = {
    val dmjDf = sql(dmjSql)
    assertDMJPlan(dmjDf.queryExecution.sparkPlan, expectBuildRight)
    checkAnswer(dmjDf, sql(baselineSql))
  }

  /**
   * Assert that executedPlan contains DistributedMapJoinExecTransformer (native offload). Triggers
   * query execution first so that AQE finalizes the plan.
   */
  private def assertNativeOffload(df: DataFrame): Unit = {
    df.collect() // Force execution so AQE finalizes the plan
    val executedPlan = df.queryExecution.executedPlan
    val dmjTransformers = collect(executedPlan) { case d: DistributedMapJoinExecTransformer => d }
    assert(
      dmjTransformers.nonEmpty,
      s"executedPlan should contain DistributedMapJoinExecTransformer:\n$executedPlan")

    val columnarShardExchanges = collect(executedPlan) { case s: ColumnarShardExchangeExec => s }
    assert(
      columnarShardExchanges.nonEmpty,
      s"executedPlan should contain ColumnarShardExchangeExec:\n$executedPlan")
  }

  /**
   * Assert that executedPlan does NOT contain DistributedMapJoinExecTransformer (fallback).
   * Triggers query execution first so that AQE finalizes the plan.
   */
  private def assertFallback(df: DataFrame): Unit = {
    df.collect() // Force execution so AQE finalizes the plan
    val executedPlan = df.queryExecution.executedPlan
    val dmjTransformers = collect(executedPlan) { case d: DistributedMapJoinExecTransformer => d }
    assert(
      dmjTransformers.isEmpty,
      s"executedPlan should NOT contain DistributedMapJoinExecTransformer " +
        s"(should fallback):\n$executedPlan")
  }

  // =========================================================================
  // Data correctness tests (mirrors Spark's DistributedMapJoinSuiteBase)
  // =========================================================================

  test("inner join build right via DMJ hint") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
        |  f.k, d.v, f.payload
        |FROM fact f JOIN dim d ON f.k = d.k
        |""".stripMargin
    val baselineSql = "SELECT f.k, d.v, f.payload FROM fact f JOIN dim d ON f.k = d.k"
    checkDMJEquals(dmjSql, baselineSql, expectBuildRight = true)
    assertNativeOffload(sql(dmjSql))
  }

  test("inner join on two keys (build right)") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
        |  f.k, f.cat, d.val, f.payload
        |FROM fact2 f JOIN dim2 d ON f.k = d.k AND f.cat = d.cat
        |""".stripMargin
    val baselineSql =
      """SELECT f.k, f.cat, d.val, f.payload FROM fact2 f JOIN dim2 d
        |ON f.k = d.k AND f.cat = d.cat""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
    assertNativeOffload(dmjDf)
  }

  test("two-key inner join with nulls on probe/build") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d) */
        |  f.k, f.cat, d.val
        |FROM fact2 f JOIN dim2 d ON f.k = d.k AND f.cat = d.cat
        |""".stripMargin
    val baselineSql =
      """SELECT f.k, f.cat, d.val FROM fact2 f JOIN dim2 d
        |ON f.k = d.k AND f.cat = d.cat""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
  }

  test("left outer join build right via DMJ hint") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=4, replica_count=1)) */
        |  f.k, d.v
        |FROM fact f LEFT OUTER JOIN dim d ON f.k = d.k
        |""".stripMargin
    val baselineSql = "SELECT f.k, d.v FROM fact f LEFT OUTER JOIN dim d ON f.k = d.k"
    checkDMJEquals(dmjSql, baselineSql, expectBuildRight = true)
    assertNativeOffload(sql(dmjSql))
  }

  test("left outer join with null keys on probe, null-supplied rows preserved") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=4, replica_count=1)) */
        |  f.k, d.v
        |FROM fact f LEFT OUTER JOIN dim d ON f.k = d.k
        |ORDER BY f.k NULLS FIRST, d.v NULLS FIRST
        |""".stripMargin
    val baselineSql =
      """SELECT f.k, d.v FROM fact f LEFT OUTER JOIN dim d ON f.k = d.k
        |ORDER BY f.k NULLS FIRST, d.v NULLS FIRST""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
  }

  test("right outer join build left via DMJ hint - unsupported, throws") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(dleft(shard_count=2, replica_count=1)) */
        |  dleft.k, dleft.v, f.payload
        |FROM dim dleft RIGHT OUTER JOIN fact f ON dleft.k = f.k
        |""".stripMargin
    val e = intercept[Exception] {
      sql(dmjSql).collect()
    }
    assert(
      e.getMessage.contains("does not support join type") ||
        e.getCause != null && e.getCause.getMessage.contains("does not support join type"),
      s"Expected UnsupportedOperationException about join type, but got: ${e.getMessage}"
    )
  }

  test("left semi join build right via DMJ hint - unsupported, throws") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
        |  f.k
        |FROM fact f LEFT SEMI JOIN dim d ON f.k = d.k
        |""".stripMargin
    val e = intercept[Exception] {
      sql(dmjSql).collect()
    }
    assert(
      e.getMessage.contains("does not support join type") ||
        e.getCause != null && e.getCause.getMessage.contains("does not support join type"),
      s"Expected UnsupportedOperationException about join type, but got: ${e.getMessage}"
    )
  }

  test("left anti join build right via DMJ hint - unsupported, throws") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
        |  f.k
        |FROM fact f LEFT ANTI JOIN dim d ON f.k = d.k
        |""".stripMargin
    val e = intercept[Exception] {
      sql(dmjSql).collect()
    }
    assert(
      e.getMessage.contains("does not support join type") ||
        e.getCause != null && e.getCause.getMessage.contains("does not support join type"),
      s"Expected UnsupportedOperationException about join type, but got: ${e.getMessage}"
    )
  }

  test("multi-table join chain (two DMJs in one query)") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d), DISTMAPJOIN(d3) */
        |  f.k, d.v, d3.tag, f.payload
        |FROM fact f
        |JOIN dim d ON f.k = d.k
        |JOIN dim3 d3 ON f.k = d3.k
        |""".stripMargin
    val baselineSql =
      """SELECT f.k, d.v, d3.tag, f.payload FROM fact f
        |JOIN dim d ON f.k = d.k
        |JOIN dim3 d3 ON f.k = d3.k""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 2)
    checkAnswer(dmjDf, sql(baselineSql))
  }

  test("self join with build left via hint") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(a(shard_count=2, replica_count=1)) */
        |  a.k, a.s, b.s
        |FROM t a JOIN t b ON a.k = b.k
        |""".stripMargin
    val baselineSql = "SELECT a.k, a.s, b.s FROM t a JOIN t b ON a.k = b.k"
    val dmjDf = sql(dmjSql)
    val applied = ensureRequirements.apply(dmjDf.queryExecution.sparkPlan)
    val dmjNode = applied.collect { case j: DistributedMapJoinExec => j }.head
    assert(
      dmjNode.buildSide == org.apache.spark.sql.catalyst.optimizer.BuildLeft,
      s"Expected build left, but got ${dmjNode.buildSide}")
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
  }

  test("subquery as build side with hint on alias") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(ds) */
        |  f.k, ds.v
        |FROM fact f
        |JOIN (SELECT k, v FROM dim WHERE k IS NOT NULL) ds ON f.k = ds.k
        |""".stripMargin
    val baselineSql =
      """SELECT f.k, ds.v FROM fact f
        |JOIN (SELECT k, v FROM dim WHERE k IS NOT NULL) ds ON f.k = ds.k""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
    assertNativeOffload(dmjDf)
  }

  test("heterogeneous two-key join (int + string)") {
    val sparkSession = spark
    import sparkSession.implicits._
    val factsHetero =
      Seq((1, "X", "p1"), (2, "Y", "p2"), (3, "Z", "p3"), (3, "W", "p4")).toDF(
        "k",
        "cat",
        "payload")
    factsHetero.createOrReplaceTempView("facts_hetero")
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d) */
        |  f.k, f.cat, d.val
        |FROM facts_hetero f JOIN dim2 d ON f.k = d.k AND f.cat = d.cat
        |""".stripMargin
    val baselineSql =
      """SELECT f.k, f.cat, d.val FROM facts_hetero f JOIN dim2 d
        |ON f.k = d.k AND f.cat = d.cat""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
  }

  // =========================================================================
  // Plan structure / offload validation tests
  // =========================================================================

  test("plan: inner join offloaded to DistributedMapJoinExecTransformer") {
    val df = sql("""SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
                   |  f.k, d.v, f.payload
                   |FROM fact f JOIN dim d ON f.k = d.k
                   |""".stripMargin)
    assertNativeOffload(df)
  }

  test("plan: ColumnarShardExchangeExec preserves numShards and replicaCount") {
    val df = sql("""SELECT /*+ DISTMAPJOIN(d(shard_count=7, replica_count=2)) */
                   |  f.k, d.v
                   |FROM fact f JOIN dim d ON f.k = d.k
                   |""".stripMargin)
    df.collect() // Force execution so AQE finalizes the plan
    val executedPlan = df.queryExecution.executedPlan
    val columnarShardExchanges = collect(executedPlan) { case s: ColumnarShardExchangeExec => s }
    assert(columnarShardExchanges.nonEmpty, "Should have ColumnarShardExchangeExec")
    val exchange = columnarShardExchanges.head
    assert(exchange.numShards == 7, s"numShards should be 7, got ${exchange.numShards}")
    assert(exchange.replicaCount == 2, s"replicaCount should be 2, got ${exchange.replicaCount}")
  }

  test("plan: without DISTMAPJOIN hint, no DMJ plan is generated") {
    val df = sql("SELECT f.k, d.v, f.payload FROM fact f JOIN dim d ON f.k = d.k")
    val sparkPlan = df.queryExecution.sparkPlan
    val dmjExecs = sparkPlan.collect { case d: DistributedMapJoinExec => d }
    assert(
      dmjExecs.isEmpty,
      s"Without DISTMAPJOIN hint, sparkPlan should NOT contain DistributedMapJoinExec:" +
        s"\n$sparkPlan")
  }

  test("plan: right outer join throws UnsupportedOperationException") {
    val df = sql("""SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
                   |  d.k, d.v, f.payload
                   |FROM dim d RIGHT OUTER JOIN fact f ON d.k = f.k
                   |""".stripMargin)
    val e = intercept[Exception](df.collect())
    assert(
      e.getMessage.contains("does not support join type") ||
        e.getCause != null && e.getCause.getMessage.contains("does not support join type"))
  }

  test("plan: left semi join throws UnsupportedOperationException") {
    val df = sql("""SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
                   |  f.k
                   |FROM fact f LEFT SEMI JOIN dim d ON f.k = d.k
                   |""".stripMargin)
    val e = intercept[Exception](df.collect())
    assert(
      e.getMessage.contains("does not support join type") ||
        e.getCause != null && e.getCause.getMessage.contains("does not support join type"))
  }

  test("plan: left anti join throws UnsupportedOperationException") {
    val df = sql("""SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
                   |  f.k
                   |FROM fact f LEFT ANTI JOIN dim d ON f.k = d.k
                   |""".stripMargin)
    val e = intercept[Exception](df.collect())
    assert(
      e.getMessage.contains("does not support join type") ||
        e.getCause != null && e.getCause.getMessage.contains("does not support join type"))
  }

  test("plan: multi-table join chain produces 2 ColumnarShardExchangeExec") {
    val df = sql("""SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)),
                   |         DISTMAPJOIN(d3(shard_count=3, replica_count=1)) */
                   |  f.k, d.v, d3.tag, f.payload
                   |FROM fact f
                   |JOIN dim d ON f.k = d.k
                   |JOIN dim3 d3 ON f.k = d3.k
                   |""".stripMargin)
    df.collect() // Force execution so AQE finalizes the plan
    val executedPlan = df.queryExecution.executedPlan
    val dmjTransformers = collect(executedPlan) { case d: DistributedMapJoinExecTransformer => d }
    assert(
      dmjTransformers.size == 2,
      s"Multi-table join chain should produce 2 DistributedMapJoinExecTransformer, " +
        s"got ${dmjTransformers.size}:\n$executedPlan")
    val columnarShardExchanges = collect(executedPlan) { case s: ColumnarShardExchangeExec => s }
    assert(
      columnarShardExchanges.size == 2,
      s"Multi-table join chain should produce 2 ColumnarShardExchangeExec, " +
        s"got ${columnarShardExchanges.size}:\n$executedPlan"
    )
  }

  // =========================================================================
  // Hash consistency tests — verify Spark-side and C++ side hash agreement
  // =========================================================================

  test("pure string key inner join via DMJ (hash consistency)") {
    val sparkSession = spark
    import sparkSession.implicits._

    val stringDim = Seq(
      ("alice", "engineer"),
      ("bob", "designer"),
      ("charlie", "manager"),
      ("dave", "analyst"),
      ("eve", "scientist")
    ).toDF("name", "role")
    stringDim.createOrReplaceTempView("string_dim")

    val stringFact = Seq.tabulate(200) { i =>
      val names = Seq("alice", "bob", "charlie", "dave", "eve", "unknown")
      (names(i % names.length), s"event_$i")
    }.toDF("name", "event")
    stringFact.createOrReplaceTempView("string_fact")

    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
        |  f.name, d.role, f.event
        |FROM string_fact f JOIN string_dim d ON f.name = d.name
        |""".stripMargin
    val baselineSql =
      """SELECT f.name, d.role, f.event
        |FROM string_fact f JOIN string_dim d ON f.name = d.name""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
    assertNativeOffload(dmjDf)
  }

  test("pure string key left outer join via DMJ (hash consistency)") {
    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=4, replica_count=1)) */
        |  f.name, d.role
        |FROM string_fact f LEFT OUTER JOIN string_dim d ON f.name = d.name
        |ORDER BY f.name NULLS FIRST, d.role NULLS FIRST
        |""".stripMargin
    val baselineSql =
      """SELECT f.name, d.role
        |FROM string_fact f LEFT OUTER JOIN string_dim d ON f.name = d.name
        |ORDER BY f.name NULLS FIRST, d.role NULLS FIRST""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
    assertNativeOffload(dmjDf)
  }

  test("multi-string-key DMJ with varied shard counts (hash routing consistency)") {
    val sparkSession = spark
    import sparkSession.implicits._

    val multiKeyDim = Seq(
      ("us", "ca", "west"),
      ("us", "ny", "east"),
      ("uk", "london", "eu"),
      ("jp", "tokyo", "asia"),
      ("de", "berlin", "eu")
    ).toDF("country", "city", "region")
    multiKeyDim.createOrReplaceTempView("geo_dim")

    val multiKeyFact = Seq(
      ("us", "ca", "order1"),
      ("us", "ny", "order2"),
      ("uk", "london", "order3"),
      ("jp", "tokyo", "order4"),
      ("de", "berlin", "order5"),
      ("us", "ca", "order6"),
      ("fr", "paris", "order7")
    ).toDF("country", "city", "order_id")
    multiKeyFact.createOrReplaceTempView("geo_fact")

    // Test with different shard counts to exercise different mod distributions
    for (shardCount <- Seq(2, 3, 5, 7)) {
      val dmjSql =
        s"""SELECT /*+ DISTMAPJOIN(d(shard_count=$shardCount, replica_count=1)) */
           |  f.country, f.city, d.region, f.order_id
           |FROM geo_fact f JOIN geo_dim d ON f.country = d.country AND f.city = d.city
           |""".stripMargin
      val baselineSql =
        """SELECT f.country, f.city, d.region, f.order_id
          |FROM geo_fact f JOIN geo_dim d ON f.country = d.country AND f.city = d.city
          |""".stripMargin
      val dmjDf = sql(dmjSql)
      checkAnswer(dmjDf, sql(baselineSql))
    }
  }

  test("string key with special characters (unicode, empty, whitespace)") {
    val sparkSession = spark
    import sparkSession.implicits._

    val specialDim = Seq(
      ("", "empty"),
      (" ", "space"),
      ("hello world", "with_space"),
      ("café", "accent"),
      ("中文测试", "chinese"),
      ("emoji😀", "emoji"),
      ("a" * 200, "long_key")
    ).toDF("k", "label")
    specialDim.createOrReplaceTempView("special_dim")

    val specialFact = Seq(
      ("", "f1"),
      (" ", "f2"),
      ("hello world", "f3"),
      ("café", "f4"),
      ("中文测试", "f5"),
      ("emoji😀", "f6"),
      ("a" * 200, "f7"),
      ("no_match", "f8")
    ).toDF("k", "payload")
    specialFact.createOrReplaceTempView("special_fact")

    val dmjSql =
      """SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
        |  f.k, d.label, f.payload
        |FROM special_fact f JOIN special_dim d ON f.k = d.k
        |""".stripMargin
    val baselineSql =
      """SELECT f.k, d.label, f.payload
        |FROM special_fact f JOIN special_dim d ON f.k = d.k""".stripMargin
    val dmjDf = sql(dmjSql)
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    checkAnswer(dmjDf, sql(baselineSql))
    assertNativeOffload(dmjDf)
  }

  // =========================================================================
  // Plan structure / offload validation tests
  // =========================================================================

  test("plan: AQE enabled still produces DistributedMapJoinExecTransformer") {
    withSQLConf("spark.sql.adaptive.enabled" -> "true") {
      val df = sql("""SELECT /*+ DISTMAPJOIN(d(shard_count=3, replica_count=1)) */
                     |  f.k, d.v, f.payload
                     |FROM fact f JOIN dim d ON f.k = d.k
                     |""".stripMargin)
      val sparkPlan = df.queryExecution.sparkPlan
      val dmjExecs = sparkPlan.collect { case d: DistributedMapJoinExec => d }
      assert(
        dmjExecs.nonEmpty,
        s"With AQE, sparkPlan should still contain DistributedMapJoinExec:\n$sparkPlan")
      val executedPlan = df.queryExecution.executedPlan
      assert(
        executedPlan
          .isInstanceOf[org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec],
        s"With AQE enabled, executedPlan should be AdaptiveSparkPlanExec:\n$executedPlan"
      )
    }
  }

  test("TPC-DS style multi-table join with DISTMAPJOIN on customer") {
    val dmjSql =
      """SELECT /*+ MAPJOIN(d), MAPJOIN(i),
        |        DISTMAPJOIN(c(shard_count=3, replica_count=1)) */
        |  i.i_brand_id,
        |  i.i_brand,
        |  c.c_birth_country,
        |  d.d_year,
        |  d.d_moy,
        |  SUM(ss.ss_ext_sales_price) AS sales_amt
        |FROM store_sales ss
        |JOIN item i ON ss.ss_item_sk = i.i_item_sk
        |JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
        |JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
        |WHERE d.d_year BETWEEN 2000 AND 2002
        |  AND i.i_category IN ('Women', 'Men', 'Electronics')
        |GROUP BY i.i_brand_id, i.i_brand, c.c_birth_country, d.d_year, d.d_moy
        |ORDER BY sales_amt DESC
        |""".stripMargin
    val baselineSql =
      """SELECT
        |  i.i_brand_id,
        |  i.i_brand,
        |  c.c_birth_country,
        |  d.d_year,
        |  d.d_moy,
        |  SUM(ss.ss_ext_sales_price) AS sales_amt
        |FROM store_sales ss
        |JOIN item i ON ss.ss_item_sk = i.i_item_sk
        |JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
        |JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
        |WHERE d.d_year BETWEEN 2000 AND 2002
        |  AND i.i_category IN ('Women', 'Men', 'Electronics')
        |GROUP BY i.i_brand_id, i.i_brand, c.c_birth_country, d.d_year, d.d_moy
        |ORDER BY sales_amt DESC
        |""".stripMargin
    val dmjDf = sql(dmjSql)
    // Verify DMJ is present in the plan
    assertDMJPlanCount(dmjDf.queryExecution.sparkPlan, expectedCount = 1)
    // Verify correctness — this triggers AQE execution
    checkAnswer(dmjDf, sql(baselineSql))
    // Print the final offloaded plan after AQE has finished
    val finalPlan = dmjDf.queryExecution.executedPlan
    logInfo(s"[DEBUG] TPC-DS style AQE finalPlan:\n${finalPlan.toString}")
  }
}
