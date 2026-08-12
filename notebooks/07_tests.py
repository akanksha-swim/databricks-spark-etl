# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Phase 12 — End-to-End Testing
# MAGIC
# MAGIC This notebook validates the current ETL pipeline:
# MAGIC
# MAGIC ```text
# MAGIC Bronze
# MAGIC    ↓
# MAGIC Silver
# MAGIC    ↓
# MAGIC Data Quality
# MAGIC    ↓
# MAGIC Gold
# MAGIC ```
# MAGIC
# MAGIC Tests cover:
# MAGIC - Silver data presence
# MAGIC - Required columns
# MAGIC - Data types
# MAGIC - NULL customer IDs
# MAGIC - Duplicate order IDs
# MAGIC - Invalid quantities
# MAGIC - Invalid prices
# MAGIC - Invalid discounts
# MAGIC - Invalid statuses
# MAGIC - Invalid dates
# MAGIC - Quarantine records
# MAGIC - Quarantine rejection reasons
# MAGIC - Gold tables
# MAGIC - Silver/Gold reconciliation
# MAGIC - Invalid quantity detection
# MAGIC - Empty input
# MAGIC - Duplicate detection
# MAGIC
# MAGIC No external Python test package is required. The notebook uses Python `assert`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

print("Test environment initialized.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

SILVER_TABLE = "workspace.silver.orders"
QUARANTINE_TABLE = "workspace.silver.orders_quarantine"

GOLD_DAILY_TABLE = "workspace.gold.daily_sales"
GOLD_CATEGORY_TABLE = "workspace.gold.category_sales"
GOLD_COUNTRY_TABLE = "workspace.gold.country_sales"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Silver

# COMMAND ----------

silver_df = spark.table(SILVER_TABLE)

silver_count = silver_df.count()

print(f"Silver records: {silver_count}")

assert silver_count > 0, \
    "TEST FAILED: Silver table is empty."

print("TEST PASSED: Silver contains records.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Required Silver columns

# COMMAND ----------

required_columns = [
    "order_id",
    "customer_id",
    "order_date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "discount",
    "country",
    "payment_method",
    "order_status",
    "source_file",
    "batch_id",
    "ingestion_timestamp"
]

missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in silver_df.columns
]

assert not missing_columns, \
    f"TEST FAILED: Missing Silver columns: {missing_columns}"

print("TEST PASSED: All required Silver columns exist.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data type validation

# COMMAND ----------

field_types = dict(silver_df.dtypes)

print("Silver data types:")
for column_name in required_columns:
    print(f"{column_name}: {field_types.get(column_name)}")

assert field_types["order_date"] == "date", \
    "TEST FAILED: order_date must be DATE."

assert field_types["quantity"] == "int", \
    "TEST FAILED: quantity must be INT."

assert field_types["unit_price"] == "decimal(18,2)", \
    "TEST FAILED: unit_price must be DECIMAL(18,2)."

assert field_types["discount"] == "decimal(5,2)", \
    "TEST FAILED: discount must be DECIMAL(5,2)."

print("TEST PASSED: Silver data types are correct.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. NULL customer ID test

# COMMAND ----------

null_customer_count = (
    silver_df
    .filter(F.col("customer_id").isNull())
    .count()
)

assert null_customer_count == 0, \
    f"TEST FAILED: {null_customer_count} NULL customer IDs found."

print("TEST PASSED: No NULL customer IDs in Silver.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Duplicate order ID test

# COMMAND ----------

duplicate_order_count = (
    silver_df
    .groupBy("order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

assert duplicate_order_count == 0, \
    f"TEST FAILED: {duplicate_order_count} duplicate order ID groups found."

print("TEST PASSED: Silver order IDs are unique.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quantity validation

# COMMAND ----------

invalid_quantity_count = (
    silver_df
    .filter(F.col("quantity") <= 0)
    .count()
)

assert invalid_quantity_count == 0, \
    f"TEST FAILED: {invalid_quantity_count} invalid quantities found."

print("TEST PASSED: All Silver quantities are greater than zero.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Price validation

# COMMAND ----------

invalid_price_count = (
    silver_df
    .filter(F.col("unit_price") < 0)
    .count()
)

assert invalid_price_count == 0, \
    f"TEST FAILED: {invalid_price_count} invalid prices found."

print("TEST PASSED: All Silver prices are valid.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Discount validation

# COMMAND ----------

invalid_discount_count = (
    silver_df
    .filter(
        (F.col("discount") < 0)
        | (F.col("discount") > 100)
    )
    .count()
)

assert invalid_discount_count == 0, \
    f"TEST FAILED: {invalid_discount_count} invalid discounts found."

print("TEST PASSED: All Silver discounts are between 0 and 100.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Order status validation

# COMMAND ----------

valid_statuses = [
    "Completed",
    "Cancelled",
    "Pending",
    "Returned"
]

invalid_status_count = (
    silver_df
    .filter(
        ~F.col("order_status").isin(valid_statuses)
    )
    .count()
)

assert invalid_status_count == 0, \
    f"TEST FAILED: {invalid_status_count} invalid statuses found."

print("TEST PASSED: All Silver order statuses are valid.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Order date validation

# COMMAND ----------

null_date_count = (
    silver_df
    .filter(F.col("order_date").isNull())
    .count()
)

assert null_date_count == 0, \
    f"TEST FAILED: {null_date_count} NULL/invalid dates found."

print("TEST PASSED: All Silver order dates are valid.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Load quarantine

# COMMAND ----------

quarantine_df = spark.table(QUARANTINE_TABLE)

quarantine_count = quarantine_df.count()

print(f"Quarantined records: {quarantine_count}")

# The source data was intentionally generated with bad records.
assert quarantine_count > 0, \
    "TEST FAILED: Expected invalid source records in quarantine."

print("TEST PASSED: Invalid records were quarantined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Quarantine records must have rejection reasons

# COMMAND ----------

missing_reason_count = (
    quarantine_df
    .filter(
        F.col("validation_error").isNull()
        | (F.trim(F.col("validation_error")) == "")
    )
    .count()
)

assert missing_reason_count == 0, \
    f"TEST FAILED: {missing_reason_count} quarantine records have no rejection reason."

print("TEST PASSED: All quarantine records have rejection reasons.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Gold tables must exist

# COMMAND ----------

gold_tables = [
    GOLD_DAILY_TABLE,
    GOLD_CATEGORY_TABLE,
    GOLD_COUNTRY_TABLE
]

for table_name in gold_tables:
    assert spark.catalog.tableExists(table_name), \
        f"TEST FAILED: Gold table does not exist: {table_name}"

print("TEST PASSED: All Gold tables exist.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Gold tables must contain data

# COMMAND ----------

for table_name in gold_tables:

    count = spark.table(table_name).count()

    print(f"{table_name}: {count} records")

    assert count > 0, \
        f"TEST FAILED: Gold table is empty: {table_name}"

print("TEST PASSED: All Gold tables contain data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Silver → Gold sales reconciliation
# MAGIC
# MAGIC This verifies that the business calculation in Gold agrees with the validated Silver data.

# COMMAND ----------

silver_sales_df = (
    silver_df
    .withColumn(
        "net_sales",
        F.col("quantity")
        * F.col("unit_price")
        * (
            F.lit(1)
            - F.col("discount") / F.lit(100)
        )
    )
)

silver_net_sales = (
    silver_sales_df
    .agg(F.sum("net_sales").alias("total"))
    .first()["total"]
)

gold_net_sales = (
    spark.table(GOLD_DAILY_TABLE)
    .agg(F.sum("net_sales").alias("total"))
    .first()["total"]
)

silver_net_sales_value = (
    float(silver_net_sales)
    if silver_net_sales is not None
    else 0.0
)

gold_net_sales_value = (
    float(gold_net_sales)
    if gold_net_sales is not None
    else 0.0
)

difference = abs(
    silver_net_sales_value
    - gold_net_sales_value
)

print(f"Silver net sales: {silver_net_sales_value:.2f}")
print(f"Gold net sales  : {gold_net_sales_value:.2f}")
print(f"Difference      : {difference:.4f}")

assert difference < 0.01, \
    f"TEST FAILED: Silver/Gold reconciliation difference = {difference}"

print("TEST PASSED: Silver and Gold net sales reconcile.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Unit test — invalid quantity

# COMMAND ----------

test_schema = T.StructType([
    T.StructField("order_id", T.StringType(), True),
    T.StructField("quantity", T.IntegerType(), True)
])

test_data = [
    ("TEST001", 5),
    ("TEST002", -1),
    ("TEST003", 0),
    ("TEST004", 10)
]

test_df = spark.createDataFrame(
    test_data,
    test_schema
)

invalid_count = (
    test_df
    .filter(F.col("quantity") <= 0)
    .count()
)

assert invalid_count == 2, \
    f"TEST FAILED: Expected 2 invalid quantities, got {invalid_count}"

print("TEST PASSED: Invalid quantity rule works.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Unit test — empty input

# COMMAND ----------

empty_df = spark.createDataFrame(
    [],
    test_schema
)

empty_count = empty_df.count()

assert empty_count == 0, \
    "TEST FAILED: Empty input test did not return zero records."

print("TEST PASSED: Empty input is handled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Unit test — duplicate detection

# COMMAND ----------

duplicate_test_data = [
    ("ORD001", 10),
    ("ORD002", 20),
    ("ORD001", 10)
]

duplicate_test_df = spark.createDataFrame(
    duplicate_test_data,
    test_schema
)

duplicate_count = (
    duplicate_test_df
    .groupBy("order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

assert duplicate_count == 1, \
    f"TEST FAILED: Expected 1 duplicate group, got {duplicate_count}"

print("TEST PASSED: Duplicate detection works.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Unit test — invalid price

# COMMAND ----------

price_test_schema = T.StructType([
    T.StructField("unit_price", T.DoubleType(), True)
])

price_test_df = spark.createDataFrame(
    [
        (100.0,),
        (-5.0,),
        (0.0,),
        (25.5,)
    ],
    price_test_schema
)

invalid_price_test_count = (
    price_test_df
    .filter(F.col("unit_price") < 0)
    .count()
)

assert invalid_price_test_count == 1, \
    f"TEST FAILED: Expected 1 invalid price, got {invalid_price_test_count}"

print("TEST PASSED: Invalid price rule works.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Unit test — invalid discount

# COMMAND ----------

discount_test_schema = T.StructType([
    T.StructField("discount", T.DoubleType(), True)
])

discount_test_df = spark.createDataFrame(
    [
        (0.0,),
        (25.0,),
        (100.0,),
        (-1.0,),
        (101.0,)
    ],
    discount_test_schema
)

invalid_discount_test_count = (
    discount_test_df
    .filter(
        (F.col("discount") < 0)
        | (F.col("discount") > 100)
    )
    .count()
)

assert invalid_discount_test_count == 2, \
    f"TEST FAILED: Expected 2 invalid discounts, got {invalid_discount_test_count}"

print("TEST PASSED: Invalid discount rule works.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Unit test — invalid status

# COMMAND ----------

status_test_schema = T.StructType([
    T.StructField("order_status", T.StringType(), True)
])

status_test_df = spark.createDataFrame(
    [
        ("Completed",),
        ("Cancelled",),
        ("Pending",),
        ("Returned",),
        ("UNKNOWN_STATUS",)
    ],
    status_test_schema
)

invalid_status_test_count = (
    status_test_df
    .filter(
        ~F.col("order_status").isin(valid_statuses)
    )
    .count()
)

assert invalid_status_test_count == 1, \
    f"TEST FAILED: Expected 1 invalid status, got {invalid_status_test_count}"

print("TEST PASSED: Invalid status rule works.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23. Test summary
# MAGIC
# MAGIC Every preceding `assert` must succeed for this cell to execute.

# COMMAND ----------

tests = [
    "Silver not empty",
    "Required Silver columns",
    "Silver data types",
    "No NULL customer IDs",
    "Unique order IDs",
    "Positive quantities",
    "Valid prices",
    "Valid discounts",
    "Valid order statuses",
    "Valid order dates",
    "Quarantine populated",
    "Quarantine rejection reasons",
    "Gold tables exist",
    "Gold tables populated",
    "Silver/Gold reconciliation",
    "Invalid quantity detection",
    "Empty input handling",
    "Duplicate detection",
    "Invalid price detection",
    "Invalid discount detection",
    "Invalid status detection"
]

print("=" * 65)
print("END-TO-END TEST SUITE PASSED")
print("=" * 65)

for number, test_name in enumerate(tests, start=1):
    print(f"{number:02d}. PASS - {test_name}")

print("=" * 65)
print(f"TOTAL TESTS PASSED: {len(tests)}")
print("RESULT: PASS")
print("=" * 65)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 12 checkpoint
# MAGIC
# MAGIC Expected final result:
# MAGIC
# MAGIC ```text
# MAGIC END-TO-END TEST SUITE PASSED
# MAGIC TOTAL TESTS PASSED: 21
# MAGIC RESULT: PASS
# MAGIC ```
# MAGIC
# MAGIC If any test fails, fix that phase before proceeding to Workflow.
# MAGIC
# MAGIC Do not add this test notebook as a mandatory production task yet. It is our development/integration test suite.