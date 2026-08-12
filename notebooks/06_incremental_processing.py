# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Phase 11 — Incremental Processing + Idempotency
# MAGIC
# MAGIC This notebook is designed for the current project environment:
# MAGIC
# MAGIC - Databricks Serverless
# MAGIC - Unity Catalog
# MAGIC - Spark 4.1
# MAGIC - Python 3.12
# MAGIC - Volume: `/Volumes/workspace/ingestion/project_data/raw`
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Process only raw CSV files that have not already been processed.
# MAGIC
# MAGIC ```text
# MAGIC Raw Volume
# MAGIC     |
# MAGIC     v
# MAGIC Discover CSV files
# MAGIC     |
# MAGIC     v
# MAGIC processed_files audit table
# MAGIC     |
# MAGIC     +---- already processed ---> SKIP
# MAGIC     |
# MAGIC     +---- new file -----------> READ
# MAGIC                                      |
# MAGIC                                      v
# MAGIC                                Bronze Delta
# MAGIC                                      |
# MAGIC                                      v
# MAGIC                                Audit SUCCESS
# MAGIC ```
# MAGIC
# MAGIC Important:
# MAGIC - We use `_metadata.file_name` and `_metadata.file_path`.
# MAGIC - We do NOT use `input_file_name()` because the current Unity Catalog Serverless environment does not support it.
# MAGIC - We do NOT add `source_path` to Bronze because the existing Bronze schema does not contain it.
# MAGIC - We use the existing Bronze schema exactly when appending.
# MAGIC
# MAGIC This notebook is intentionally development/trial friendly. A production implementation would use a stronger transactional ingestion pattern and/or Auto Loader.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

RAW_PATH = "/Volumes/workspace/ingestion/project_data/raw"

BRONZE_TABLE = "workspace.bronze.orders"

FILE_AUDIT_TABLE = "workspace.audit.processed_files"

print("RAW_PATH       :", RAW_PATH)
print("BRONZE_TABLE   :", BRONZE_TABLE)
print("FILE_AUDIT_TABLE:", FILE_AUDIT_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Shared raw schema
# MAGIC
# MAGIC This schema is defined in this notebook deliberately.
# MAGIC
# MAGIC Notebook variables are not automatically shared between separate Databricks notebooks, so this notebook must not depend on `raw_schema` from the Bronze notebook.

# COMMAND ----------

raw_schema = T.StructType([
    T.StructField("order_id", T.StringType(), True),
    T.StructField("customer_id", T.StringType(), True),
    T.StructField("order_date", T.StringType(), True),
    T.StructField("product_id", T.StringType(), True),
    T.StructField("product_name", T.StringType(), True),
    T.StructField("category", T.StringType(), True),
    T.StructField("quantity", T.StringType(), True),
    T.StructField("unit_price", T.StringType(), True),
    T.StructField("discount", T.StringType(), True),
    T.StructField("country", T.StringType(), True),
    T.StructField("payment_method", T.StringType(), True),
    T.StructField("order_status", T.StringType(), True)
])

print("Raw schema created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the file-processing audit table if it does not exist
# MAGIC
# MAGIC The audit table records every successfully processed physical CSV file.

# COMMAND ----------

file_audit_schema = T.StructType([
    T.StructField("file_path", T.StringType(), False),
    T.StructField("file_name", T.StringType(), True),
    T.StructField("batch_id", T.StringType(), True),
    T.StructField("processed_timestamp", T.TimestampType(), True),
    T.StructField("record_count", T.LongType(), True),
    T.StructField("status", T.StringType(), True)
])

empty_audit_df = spark.createDataFrame(
    [],
    file_audit_schema
)

(
    empty_audit_df
    .write
    .format("delta")
    .mode("ignore")
    .saveAsTable(FILE_AUDIT_TABLE)
)

print(f"Audit table ready: {FILE_AUDIT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Recursively discover CSV files
# MAGIC
# MAGIC Spark writes CSV output into directories containing `part-*.csv`.
# MAGIC
# MAGIC Therefore we discover the actual CSV files instead of assuming that `orders_001` itself is a file.

# COMMAND ----------

def list_csv_files(path):
    csv_files = []

    for item in dbutils.fs.ls(path):
        if item.isDir():
            csv_files.extend(list_csv_files(item.path))
        elif item.path.lower().endswith(".csv"):
            csv_files.append(item.path)

    return csv_files


csv_files = sorted(list_csv_files(RAW_PATH))

print(f"CSV files discovered: {len(csv_files)}")

for file_path in csv_files:
    print(file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build discovered-file DataFrame

# COMMAND ----------

discovered_schema = T.StructType([
    T.StructField("file_path", T.StringType(), False)
])

discovered_df = spark.createDataFrame(
    [(path,) for path in csv_files],
    discovered_schema
)

discovered_df = (
    discovered_df
    .withColumn(
        "file_name",
        F.element_at(
            F.split(F.col("file_path"), "/"),
            -1
        )
    )
    .withColumn(
        "batch_number",
        F.regexp_extract(
            F.col("file_path"),
            r"orders_(\d+)",
            1
        )
    )
    .withColumn(
        "batch_id",
        F.when(
            F.col("batch_number") == "",
            F.lit("UNKNOWN")
        ).otherwise(
            F.concat(
                F.lit("BATCH_"),
                F.col("batch_number")
            )
        )
    )
    .drop("batch_number")
)

display(discovered_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Read already-processed files

# COMMAND ----------

processed_df = (
    spark.table(FILE_AUDIT_TABLE)
    .filter(F.col("status") == "SUCCESS")
    .select("file_path")
    .dropDuplicates()
)

print("Already successfully processed files:")
display(processed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Find only NEW files
# MAGIC
# MAGIC `left_anti` means:
# MAGIC
# MAGIC > Keep rows from the left side that do not have a matching row on the right side.
# MAGIC
# MAGIC This is the core of our file-level idempotency check.

# COMMAND ----------

new_files_df = (
    discovered_df
    .join(
        processed_df,
        on="file_path",
        how="left_anti"
    )
)

print("New files:", new_files_df.count())

display(new_files_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Collect new file paths
# MAGIC
# MAGIC The list is intentionally created only after the audit check.

# COMMAND ----------

new_file_rows = new_files_df.select("file_path").collect()

new_file_paths = [
    row["file_path"]
    for row in new_file_rows
]

print(f"Files to process: {len(new_file_paths)}")

for path in new_file_paths:
    print(path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Read only NEW files
# MAGIC
# MAGIC If there are no new files, the notebook exits this processing section without reading anything.

# COMMAND ----------

if new_file_paths:

    incremental_raw_df = (
        spark.read
        .schema(raw_schema)
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .csv(new_file_paths)
    )

    new_record_count = incremental_raw_df.count()

    print(f"New records discovered: {new_record_count}")

else:

    incremental_raw_df = None
    new_record_count = 0

    print("No new files to process.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Add Bronze metadata
# MAGIC
# MAGIC `_metadata.file_name` and `_metadata.file_path` are supported for this file-ingestion pattern.
# MAGIC
# MAGIC We store `source_file`, `batch_id`, and `ingestion_timestamp` in Bronze.
# MAGIC
# MAGIC We intentionally do NOT add `source_path`, because the existing Bronze table schema does not contain that column.

# COMMAND ----------

if incremental_raw_df is not None:

    incremental_bronze_df = (
        incremental_raw_df

        .withColumn(
            "source_file",
            F.col("_metadata.file_name")
        )

        .withColumn(
            "batch_id",
            F.regexp_extract(
                F.col("_metadata.file_path"),
                r"orders_(\d+)",
                1
            )
        )

        .withColumn(
            "batch_id",
            F.when(
                F.col("batch_id") == "",
                F.lit("UNKNOWN")
            ).otherwise(
                F.concat(
                    F.lit("BATCH_"),
                    F.col("batch_id")
                )
            )
        )

        .withColumn(
            "ingestion_timestamp",
            F.current_timestamp()
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Force Bronze column compatibility
# MAGIC
# MAGIC Before writing, retrieve the existing Bronze columns and select them in exactly the same order.
# MAGIC
# MAGIC This prevents the `DELTA_METADATA_MISMATCH` problem encountered earlier.

# COMMAND ----------

bronze_columns = spark.table(BRONZE_TABLE).columns

print("Existing Bronze columns:")
for column_name in bronze_columns:
    print(column_name)

# COMMAND ----------

if incremental_raw_df is not None:

    incremental_bronze_df = incremental_bronze_df.select(
        bronze_columns
    )

    print("Incremental Bronze schema aligned with existing table.")

    display(incremental_bronze_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Append NEW records to Bronze
# MAGIC
# MAGIC We use `append`, not `overwrite`.
# MAGIC
# MAGIC Existing Bronze data is therefore preserved.

# COMMAND ----------

if incremental_raw_df is not None:

    (
        incremental_bronze_df
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(BRONZE_TABLE)
    )

    print(
        f"Successfully appended {new_record_count} records to Bronze."
    )

else:

    print("Nothing to append.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Record successfully processed files
# MAGIC
# MAGIC The audit write happens only after the Bronze append succeeds.

# COMMAND ----------

if new_file_paths:

    audit_records = (
        new_files_df
        .withColumn(
            "processed_timestamp",
            F.current_timestamp()
        )
        .withColumn(
            "record_count",
            F.lit(new_record_count).cast("long")
        )
        .withColumn(
            "status",
            F.lit("SUCCESS")
        )
        .select(
            "file_path",
            "file_name",
            "batch_id",
            "processed_timestamp",
            "record_count",
            "status"
        )
    )

    (
        audit_records
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(FILE_AUDIT_TABLE)
    )

    print(
        f"Audit updated for {len(new_file_paths)} file(s)."
    )

else:

    print("No audit records to add.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Verify processing audit

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT
            file_name,
            batch_id,
            record_count,
            status,
            processed_timestamp
        FROM {FILE_AUDIT_TABLE}
        ORDER BY processed_timestamp
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Verify Bronze by batch

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT
            batch_id,
            COUNT(*) AS record_count
        FROM {BRONZE_TABLE}
        GROUP BY batch_id
        ORDER BY batch_id
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. IDEMPOTENCY TEST
# MAGIC
# MAGIC Run the file discovery and anti-join again.
# MAGIC
# MAGIC Previously successful files must now return zero rows.

# COMMAND ----------

csv_files_after = sorted(list_csv_files(RAW_PATH))

discovered_after_df = spark.createDataFrame(
    [(path,) for path in csv_files_after],
    discovered_schema
)

new_files_after_df = (
    discovered_after_df
    .join(
        spark.table(FILE_AUDIT_TABLE)
        .filter(F.col("status") == "SUCCESS")
        .select("file_path")
        .dropDuplicates(),
        on="file_path",
        how="left_anti"
    )
)

remaining_new_files = new_files_after_df.count()

print(
    f"New files remaining after processing: {remaining_new_files}"
)

display(new_files_after_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Expected idempotency result
# MAGIC
# MAGIC For files that were successfully processed:
# MAGIC
# MAGIC ```text
# MAGIC Run 1
# MAGIC   orders_001 -> PROCESS
# MAGIC   orders_002 -> PROCESS
# MAGIC   orders_003 -> PROCESS
# MAGIC
# MAGIC Run 2
# MAGIC   orders_001 -> SKIP
# MAGIC   orders_002 -> SKIP
# MAGIC   orders_003 -> SKIP
# MAGIC ```
# MAGIC
# MAGIC Therefore the final value should be:
# MAGIC
# MAGIC `New files remaining after processing: 0`
# MAGIC
# MAGIC If `orders_004` was created but not yet processed, the value should be `1` before the processing section runs and `0` after it runs.

# COMMAND ----------

if remaining_new_files == 0:
    print("IDEMPOTENCY CHECK PASSED.")
else:
    print(
        f"{remaining_new_files} new file(s) still require processing."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Final validation
# MAGIC
# MAGIC Check that the audit table and Bronze table are populated.

# COMMAND ----------

audit_count = spark.table(FILE_AUDIT_TABLE).count()
bronze_count = spark.table(BRONZE_TABLE).count()

print(f"Audit records : {audit_count}")
print(f"Bronze records: {bronze_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 11 checkpoint
# MAGIC
# MAGIC Expected architecture:
# MAGIC
# MAGIC ```text
# MAGIC Raw Volume
# MAGIC     |
# MAGIC     v
# MAGIC Discover CSV files
# MAGIC     |
# MAGIC     v
# MAGIC workspace.audit.processed_files
# MAGIC     |
# MAGIC     +---- SUCCESS ---> SKIP
# MAGIC     |
# MAGIC     +---- NEW -------> Spark Read
# MAGIC                           |
# MAGIC                           v
# MAGIC                     Bronze Delta
# MAGIC                           |
# MAGIC                           v
# MAGIC                     Audit SUCCESS
# MAGIC ```
# MAGIC
# MAGIC ### Important trial limitation
# MAGIC
# MAGIC This implementation provides file-level idempotency for the learning project. There is a small failure window between the Bronze append and the audit write. Production ingestion should use a stronger transactional pattern, Auto Loader, or another ingestion mechanism designed for exactly-once/near-exactly-once processing.
# MAGIC
# MAGIC The next phase is testing.