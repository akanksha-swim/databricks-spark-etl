from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def row_count(df: DataFrame) -> int:
    """
    Return total number of records.
    """
    return df.count()


def null_count(df: DataFrame, column_name: str) -> int:
    """
    Return number of NULL values in a column.
    """
    return (
        df.filter(F.col(column_name).isNull())
        .count()
    )


def duplicate_count(
    df: DataFrame,
    column_name: str
) -> int:
    """
    Return number of duplicate records
    based on a business key.
    """

    duplicate_groups = (
        df.groupBy(column_name)
        .count()
        .filter(F.col("count") > 1)
    )

    return duplicate_groups.count()


def invalid_quantity_count(df: DataFrame) -> int:
    """
    Quantity must be greater than zero.
    """

    return (
        df.filter(
            F.col("quantity") <= 0
        )
        .count()
    )


def invalid_price_count(df: DataFrame) -> int:
    """
    Unit price must be greater than or equal to zero.
    """

    return (
        df.filter(
            F.col("unit_price") < 0
        )
        .count()
    )


def invalid_discount_count(df: DataFrame) -> int:
    """
    Discount must be between 0 and 100.
    """

    return (
        df.filter(
            (F.col("discount") < 0)
            | (F.col("discount") > 100)
        )
        .count()
    )


def invalid_status_count(
    df: DataFrame
) -> int:

    valid_statuses = [
        "Completed",
        "Cancelled",
        "Pending",
        "Returned"
    ]

    return (
        df.filter(
            ~F.col("order_status")
            .isin(valid_statuses)
        )
        .count()
    )