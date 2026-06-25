from pyspark.sql import functions as F
from pyspark.sql.window import Window

id_col = "party_id"
date_col = "data_date"

# Columns to forward fill
cols_to_ffill = [
    c for c in df.columns
    if c not in [id_col, date_col]
]

# 1) Convert sparse dates to month start
df_m = (
    df
    .withColumn("month_date", F.trunc(F.col(date_col), "month"))
)

# 2) Create each customer's min/max month
bounds = (
    df_m
    .groupBy(id_col)
    .agg(
        F.min("month_date").alias("min_month"),
        F.max("month_date").alias("max_month")
    )
)

# 3) Generate next 12 months after each customer's last available month
customer_months = (
    bounds
    .withColumn(
        "month_date",
        F.explode(
            F.sequence(
                F.col("min_month"),
                F.add_months(F.col("max_month"), 12),
                F.expr("interval 1 month")
            )
        )
    )
    .select(id_col, "month_date")
)

# 4) Join original sparse table onto full customer-month grid
df_grid = (
    customer_months
    .join(df_m.drop(date_col), on=[id_col, "month_date"], how="left")
)

# 5) Forward fill using last available value within previous 12 months
w = (
    Window
    .partitionBy(id_col)
    .orderBy("month_date")
    .rowsBetween(-12, 0)
)

for c in cols_to_ffill:
    df_grid = df_grid.withColumn(
        c,
        F.last(F.col(c), ignorenulls=True).over(w)
    )

result = df_grid.withColumnRenamed("month_date", date_col)
