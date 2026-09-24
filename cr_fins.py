from pyspark.sql import functions as F

# ============================================================
# READ INFLATION TABLE
# ============================================================

INFLATION_TABLE = "src_edwlive_dm_cad.mva_trend_inflation_rate"

inflation = (
    spark.table(INFLATION_TABLE)
    .select(
        F.to_date(F.col("data_date")).alias("inf_data_date"),
        F.col("inflation_rate").cast("double").alias("inflation_rate")
    )
    .dropDuplicates(["inf_data_date"])
)


# ============================================================
# FINANCIAL COLUMNS TO INFLATION-ADJUST
# ============================================================

financial_cols = [
    "produce_depreciation_exp_6203",
    "operational_profit_loss_82",
    "administrative_depr_exp_6303",
    "net_sales_80",
    "total_assets_812",
    "equities_5",
    "net_period_profit_loss_84",
    "revolving_asset_1",
    "fixed_asset_2",
    "short_term_foreign_resource_3",
    "inventory_15",
    "other_receivable_13",
    "other_liability_33",
]


# ============================================================
# CLEAN UP EXISTING JOINED TABLE
# ============================================================
# We already have cr_rds + fin in `joined`.
#
# Select the RDS keys once and keep the financial columns.
# close_date is already the ACTUAL historical financial
# statement date, including for prev_financial_indicator = 1.
# ============================================================

joined = joined.select(
    F.col("r.party_id").alias("party_id"),
    F.col("r.month_start").alias("month_start"),
    F.col("r.proposal_id").alias("proposal_id"),

    F.col("f.data_date").alias("data_date"),
    F.col("f.prev_financial_indicator").alias(
        "prev_financial_indicator"
    ),
    F.col("f.financial_table_id").alias("financial_table_id"),
    F.to_date(F.col("f.close_date")).alias("close_date"),

    *[
        F.col(f"f.{col_name}").cast("double").alias(col_name)
        for col_name in financial_cols
    ]
)


# ============================================================
# JOIN INFLATION USING CLOSE_DATE
# ============================================================
#
# No special prev_financial_indicator logic is needed:
#
# prev = 0 -> close_date is the current statement's date
# prev = 1 -> close_date is already the historical statement date
#
# Therefore each financial statement automatically receives
# the inflation coefficient corresponding to its own close_date.
# ============================================================

joined = (
    joined
    .join(
        inflation,
        F.col("close_date") == F.col("inf_data_date"),
        "left"
    )
)


# ============================================================
# APPLY INFLATION ADJUSTMENT
# ============================================================
#
# adjusted value = original value / inflation_rate
#
# Existing NULL financial values remain NULL.
# Missing/zero inflation rates result in NULL.
# ============================================================

for col_name in financial_cols:
    joined = joined.withColumn(
        col_name,
        F.when(
            F.col(col_name).isNull(),
            F.lit(None).cast("double")
        )
        .when(
            F.col("inflation_rate").isNull()
            | (F.col("inflation_rate") == 0),
            F.lit(None).cast("double")
        )
        .otherwise(
            F.col(col_name) / F.col("inflation_rate")
        )
    )


# ============================================================
# REMOVE INFLATION JOIN KEY
# ============================================================

joined = joined.drop("inf_data_date")
