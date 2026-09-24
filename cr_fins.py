from pyspark.sql import functions as F

cr_rds_j = (
    cr_rds
    .withColumn("party_id_join", F.col("party_id").cast("decimal(16,0)"))
    .withColumn("month_start_join", F.to_date("month_start", "yyyy-MM-dd"))
)

fin_j = (
    fin
    .withColumn("party_id_join", F.col("party_id").cast("decimal(16,0)"))
    .withColumn("data_date_join", F.col("data_date").cast("date"))
)

joined = (
    cr_rds.alias("r")
    .join(
        fin.alias("f"),
        (F.col("r.party_id").cast("decimal(16,0)") == F.col("f.party_id")) &
        (F.date_sub(F.to_date("r.month_start", "yyyy-MM-dd"), 1) == F.col("f.data_date")),
        "left"
    )
)



#####

from pyspark.sql import functions as F

# ============================================================
# 1. READ INFLATION TABLE
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
# 2. JOIN CR_RDS WITH FINANCIAL TABLE
# ============================================================

joined = (
    cr_rds.alias("r")
    .join(
        fin.alias("f"),
        (
            (F.col("r.party_id") == F.col("f.party_id"))
            &
            (
                F.date_sub(
                    F.to_date(F.col("r.month_start"), "yyyy-MM-dd"),
                    1
                )
                == F.to_date(F.col("f.data_date"))
            )
        ),
        "left"
    )
)


# ============================================================
# 3. JOIN INFLATION RATE USING ACTUAL CLOSE_DATE
# ============================================================
#
# close_date is already the actual historical statement date.
#
# Therefore:
#   prev_financial_indicator = 0 -> use its close_date
#   prev_financial_indicator = 1 -> use its close_date
#
# No additional -12 month adjustment is required.
# ============================================================

joined = (
    joined
    .withColumn(
        "close_date_parsed",
        F.to_date(F.col("f.close_date"))
    )
    .join(
        inflation.alias("i"),
        F.col("close_date_parsed") == F.col("i.inf_data_date"),
        "left"
    )
)


# ============================================================
# 4. FINANCIAL COLUMNS TO INFLATION-ADJUST
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
# 5. APPLY INFLATION ADJUSTMENT
# ============================================================
#
# adjusted_value = original_value / inflation_rate
#
# NULL financial value  -> NULL
# NULL inflation rate   -> NULL
# inflation rate = 0    -> NULL
# ============================================================

for col_name in financial_cols:
    joined = joined.withColumn(
        col_name,
        F.when(
            F.col(f"f.{col_name}").isNull(),
            F.lit(None).cast("double")
        )
        .when(
            F.col("i.inflation_rate").isNull()
            | (F.col("i.inflation_rate") == 0),
            F.lit(None).cast("double")
        )
        .otherwise(
            F.col(f"f.{col_name}").cast("double")
            / F.col("i.inflation_rate")
        )
    )


# ============================================================
# 6. FINAL SELECT
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
    F.col("f.close_date").alias("close_date"),

    # Inflation-adjusted financial values
    *[
        F.col(col_name)
        for col_name in financial_cols
    ],

    # Keep temporarily for checking that the correct
    # inflation coefficient was joined
    F.col("i.inflation_rate").alias("inflation_rate")
)
