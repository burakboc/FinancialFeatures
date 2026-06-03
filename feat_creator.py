import yaml
from functools import reduce
from pyspark.sql import functions as F

# ============================================================
# PARAMETERS
# ============================================================

YAML_PATH = "/path/to/feature_config.yml"

FIN_TABLE = "analytics_risk_sb.financial_features_sme_limit_prometeia_oy_v2"
RDS_TABLE = "analytics_risk_sb.ala_target_curr_100_250_fin"
INFLATION_TABLE = "src_edwlive_dm_cad.mva_inflation_rate_new"

OUT_TABLE = "analytics_risk_sb.financial_features_all"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_list(spec, key):
    value = spec.get(key, [])
    return value if value is not None else []


def is_last_year_col(col_name):
    return col_name.endswith("_ly") or col_name.endswith("_1y")


def base_col_name(col_name):
    if col_name.endswith("_ly"):
        return col_name[:-3]
    if col_name.endswith("_1y"):
        return col_name[:-3]
    return col_name


def build_signed_sum(pos_cols, neg_cols):
    pos_exprs = [F.coalesce(F.col(c), F.lit(0.0)) for c in pos_cols]
    neg_exprs = [F.coalesce(F.col(c), F.lit(0.0)) for c in neg_cols]

    pos_sum = reduce(lambda a, b: a + b, pos_exprs) if pos_exprs else F.lit(0.0)
    neg_sum = reduce(lambda a, b: a + b, neg_exprs) if neg_exprs else F.lit(0.0)

    return pos_sum - neg_sum


def parse_feature_config(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    features = cfg.get("features", cfg)

    parsed = {}
    all_cols = set()

    for feat_name, spec in features.items():
        numerator = get_list(spec, "numerator")
        neg_numerator = get_list(spec, "neg_numerator")
        denominator = get_list(spec, "denominator")
        neg_denominator = get_list(spec, "neg_denominator")

        parsed[feat_name] = {
            "numerator": numerator,
            "neg_numerator": neg_numerator,
            "denominator": denominator,
            "neg_denominator": neg_denominator,
        }

        all_cols.update(numerator)
        all_cols.update(neg_numerator)
        all_cols.update(denominator)
        all_cols.update(neg_denominator)

    return parsed, sorted(all_cols)


def sql_col_list(cols):
    return ", ".join(cols)


# ============================================================
# 1. READ YAML AND FIND REQUIRED RAW COLUMNS
# ============================================================

feature_config, yaml_cols = parse_feature_config(YAML_PATH)

current_cols = sorted([
    c for c in yaml_cols
    if not is_last_year_col(c)
])

last_year_cols_requested = sorted([
    c for c in yaml_cols
    if is_last_year_col(c)
])

last_year_base_cols = sorted(set([
    base_col_name(c)
    for c in last_year_cols_requested
]))

required_fin_cols = sorted(set(current_cols + last_year_base_cols))

print("Current financial columns:")
print(current_cols)

print("Last-year financial base columns:")
print(last_year_base_cols)


# ============================================================
# 2. READ RDS WITH spark.sql
# ============================================================

rds_cols = ["party_id", "data_date"]
rds_column_names = sql_col_list(rds_cols)

rds = (
    spark.sql(f"""
        select {rds_column_names}
        from {RDS_TABLE}
    """)
    .select(
        F.col("party_id").cast("int").alias("party_id"),
        F.col("data_date").cast("date").alias("data_date")
    )
    .dropDuplicates(["party_id", "data_date"])
)


# ============================================================
# 3. READ FINANCIAL TABLE WITH spark.sql
# close_date format example: "31.12.2019"
# ============================================================

fin_cols_to_select = (
    ["party_id", "close_date", "prev_financial_indicator"]
    + required_fin_cols
)

fin_column_names = sql_col_list(fin_cols_to_select)

fin = (
    spark.sql(f"""
        select {fin_column_names}
        from {FIN_TABLE}
    """)
    .withColumn("party_id", F.col("party_id").cast("int"))
    .withColumn("prev_financial_indicator", F.col("prev_financial_indicator").cast("int"))
    .withColumn("close_date", F.to_date(F.col("close_date"), "dd.MM.yyyy"))
)


# ============================================================
# 4. FILTER FINANCIALS TO ONLY RDS PARTY_ID / DATE PAIRS EARLY
# ============================================================

rds_keys_for_fin_join = (
    rds
    .select(
        F.col("party_id"),
        F.col("data_date").alias("close_date")
    )
    .dropDuplicates(["party_id", "close_date"])
)

fin = (
    fin
    .join(
        rds_keys_for_fin_join,
        on=["party_id", "close_date"],
        how="inner"
    )
)


# ============================================================
# 5. SPLIT CURRENT AND LAST-YEAR FINANCIALS
# ============================================================

fin_0 = (
    fin
    .filter(F.col("prev_financial_indicator") == 0)
    .select(
        "party_id",
        "close_date",
        *current_cols
    )
)

fin_1_select_exprs = [
    F.col("party_id"),
    F.col("close_date")
] + [
    F.col(c).alias(f"{c}_1y")
    for c in last_year_base_cols
]

fin_1 = (
    fin
    .filter(F.col("prev_financial_indicator") == 1)
    .select(*fin_1_select_exprs)
)


# ============================================================
# 6. COMBINE CURRENT + LAST-YEAR FINANCIALS
# ============================================================

fin_wide = (
    fin_0
    .join(
        fin_1,
        on=["party_id", "close_date"],
        how="left"
    )
)


# ============================================================
# 7. READ INFLATION TABLE WITH spark.sql
# year_month example: 202604
# ============================================================

inflation_cols = ["year_month", "inflation_rate"]
inflation_column_names = sql_col_list(inflation_cols)

inflation = (
    spark.sql(f"""
        select {inflation_column_names}
        from {INFLATION_TABLE}
    """)
    .select(
        F.col("year_month").cast("int").alias("inflation_year_month"),
        F.col("inflation_rate").cast("double").alias("inflation_rate")
    )
)


# ============================================================
# 8. JOIN INFLATION RATE
# Uses end-of-month one month before close_date.
# ============================================================

fin_wide = (
    fin_wide
    .withColumn(
        "inflation_year_month",
        F.date_format(
            F.last_day(F.add_months(F.col("close_date"), -1)),
            "yyyyMM"
        ).cast("int")
    )
    .join(
        inflation,
        on="inflation_year_month",
        how="left"
    )
)


# ============================================================
# 9. APPLY INFLATION ADJUSTMENT TO RAW FINANCIAL COLUMNS
# ============================================================

raw_cols_to_adjust = [
    c for c in fin_wide.columns
    if c not in [
        "party_id",
        "close_date",
        "inflation_year_month",
        "inflation_rate"
    ]
]

for c in raw_cols_to_adjust:
    fin_wide = fin_wide.withColumn(
        c,
        F.col(c).cast("double") * F.col("inflation_rate")
    )


# ============================================================
# 10. CREATE FEATURES FROM YAML
# ============================================================

feature_exprs = []

for feat_name, spec in feature_config.items():
    numerator_expr = build_signed_sum(
        spec["numerator"],
        spec["neg_numerator"]
    )

    denominator_cols = spec["denominator"] + spec["neg_denominator"]

    if denominator_cols:
        denominator_expr = build_signed_sum(
            spec["denominator"],
            spec["neg_denominator"]
        )

        feature_expr = (
            F.when(
                denominator_expr.isNull() | (denominator_expr == 0),
                F.lit(None).cast("double")
            )
            .otherwise(numerator_expr / denominator_expr)
            .alias(feat_name)
        )
    else:
        feature_expr = numerator_expr.alias(feat_name)

    feature_exprs.append(feature_expr)


financial_features = fin_wide.select(
    F.col("party_id"),
    F.col("close_date").alias("data_date"),
    *feature_exprs
)


# ============================================================
# 11. LEFT JOIN FEATURES INTO RDS
# ============================================================

final_df = (
    rds
    .join(
        financial_features,
        on=["party_id", "data_date"],
        how="left"
    )
)


# ============================================================
# 12. SAVE OUTPUT TABLE
# ============================================================

(
    final_df
    .write
    .mode("overwrite")
    .saveAsTable(OUT_TABLE)
)
