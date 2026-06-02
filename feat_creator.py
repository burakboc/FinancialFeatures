import yaml
import pandas as pd
from functools import reduce
from pyspark.sql import functions as F

# ============================================================
# PARAMETERS
# ============================================================

YAML_PATH = "/path/to/feature_config.yml"
INFLATION_XLSX_PATH = "/path/to/inflation_adj.xlsx"

FIN_TABLE = "analytics_risk_sb.financial_features_sme_limit_prometeia_oy_v2"
RDS_TABLE = "risk_analytics_sb.ala_target_fin"
OUT_TABLE = "analytics_risk_sb.financial_features_all"

KEYS = ["party_id", "close_date"]


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
    """
    Calculates:
      sum(pos_cols) - sum(neg_cols)
    """
    pos_exprs = [F.coalesce(F.col(c), F.lit(0.0)) for c in pos_cols]
    neg_exprs = [F.coalesce(F.col(c), F.lit(0.0)) for c in neg_cols]

    pos_sum = reduce(lambda a, b: a + b, pos_exprs) if pos_exprs else F.lit(0.0)
    neg_sum = reduce(lambda a, b: a + b, neg_exprs) if neg_exprs else F.lit(0.0)

    return pos_sum - neg_sum


def parse_feature_config(yaml_path):
    """
    Expected YAML format:

    YKB_F15:
      numerator:
        - profit_before_int_tax_9044
      denominator:
        - net_sales_80
      neg_denominator:
        - net_sales_80_1y
    """
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
# 2. READ FINANCIAL TABLE
# ============================================================

fin_cols_to_select = KEYS + ["prev_financial_indicator"] + required_fin_cols

fin = (
    spark.table(FIN_TABLE)
    .select(*fin_cols_to_select)
    .withColumn("close_date", F.to_date(F.col("close_date")))
)


# ============================================================
# 3. GET CURRENT FINANCIALS: prev_financial_indicator = 0
# ============================================================

fin_0 = (
    fin
    .filter(F.col("prev_financial_indicator") == 0)
    .select(*KEYS, *current_cols)
)


# ============================================================
# 4. GET LAST-YEAR FINANCIALS: prev_financial_indicator = 1
# ============================================================

fin_1_select_exprs = KEYS + [
    F.col(c).alias(f"{c}_1y")
    for c in last_year_base_cols
]

fin_1 = (
    fin
    .filter(F.col("prev_financial_indicator") == 1)
    .select(*fin_1_select_exprs)
)


# ============================================================
# 5. COMBINE CURRENT + LAST-YEAR FINANCIALS
# ============================================================

fin_wide = (
    fin_0
    .join(fin_1, on=KEYS, how="left")
)


# ============================================================
# 6. READ INFLATION EXCEL
# ============================================================

infl_pd = pd.read_excel(INFLATION_XLSX_PATH)

# Expected Excel columns:
# date | inflation_rate
infl_pd.columns = [c.strip().lower() for c in infl_pd.columns]

infl_spark = (
    spark.createDataFrame(infl_pd)
    .withColumn("inflation_date", F.to_date(F.col("date")))
    .withColumn("inflation_rate", F.col("inflation_rate").cast("double"))
    .select("inflation_date", "inflation_rate")
)


# ============================================================
# 7. JOIN INFLATION RATE
# Uses end of month, one month before close_date
# ============================================================

fin_wide = (
    fin_wide
    .withColumn(
        "inflation_date",
        F.last_day(F.add_months(F.col("close_date"), -1))
    )
    .join(infl_spark, on="inflation_date", how="left")
)


# ============================================================
# 8. APPLY INFLATION ADJUSTMENT TO RAW FINANCIAL COLUMNS
# ============================================================

raw_cols_to_adjust = [
    c for c in fin_wide.columns
    if c not in KEYS + ["inflation_date", "inflation_rate"]
]

for c in raw_cols_to_adjust:
    fin_wide = fin_wide.withColumn(
        c,
        F.col(c).cast("double") * F.col("inflation_rate")
    )


# ============================================================
# 9. CREATE FEATURES FROM YAML
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
    *KEYS,
    *feature_exprs
)


# ============================================================
# 10. READ RDS AND LEFT JOIN FEATURES INTO IT
# ============================================================

rds = (
    spark.table(RDS_TABLE)
    .withColumn("close_date", F.to_date(F.col("close_date")))
)

final_df = (
    rds
    .join(financial_features, on=KEYS, how="left")
)


# ============================================================
# 11. SAVE OUTPUT TABLE
# ============================================================

(
    final_df
    .write
    .mode("overwrite")
    .saveAsTable(OUT_TABLE)
)
