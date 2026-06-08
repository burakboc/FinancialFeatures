import yaml
from functools import reduce
from pyspark.sql import functions as F

# ============================================================
# CONFIG
# ============================================================

raw_table = "YOUR_RAW_FINANCIALS_TABLE"
out_table = "analytics_risk_sb.ala_prom_financials_all"
yml_path = "longlist.yml"

mapping = {
    "P_30": "financial_liability_30",
    "P_40": "financial_liability_40",
    "A_10": "liquid_asset_10",
    "A_1": "revolving_asset_1",
    "A_2": "fixed_asset_2",
    "G_62": "sales_cost_62",
    "P_32": "commercial_liability_32",
    "P_42": "commercial_liability_42",
    "P_43": "other_liability_43",
    "P_3": "short_term_foreign_resource_3",
    "P_5": "equities_5",
    "P_4": "long_term_foreign_resource_4",
    "A_15": "inventory_15",
    "A_26": "nontangible_asset_26",
    "A_25": "tangible_asset_25",
    "G_66": "financial_expense_66",
    "A_12": "commercial_receivable_12",
    "A_22": "commercial_receivable_22",
    "P_35": "annual_construction_payment_35",
    "P_34": "advanced_received_34",
    "A_17": "annual_construction_cost_17",
    "G_60": "gross_sales_60",
    "A_11": "security_11",
}

# ============================================================
# HELPERS
# ============================================================

def q(col_name):
    return f"`{col_name}`"

def c(col_name):
    return F.col(col_name)

def safe_div(num, den):
    return (
        F.when(den.isNull() | (den == 0), F.lit(None).cast("double"))
        .otherwise(num / den)
    )

def sum_cols(cols):
    if not cols:
        return F.lit(0.0)

    return reduce(
        lambda a, b: a + b,
        [F.coalesce(c(x), F.lit(0.0)) for x in cols]
    )

def feature_raw_dependencies(feature_name, longlist, visited=None):
    if visited is None:
        visited = set()

    if feature_name in visited:
        return set()

    visited.add(feature_name)

    spec = longlist[feature_name]
    func = spec["func"]
    args = spec.get("args", {})

    if func == "calculate_financial_aggregate":
        return (
            set(args.get("added_accounts", []) or [])
            | set(args.get("subtracted_accounts", []) or [])
        )

    if func == "calculate_custom_financial_ratio":
        return (
            set(args.get("required_columns", []) or [])
            | set(args.get("columns_to_shift", []) or [])
        )

    if func == "calculate_financial_ratio":
        deps = set()

        for parent_feature in [
            args["numerator_column_name"],
            args["denominator_column_name"],
        ]:
            if parent_feature in longlist:
                deps |= feature_raw_dependencies(parent_feature, longlist, visited)
            else:
                deps.add(parent_feature)

        return deps

    return set()

# ============================================================
# LOAD YAML
# ============================================================

with open(yml_path, "r") as f:
    longlist = yaml.safe_load(f)

needed_codes = set()
cols_to_prev = set()

for feature_name, spec in longlist.items():
    args = spec.get("args", {})

    needed_codes.update(args.get("added_accounts", []) or [])
    needed_codes.update(args.get("subtracted_accounts", []) or [])
    needed_codes.update(args.get("required_columns", []) or [])

    prev_cols = args.get("columns_to_shift", []) or []
    needed_codes.update(prev_cols)
    cols_to_prev.update(prev_cols)
    
# ============================================================
# CHECK AVAILABLE RAW COLUMNS - CASE INSENSITIVE
# ============================================================

raw_table_columns = spark.table(raw_table).columns

raw_col_lookup = {
    col_name.lower(): col_name
    for col_name in raw_table_columns
}

def resolve_raw_col(col_name):
    return raw_col_lookup.get(col_name.lower())

required_key_cols = ["party_id", "data_date", "prev_financial_indicator"]

missing_key_cols = [
    col_name
    for col_name in required_key_cols
    if resolve_raw_col(col_name) is None
]

if missing_key_cols:
    raise ValueError(f"Missing required key columns in raw table: {missing_key_cols}")

actual_party_id_col = resolve_raw_col("party_id")
actual_data_date_col = resolve_raw_col("data_date")
actual_prev_indicator_col = resolve_raw_col("prev_financial_indicator")

available_codes = set()
missing_code_to_raw_col = {}
code_to_actual_raw_col = {}

for code in needed_codes:
    mapped_raw_col = mapping.get(code, code)
    actual_raw_col = resolve_raw_col(mapped_raw_col)

    if actual_raw_col is not None:
        available_codes.add(code)
        code_to_actual_raw_col[code] = actual_raw_col
    else:
        missing_code_to_raw_col[code] = mapped_raw_col

calculable_features = []
skipped_features = {}

for feature_name in longlist.keys():
    deps = feature_raw_dependencies(feature_name, longlist)
    missing_deps = sorted(deps - available_codes)

    if missing_deps:
        skipped_features[feature_name] = missing_deps
    else:
        calculable_features.append(feature_name)

print("=" * 100)
print(f"Available raw codes: {len(available_codes)}")
print(f"Missing raw codes: {len(missing_code_to_raw_col)}")
print(f"Calculable features: {len(calculable_features)}")
print(f"Skipped features: {len(skipped_features)}")

print("\nAvailable code -> actual raw table column:")
for code in sorted(available_codes):
    print(f"  {code} -> {code_to_actual_raw_col[code]}")

print("\nMissing raw codes / expected raw table columns:")
for code, raw_col in sorted(missing_code_to_raw_col.items()):
    print(f"  {code} -> {raw_col}")

print("=" * 100)

needed_codes = needed_codes & available_codes
cols_to_prev = cols_to_prev & available_codes

raw_cols = {
    code: code_to_actual_raw_col[code]
    for code in needed_codes
}

# ============================================================
# READ RAW DATA FROM HIVE
# ============================================================

column_names = [
    actual_party_id_col,
    actual_data_date_col,
    actual_prev_indicator_col,
] + sorted(set(raw_cols.values()))

column_names_sql = ", ".join(q(x) for x in column_names)

raw = spark.sql(f"""
    SELECT {column_names_sql}
    FROM {raw_table}
    WHERE {q(actual_prev_indicator_col)} IN (0, 1)
""")

# ============================================================
# INFLATION ADJUSTMENT
# ============================================================

inflation_table = "src_edwlive_dm_cad.mva_inflation_rate_new"

inflation_df = (
    spark.table(inflation_table)
    .select(
        F.col("year_month").cast("int").alias("inflation_year_month"),
        F.col("inflation_rate").cast("double").alias("inflation_rate")
    )
)

raw = (
    raw
    .withColumn(
        "year_month",
        F.date_format(
            F.coalesce(
                F.to_date(F.col("data_date"), "yyyyMMdd"),
                F.to_date(F.col("data_date"), "yyyy-MM-dd"),
                F.to_date(F.col("data_date"), "dd.MM.yyyy")
            ),
            "yyyyMM"
        ).cast("int")
    )
    .join(
        inflation_df,
        F.col("year_month") == F.col("inflation_year_month"),
        "left"
    )
)

for raw_col in sorted(set(raw_cols.values())):
    raw = raw.withColumn(
        raw_col,
        F.when(
            F.col("inflation_rate").isNull() | (F.col("inflation_rate") == 0),
            F.col(raw_col).cast("double")
        ).otherwise(
            F.col(raw_col).cast("double") / F.col("inflation_rate")
        )
    )

raw = raw.drop("year_month", "inflation_year_month", "inflation_rate")

select_exprs = [
    F.col(actual_party_id_col).cast("long").alias("party_id"),
    F.coalesce(
        F.to_date(F.col(actual_data_date_col), "yyyyMMdd"),
        F.to_date(F.col(actual_data_date_col), "yyyy-MM-dd"),
        F.to_date(F.col(actual_data_date_col), "dd.MM.yyyy")
    ).alias("data_date"),
    F.col(actual_prev_indicator_col).cast("int").alias("prev_financial_indicator"),
]

for code in sorted(needed_codes):
    select_exprs.append(F.col(raw_cols[code]).cast("double").alias(code))

df = raw.select(*select_exprs)

# ============================================================
# CURRENT + PREVIOUS FINANCIAL ROW JOIN
# ============================================================

curr_df = (
    df
    .filter(F.col("prev_financial_indicator") == 0)
    .drop("prev_financial_indicator")
    .dropDuplicates(["party_id", "data_date"])
)

prev_select = [
    F.col("party_id"),
    F.col("data_date"),
]

for col_name in sorted(cols_to_prev):
    prev_select.append(F.col(col_name).alias(f"{col_name}_prev"))

prev_df = (
    df
    .filter(F.col("prev_financial_indicator") == 1)
    .select(*prev_select)
    .dropDuplicates(["party_id", "data_date"])
)

base_df = curr_df.join(prev_df, on=["party_id", "data_date"], how="left")

# ============================================================
# 1) FINANCIAL AGGREGATES
# ============================================================

feature_df = base_df.select("party_id", "data_date")

for feature_name in calculable_features:
    spec = longlist[feature_name]

    if spec["func"] != "calculate_financial_aggregate":
        continue

    args = spec["args"]
    added = args.get("added_accounts", []) or []
    subtracted = args.get("subtracted_accounts", []) or []

    feature_df = feature_df.withColumn(
        feature_name,
        (sum_cols(added) - sum_cols(subtracted)).cast("double")
    )

# ============================================================
# 2) STANDARD FINANCIAL RATIOS
# ============================================================

for feature_name in calculable_features:
    spec = longlist[feature_name]

    if spec["func"] != "calculate_financial_ratio":
        continue

    args = spec["args"]

    numerator_col = args["numerator_column_name"]
    denominator_col = args["denominator_column_name"]

    feature_df = feature_df.withColumn(
        feature_name,
        safe_div(c(numerator_col), c(denominator_col)).cast("double")
    )

# ============================================================
# 3) CUSTOM FINANCIAL RATIOS
# ============================================================

joined = feature_df.join(base_df, on=["party_id", "data_date"], how="left")

custom_exprs = {
    "DC_31": safe_div(
        (
            c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
            + c("G_692_prev") + c("G_66_prev") + c("G_691_prev") + c("Amortisman_ve_Itfa_Giderleri_prev")
        ) / 2,
        c("P_3") - c("P_31")
    ),

    "EF_13": safe_div(
        c("G_0") + c("A_15") - c("A_15_prev"),
        c("A_15")
    ),

    "EF_14": safe_div(
        c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        + c("A_15") - c("A_15_prev"),
        c("A_15")
    ),

    "EF_17": safe_div(
        365 * (c("A_15") + c("A_15_prev")) / 2,
        c("G_62")
    ),

    "EF_19": safe_div(
        365 * ((c("P_32") + c("P_42") + c("P_32_prev") + c("P_42_prev")) / 2),
        c("G_62")
    ),

    "EF_20": safe_div(
        365 * ((c("A_12") + c("A_22") + c("A_12_prev") + c("A_22_prev")) / 2),
        c("G_600") + c("G_601") + c("G_602") - (c("G_610") + c("G_611") + c("G_612"))
    ),

    "FS_12": safe_div(
        c("P_30") - c("P_30_prev") + c("P_40") - c("P_40_prev"),
        c("A_1") + c("A_2")
    ),

    "FS_53_CONT": safe_div(
        (
            c("G_600") + c("G_601") + c("G_602")
            - (c("G_610") + c("G_611") + c("G_612"))
            + c("P_35") + c("P_35_prev")
        ) / 2 + c("P_34"),
        (c("A_17") + c("A_17_prev")) / 2 + c("A_15") + c("G_62")
    ),

    "FS_55_CONT": safe_div(
        (c("P_35") + c("P_35_prev")) / 2,
        (c("A_17") + c("A_17_prev")) / 2
    ),

    "FS_56_CONT": safe_div(
        (
            c("G_600") + c("G_601") + c("G_602")
            - (c("G_610") + c("G_611") + c("G_612"))
            + c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
            + c("P_35") + c("P_35_prev")
        ) / 2,
        (c("A_17") + c("A_17_prev") + c("G_62") + c("G_62_prev")) / 2
    ),

    "LQ_13": safe_div(
        c("A_1") - c("P_3"),
        c("G_60") + c("A_15") - c("A_15_prev")
    ),

    "LQ_14": safe_div(
        c("A_1") - c("P_3"),
        c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        + c("A_15") - c("A_15_prev")
    ),

    "LQ_50": safe_div(
        c("A_12") + c("A_10") + c("A_11") + (c("A_15") + c("A_15_prev")) / 2,
        c("P_3") - c("P_331")
    ),

    "PROF_32": safe_div(
        c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        - (c("A_25") - c("A_25_prev") + c("Amortisman_ve_Itfa_Giderleri")),
        c("G_66") - c("G_642")
    ),

    "PROF_33": safe_div(
        c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        - (c("A_25") - c("A_25_prev") + c("Amortisman_ve_Itfa_Giderleri")),
        c("P_30") + c("P_40") + c("G_66") - c("A_10")
    ),

    "TR_01": safe_div(c("G_60") - c("G_60_prev"), c("G_60_prev")),
    "TR_04": safe_div(c("G_692") - c("G_692_prev"), c("G_692_prev")),

    "TR_05": safe_div(
        c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        - (c("G_692_prev") + c("G_66_prev") + c("G_691_prev") + c("Amortisman_ve_Itfa_Giderleri_prev")),
        c("G_692_prev") + c("G_66_prev") + c("G_691_prev") + c("Amortisman_ve_Itfa_Giderleri_prev")
    ),

    "TR_06": safe_div(c("P_5") - c("P_5_prev"), c("P_5_prev")),
    "TR_07": safe_div(c("P_3") - c("P_3_prev"), c("P_3_prev")),

    "TR_08": safe_div(
        c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        + c("G_62")
        - (
            c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
            + c("G_62_prev")
        ),
        c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
        - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
        + c("G_62_prev")
    ),

    "TR_09": safe_div(
        c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        - (
            c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
        ),
        c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
        - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
    ),

    "TR_10": safe_div(
        c("A_1") + c("A_2") - (c("A_1_prev") + c("A_2_prev")),
        c("A_1_prev") + c("A_2_prev")
    ),

    "TR_11": safe_div(
        c("A_1") + c("P_3") - (c("A_1_prev") + c("P_3_prev")),
        c("A_1_prev") + c("P_3_prev")
    ),

    "TR_12": safe_div(c("P_4") - c("P_4_prev"), c("P_4_prev")),
    "TR_13": safe_div(c("A_25") - c("A_25_prev"), c("A_25_prev")),
    "TR_14": safe_div(c("A_2") - c("A_2_prev"), c("A_2_prev")),

    "TR_15": safe_div(
        c("A_12")
        - (c("G_600") + c("G_601") + c("G_602") - (c("G_610") + c("G_611") + c("G_612")))
        - (
            c("A_12_prev")
            - (
                c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
                - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
            )
        ),
        c("A_12_prev")
        - (
            c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
        )
    ),

    "TR_17": safe_div(
        c("P_30") + c("P_40") - (c("P_30") + c("P_40_prev")),
        c("P_30_prev") + c("P_40_prev")
    ),

    "TR_18": safe_div(c("P_30") - c("P_30_prev"), c("P_30_prev")),
    "TR_19": safe_div(c("P_40") - c("P_40_prev"), c("P_40_prev")),

    "CONST_06": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri"),
        c("G_66")
    ),

    "CONST_07": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        - (c("A_12") - c("A_12_prev"))
        + (c("P_32") - c("P_32_prev")),
        c("P_30") + c("G_66")
    ),

    "CONST_09": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri"),
        c("P_3") + c("P_40") + c("P_42") + c("P_43")
    ),

    "CONST_11": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691"),
        c("P_3") + c("P_40") + c("P_42") + c("P_43") - c("P_35")
    ),

    "CONST_17": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        + c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        - (
            c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
        ),
        c("G_66")
    ),

    "CONST_18": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        + c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        - (
            c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
        )
        - (c("A_12") - c("A_12_prev"))
        + (c("P_32") - c("P_32_prev")),
        c("P_30") + c("G_66")
    ),

    "CONST_20": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        + c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        - (
            c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
        ),
        c("P_3") + c("P_40") + c("P_42") + c("P_43")
    ),

    "CONST_22": safe_div(
        c("P_35") - c("A_17")
        - (c("P_35_prev") - c("A_17_prev"))
        + c("G_692") + c("G_66") + c("G_691") + c("Amortisman_ve_Itfa_Giderleri")
        + c("G_600") + c("G_601") + c("G_602")
        - (c("G_610") + c("G_611") + c("G_612"))
        - (
            c("G_600_prev") + c("G_601_prev") + c("G_602_prev")
            - (c("G_610_prev") + c("G_611_prev") + c("G_612_prev"))
        ),
        c("P_3") + c("P_40") + c("P_42") + c("P_43") - c("P_35")
    ),

    "CONST_FS_12": safe_div(
        c("P_30") - c("P_30_prev") + c("P_40") - c("P_40_prev"),
        c("A_1") + c("A_2") - c("A_17")
    ),

    "CONST_TR_02": safe_div(
        c("A_1") + c("A_2") - c("A_17")
        - (c("A_1_prev") + c("A_2_prev") - c("A_17_prev")),
        c("A_1_prev") + c("A_2_prev") - c("A_17_prev")
    ),
}

for feature_name in calculable_features:
    spec = longlist[feature_name]

    if spec["func"] != "calculate_custom_financial_ratio":
        continue

    if feature_name not in custom_exprs:
        skipped_features[feature_name] = ["custom formula not implemented"]
        print(f"WARNING: Skipping {feature_name}, custom formula not implemented.")
        continue

    joined = joined.withColumn(feature_name, custom_exprs[feature_name].cast("double"))

# ============================================================
# FINAL SELECT + SAVE TO HIVE
# ============================================================

actually_calculated_features = [
    f for f in calculable_features
    if f in joined.columns
]

final_cols = ["party_id", "data_date"] + actually_calculated_features

final_df = joined.select(*final_cols)

print("=" * 100)
print(f"Final number of features saved: {len(actually_calculated_features)}")
print("Final saved features:")
for f in actually_calculated_features:
    print(f"  {f}")
print("=" * 100)

spark.sql("CREATE DATABASE IF NOT EXISTS analytics_risk_sb")

(
    final_df
    .write
    .mode("overwrite")
    .format("orc")
    .saveAsTable(out_table)
)
