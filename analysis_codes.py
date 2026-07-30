from pyspark.sql import functions as F

df = df.withColumn(
    "year_month",
    F.date_format("data_date", "yyyy-MM")
)

feature_cols = [
    c for c in df.columns
    if c not in ["party_id", "data_date", "year_month", "target"]
]

exprs = []

for c in feature_cols:
    exprs.append(
        (
            F.sum(
                F.when(F.col(c).isNull(), 1).otherwise(0)
            ) / F.count("*")
        ).alias(c)
    )

monthly_nulls = (
    df.groupBy("year_month")
      .agg(*exprs)
      .orderBy("year_month")
)


# COMPARISON

target_month = "2020-06"

# Features contained in monthly_nulls
feature_cols = [
    c for c in monthly_nulls.columns
    if c != "year_month"
]

# Convert wide table:
# year_month | feature_1 | feature_2 | ...
#
# Into long table:
# year_month | feature | null_rate
stack_expression = ", ".join(
    [f"'{c}', `{c}`" for c in feature_cols]
)

monthly_nulls_long = monthly_nulls.select(
    "year_month",
    F.expr(
        f"stack({len(feature_cols)}, {stack_expression}) "
        "as (feature, null_rate)"
    )
)

# Compare target month against all other months
null_comparison = (
    monthly_nulls_long
    .groupBy("feature")
    .agg(
        F.max(
            F.when(
                F.col("year_month") == target_month,
                F.col("null_rate")
            )
        ).alias("target_month_null_rate"),

        F.avg(
            F.when(
                F.col("year_month") != target_month,
                F.col("null_rate")
            )
        ).alias("other_months_avg_null_rate"),

        F.stddev(
            F.when(
                F.col("year_month") != target_month,
                F.col("null_rate")
            )
        ).alias("other_months_std_null_rate"),

        F.min(
            F.when(
                F.col("year_month") != target_month,
                F.col("null_rate")
            )
        ).alias("other_months_min_null_rate"),

        F.max(
            F.when(
                F.col("year_month") != target_month,
                F.col("null_rate")
            )
        ).alias("other_months_max_null_rate")
    )
    .withColumn(
        "difference",
        F.col("target_month_null_rate")
        - F.col("other_months_avg_null_rate")
    )
    .withColumn(
        "absolute_difference",
        F.abs(F.col("difference"))
    )
    .withColumn(
        "z_score",
        F.when(
            F.col("other_months_std_null_rate") > 0,
            F.col("difference")
            / F.col("other_months_std_null_rate")
        )
    )
    .withColumn(
        "within_historical_range",
        (
            F.col("target_month_null_rate")
            >= F.col("other_months_min_null_rate")
        )
        & (
            F.col("target_month_null_rate")
            <= F.col("other_months_max_null_rate")
        )
    )
    .orderBy(F.desc("absolute_difference"))
)

null_comparison.show(100, truncate=False)

null_comparison_pct = null_comparison.select(
    "feature",

    F.round(
        F.col("target_month_null_rate") * 100,
        2
    ).alias("2020_06_null_pct"),

    F.round(
        F.col("other_months_avg_null_rate") * 100,
        2
    ).alias("other_months_avg_null_pct"),

    F.round(
        F.col("difference") * 100,
        2
    ).alias("difference_percentage_points"),

    F.round("z_score", 2).alias("z_score"),

    "within_historical_range"
).orderBy(
    F.desc(F.abs(F.col("difference_percentage_points")))
)

null_comparison_pct.show(100, truncate=False)

# NEIGHBORS

neighbor_months = ["2020-04", "2020-05", "2020-07", "2020-08"]

neighbor_comparison = (
    monthly_nulls_long
    .groupBy("feature")
    .agg(
        F.max(
            F.when(
                F.col("year_month") == target_month,
                F.col("null_rate")
            )
        ).alias("target_month_null_rate"),

        F.avg(
            F.when(
                F.col("year_month").isin(neighbor_months),
                F.col("null_rate")
            )
        ).alias("neighbor_months_avg_null_rate")
    )
    .withColumn(
        "difference",
        F.col("target_month_null_rate")
        - F.col("neighbor_months_avg_null_rate")
    )
    .withColumn(
        "absolute_difference",
        F.abs("difference")
    )
    .orderBy(F.desc("absolute_difference"))
)

neighbor_comparison.select(
    "feature",
    F.round(
        F.col("target_month_null_rate") * 100,
        2
    ).alias("2020_06_null_pct"),
    F.round(
        F.col("neighbor_months_avg_null_rate") * 100,
        2
    ).alias("neighbor_months_avg_null_pct"),
    F.round(
        F.col("difference") * 100,
        2
    ).alias("difference_percentage_points")
).show(100, truncate=False)

# PLOTS

monthly_nulls_pd = (
    monthly_nulls_long
    .orderBy("year_month", "feature")
    .toPandas()
)

# PLOT SPECIFIC 

import pandas as pd
import matplotlib.pyplot as plt

feature_to_plot = "your_feature_name"

plot_df = (
    monthly_nulls_pd[
        monthly_nulls_pd["feature"] == feature_to_plot
    ]
    .copy()
)

plot_df["year_month"] = pd.to_datetime(
    plot_df["year_month"],
    format="%Y-%m"
)

plot_df = plot_df.sort_values("year_month")

plt.figure(figsize=(12, 5))

plt.plot(
    plot_df["year_month"],
    plot_df["null_rate"] * 100,
    marker="o"
)

# Highlight June 2020
target_date = pd.Timestamp("2020-06-01")

target_row = plot_df[
    plot_df["year_month"] == target_date
]

if not target_row.empty:
    target_rate = target_row["null_rate"].iloc[0] * 100

    plt.scatter(
        target_date,
        target_rate,
        s=120,
        zorder=3
    )

    plt.annotate(
        f"2020-06: {target_rate:.2f}%",
        xy=(target_date, target_rate),
        xytext=(10, 15),
        textcoords="offset points"
    )

plt.axvline(
    target_date,
    linestyle="--",
    alpha=0.7,
    label="2020-06"
)

plt.title(f"Monthly Null Rate — {feature_to_plot}")
plt.xlabel("Month")
plt.ylabel("Null Rate (%)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# GET TOP 10 UNUSUALS

top_n = 10

top_features = (
    null_comparison
    .orderBy(F.desc("absolute_difference"))
    .limit(top_n)
    .select("feature")
    .rdd.flatMap(lambda row: row)
    .collect()
)

top_features

# PLOT UNUSUALS

import pandas as pd
import matplotlib.pyplot as plt

target_date = pd.Timestamp("2020-06-01")

for feature in top_features:

    plot_df = (
        monthly_nulls_pd[
            monthly_nulls_pd["feature"] == feature
        ]
        .copy()
    )

    plot_df["year_month"] = pd.to_datetime(
        plot_df["year_month"],
        format="%Y-%m"
    )

    plot_df = plot_df.sort_values("year_month")

    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_df["year_month"],
        plot_df["null_rate"] * 100,
        marker="o"
    )

    plt.axvline(
        target_date,
        linestyle="--",
        alpha=0.7,
        label="2020-06"
    )

    target_row = plot_df[
        plot_df["year_month"] == target_date
    ]

    if not target_row.empty:
        target_rate = target_row["null_rate"].iloc[0] * 100

        plt.scatter(
            target_date,
            target_rate,
            s=120,
            zorder=3
        )

        plt.annotate(
            f"{target_rate:.2f}%",
            xy=(target_date, target_rate),
            xytext=(10, 15),
            textcoords="offset points"
        )

    plt.title(f"Monthly Null Rate — {feature}")
    plt.xlabel("Month")
    plt.ylabel("Null Rate (%)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
