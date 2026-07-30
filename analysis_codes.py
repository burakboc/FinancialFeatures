comparison.select(
    F.sum(
        F.when(F.col("m.feature_x").isNull(), 1).otherwise(0)
    ).alias("model_nulls"),

    F.sum(
        F.when(F.col("s.raw_feature").isNull(), 1).otherwise(0)
    ).alias("source_nulls"),

    F.sum(
        F.when(
            F.col("m.feature_x").isNull()
            & F.col("s.raw_feature").isNull(),
            1
        ).otherwise(0)
    ).alias("both_null")
).show()


comparison_202006 = comparison.filter(
    F.date_format("data_date", "yyyy-MM") == "2020-06"
)
