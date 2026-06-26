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
    cr_rds
    .withColumn("party_id_join", F.col("party_id").cast("decimal(16,0)"))
    .withColumn("month_start_join", F.to_date("month_start", "yyyy-MM-dd"))
    .alias("r")
    .join(
        fin
        .withColumn("party_id_join", F.col("party_id").cast("decimal(16,0)"))
        .withColumn("data_date_join", F.col("data_date").cast("date"))
        .alias("f"),
        on=[
            F.col("r.party_id_join") == F.col("f.party_id_join"),
            F.col("r.month_start_join") == F.col("f.data_date_join")
        ],
        how="left"
    )
    .select(
        "r.party_id",
        "r.month_start",
        "r.proposal_id",
        *[F.col(f"f.{c}").alias(c) for c in fin.columns if c != "party_id"]
    )
)
