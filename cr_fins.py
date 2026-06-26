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
