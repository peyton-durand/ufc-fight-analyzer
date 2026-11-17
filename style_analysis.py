"""
style_analysis.py: fighter style clustering with Spark MLlib K-means (runs locally, local[*]).

Usage:
    python style_analysis.py --corr                      # Pearson correlation of the candidate features
    python style_analysis.py --eval                      # fit k = 2..10, print WSSSE / silhouette, save k_selection_<set>.csv
    python style_analysis.py --k 5                       # fit the final model, print clusters, write fighter_style_results.csv
    python style_analysis.py --features v1 --k 5         # same with the v1 feature set (default is v2)
Each --k run appends a row to data/style_output/model_runs.csv so feature sets and k values can be compared.
"""
import argparse
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

# Run Spark's Python workers with this interpreter (the venv), not whatever "python" is first on PATH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.stat import Correlation

CLEAN = "data/clean"
OUT = "data/style_output"
REFERENCE = "data/reference_styles.csv"
SEED = 42
MIN_FIGHTS = 3

# Labels only hold for the k they were named at (v1: k=5).
STYLE_LABELS = {
    "v1": {0: "Kickboxer", 1: "Clinch Fighter", 2: "Wrestle-Boxer", 3: "Volume Striker", 4: "Control Grappler"},
    "v2": {0: "Measured Counter-Fighter", 1: "Volume Striker", 2: "Wrestler", 3: "Submission Scrambler"},
}

# Generic stat name -> clean_round.csv column, with {p} the corner prefix ("r" or "b")
STAT_COLUMNS = {
    "kd": "{p}_kd",
    "sig_atmp": "{p}_sig_atmp",
    "total_atmp": "{p}_total_str_atmp",
    "head_atmp": "{p}_sig_str_atmp_head",
    "body_atmp": "{p}_sig_str_atmp_body",
    "leg_atmp": "{p}_sig_str_atmp_leg",
    "distance_atmp": "{p}_sig_str_atmp_distance",
    "clinch_atmp": "{p}_sig_str_atmp_clinch",
    "ground_atmp": "{p}_sig_str_atmp_ground",
    "td_atmp": "{p}_td_atmp",
    "sub_att": "{p}_sub_att",
    "rev": "{p}_rev",
    "ctrl_sec": "{p}_ctrl_sec",
}

# Every feature is computed and saved, whichever set the model uses
FEATURES = [
    "sig_atmp_per_min", "output_share", "nonsig_share", "kd_per_15",
    "head_share", "body_share", "leg_share",
    "distance_share", "clinch_share", "ground_share",
    "td_atmp_per_15", "ctrl_share", "sub_att_per_15", "rev_per_15",
]

FEATURE_SETS = {
    # Target and position shares; each share group sums to 1, so leg_share and ground_share are left out as redundant
    "v1": ["sig_atmp_per_min", "head_share", "body_share", "distance_share", "clinch_share",
           "td_atmp_per_15", "ctrl_share", "sub_att_per_15"],
    # 4 striking + 4 grappling
    "v2": ["sig_atmp_per_min", "clinch_share", "output_share", "kd_per_15",
           "td_atmp_per_15", "ctrl_share", "sub_att_per_15", "rev_per_15"],
}

# Candidates for --corr: the v2 features plus the ones v2 left out
CORR_FEATURES = FEATURE_SETS["v2"] + ["head_share", "body_share", "leg_share", "distance_share", "ground_share", "nonsig_share"]
CORR_THRESHOLD = 0.7


@contextmanager
def timed(stage):
    start = time.perf_counter()
    yield
    print(f"[{stage}] {time.perf_counter() - start:.1f}s")


def get_spark():
    spark = (SparkSession.builder.master("local[*]").appName("ufc-style")
             .config("spark.sql.shuffle.partitions", "8")   # the data is small; fewer partitions is faster
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# Feature prep ----------------------------------------------------------------------------------------
def load_rounds(spark):
    rounds = spark.read.csv(f"{CLEAN}/clean_round.csv", header=True, inferSchema=True)
    tracked = rounds.filter(F.col("r_ctrl_sec").isNotNull() & F.col("b_ctrl_sec").isNotNull())
    return rounds, tracked


def build_features(rounds):
    # Map: one row per (fighter_id, fight_id, round) with generic column names, plus the opponent's sig attempted
    def corner(p, opp):
        return rounds.select(
            F.col(f"{p}_id").alias("fighter_id"), "fight_id", F.col("round_no").alias("round"), "round_seconds",
            *[F.col(col.format(p=p)).alias(name) for name, col in STAT_COLUMNS.items()],
            F.col(f"{opp}_sig_atmp").alias("opp_sig_atmp"),
        )

    per_round = corner("r", "b").unionByName(corner("b", "r"))

    # Reduce: one row per fighter
    totals = per_round.groupBy("fighter_id").agg(
        F.countDistinct("fight_id").alias("fights"),
        F.sum("round_seconds").alias("seconds"),
        *[F.sum(name).alias(name) for name in [*STAT_COLUMNS, "opp_sig_atmp"]],
    )
    totals = totals.filter((F.col("fights") >= MIN_FIGHTS) & (F.col("sig_atmp") > 0))

    minutes, per_15 = F.col("seconds") / 60, F.col("seconds") / 900
    share = lambda name: F.col(f"{name}_atmp") / F.col("sig_atmp")
    features = totals.select(
        "fighter_id", "fights", "seconds",
        (F.col("sig_atmp") / minutes).alias("sig_atmp_per_min"),
        (F.col("sig_atmp") / (F.col("sig_atmp") + F.col("opp_sig_atmp"))).alias("output_share"),
        ((F.col("total_atmp") - F.col("sig_atmp")) / F.col("total_atmp")).alias("nonsig_share"),
        (F.col("kd") / per_15).alias("kd_per_15"),
        *[share(part).alias(f"{part}_share") for part in ["head", "body", "leg", "distance", "clinch", "ground"]],
        (F.col("td_atmp") / per_15).alias("td_atmp_per_15"),
        (F.col("ctrl_sec") / F.col("seconds")).alias("ctrl_share"),
        (F.col("sub_att") / per_15).alias("sub_att_per_15"),
        (F.col("rev") / per_15).alias("rev_per_15"),
    )
    return features.select("fighter_id", "fights", "seconds", *FEATURES)


def write_features(features):
    path = f"{OUT}/fighter_features.parquet"
    if os.name != "nt" or os.environ.get("HADOOP_HOME"):
        features.write.mode("overwrite").parquet(path)
        print(f"Wrote {path} (Spark parquet writer)")
    else:
        features.toPandas().to_parquet(path, engine="pyarrow", index=False)
        print(f"Wrote {path} (pandas + pyarrow: no HADOOP_HOME on Windows)")


# Correlation -----------------------------------------------------------------------------------------
def correlation(features):
    assembled = VectorAssembler(inputCols=CORR_FEATURES, outputCol="candidates").transform(features)
    matrix = Correlation.corr(assembled, "candidates", "pearson").head()[0].toArray()
    corr = pd.DataFrame(matrix, index=CORR_FEATURES, columns=CORR_FEATURES)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n" + corr.round(2).to_string())

    pairs = [(a, b, corr.at[a, b]) for i, a in enumerate(CORR_FEATURES) for b in CORR_FEATURES[i + 1:]
             if abs(corr.at[a, b]) > CORR_THRESHOLD]
    print(f"\nPairs with |r| > {CORR_THRESHOLD}:")
    for a, b, r in sorted(pairs, key=lambda p: -abs(p[2])):
        print(f"  {a:<18} {b:<18} {r:+.2f}")
    if not pairs:
        print("  none")
    return corr


# Model -----------------------------------------------------------------------------------------------
def scale(features, model_features):
    assembled = VectorAssembler(inputCols=model_features, outputCol="raw").transform(features)
    scaler = StandardScaler(inputCol="raw", outputCol="scaled", withMean=True, withStd=True).fit(assembled)
    return scaler.transform(assembled)


def fit(scaled, k):
    return KMeans(k=k, seed=SEED, featuresCol="scaled", predictionCol="cluster").fit(scaled)


def silhouette(clustered):
    return ClusteringEvaluator(featuresCol="scaled", predictionCol="cluster",
                               metricName="silhouette", distanceMeasure="squaredEuclidean").evaluate(clustered)


def evaluate_k(scaled, ks):
    rows = []
    for k in ks:
        start = time.perf_counter()
        model = fit(scaled, k)
        seconds = time.perf_counter() - start
        rows.append({"k": k, "wssse": model.summary.trainingCost,
                     "silhouette": silhouette(model.transform(scaled)), "fit_seconds": seconds})
    return pd.DataFrame(rows)


def labeller(feature_set):
    labels = STYLE_LABELS.get(feature_set, {})
    return lambda cluster: labels.get(cluster, f"Cluster {cluster}")


def fighter_names(spark):
    return spark.read.csv(f"{CLEAN}/clean_fighter.csv", header=True).select("fighter_id", "fighter_name")


def describe_clusters(clustered, names, label):
    centers = clustered.groupBy("cluster").agg(F.count("*").alias("size"), *[F.avg(f).alias(f) for f in FEATURES])
    named = clustered.join(names, "fighter_id")

    for c in centers.orderBy("cluster").collect():
        print(f"\n{label(c['cluster'])} (cluster {c['cluster']}): {c['size']} fighters")
        print(f"  striking  sig attempted/min {c['sig_atmp_per_min']:.1f}   output share {c['output_share']:.0%}   "
              f"knockdowns/15 {c['kd_per_15']:.2f}   non-sig {c['nonsig_share']:.0%}")
        print(f"  grappling td attempted/15 {c['td_atmp_per_15']:.2f}   control {c['ctrl_share']:.0%}   "
              f"sub attempts/15 {c['sub_att_per_15']:.2f}   reversals/15 {c['rev_per_15']:.2f}")
        print(f"  target    head {c['head_share']:.0%}  body {c['body_share']:.0%}  leg {c['leg_share']:.0%}")
        print(f"  position  distance {c['distance_share']:.0%}  clinch {c['clinch_share']:.0%}  ground {c['ground_share']:.0%}")

        members = named.filter(F.col("cluster") == c["cluster"])
        top = members.orderBy(F.desc("fights"), "fighter_name").limit(5).collect()
        print("  e.g.    " + ", ".join(f"{r['fighter_name']} ({r['fights']})" for r in top))


# Reference scoring -----------------------------------------------------------------------------------
# Cross-tab of expected style x cluster, and purity: for each cluster, the count of its most common expected style,
# summed and divided by the number of reference fighters that were clustered
def score_reference(clustered, names, label):
    ref = pd.read_csv(REFERENCE)
    known = names.toPandas()
    result = clustered.select("fighter_id", "cluster", "fights").toPandas()

    found = ref.merge(known, on="fighter_name", how="left")
    missing = found.loc[found["fighter_id"].isna(), "fighter_name"].tolist()
    found = found.merge(result, on="fighter_id", how="left")
    # A name shared by two fighters: keep the one with the most clustered fights
    found = found.sort_values("fights", ascending=False).drop_duplicates("fighter_name")
    below = sorted(set(found.loc[found["fighter_id"].notna() & found["cluster"].isna(), "fighter_name"]))
    scored = found[found["cluster"].notna()].copy()
    scored["cluster"] = scored["cluster"].astype(int).map(lambda c: f"{c} {label(c)}")

    print(f"\nReference fighters: {len(scored)} of {len(ref)} clustered")
    if missing:
        print(f"  not found in clean_fighter.csv: {', '.join(missing)}")
    if below:
        print(f"  under the {MIN_FIGHTS}-fight cutoff (rounds with control time): {', '.join(below)}")

    table = pd.crosstab(scored["expected_style"], scored["cluster"])
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n" + table.to_string())
    purity = table.max(axis=0).sum() / len(scored) if len(scored) else float("nan")
    print(f"\nPurity: {purity:.3f}")
    return purity


def log_run(row):
    path = f"{OUT}/model_runs.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"Appended to {path}")


def write_results(clustered, feature_set, label):
    path = f"{OUT}/fighter_style_results.csv"
    result = (clustered.select("fighter_id", "cluster", "fights", *FEATURES)
                       .orderBy("cluster", F.desc("fights"), "fighter_id").toPandas())
    result.insert(1, "feature_set", feature_set)
    result.insert(3, "label", result["cluster"].map(label))
    result.to_csv(path, index=False)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--corr", action="store_true", help="Pearson correlation of the candidate features")
    mode.add_argument("--eval", action="store_true", help="fit k = 2..10 and compare WSSSE and silhouette")
    mode.add_argument("--k", type=int, help="fit the final model with k clusters")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="v2", help="feature set for the model (default v2)")
    args = ap.parse_args()

    spark = get_spark()
    os.makedirs(OUT, exist_ok=True)
    model_features, label = FEATURE_SETS[args.features], labeller(args.features)

    # Spark is lazy: each stage is cached and counted so its timing measures that stage's work
    with timed("load"):
        rounds, tracked = load_rounds(spark)
        tracked = tracked.cache()
        n_tracked = tracked.count()
    print(f"rounds: {rounds.count()}, with control time: {n_tracked}")

    with timed("feature build"):
        features = build_features(tracked).cache()
        n_fighters = features.count()
    print(f"fighters with >= {MIN_FIGHTS} fights and significant strikes attempted: {n_fighters}")

    if args.corr:
        with timed("correlation"):
            corr = correlation(features)
        with timed("writing"):
            corr.to_csv(f"{OUT}/feature_correlation.csv")
            print(f"Wrote {OUT}/feature_correlation.csv")
        return

    print(f"model features, {args.features} ({len(model_features)}): {', '.join(model_features)}")
    with timed("scaling"):
        scaled = scale(features, model_features).cache()
        scaled.count()

    if args.eval:
        with timed("fitting"):
            table = evaluate_k(scaled, range(2, 11))
        print("\n" + table.to_string(index=False, formatters={
            "wssse": "{:.1f}".format, "silhouette": "{:.3f}".format, "fit_seconds": "{:.1f}".format}))
        with timed("writing"):
            write_features(features)
            table.to_csv(f"{OUT}/k_selection_{args.features}.csv", index=False)
            print(f"Wrote {OUT}/k_selection_{args.features}.csv")
        return

    with timed("fitting"):
        start = time.perf_counter()
        model = fit(scaled, args.k)
        fit_seconds = time.perf_counter() - start
        clustered = model.transform(scaled).cache()
        clustered.count()

    names = fighter_names(spark).cache()
    describe_clusters(clustered, names, label)
    purity = score_reference(clustered, names, label)
    wssse, sil = model.summary.trainingCost, silhouette(clustered)
    print(f"WSSSE {wssse:.1f}   silhouette {sil:.3f}\n")

    with timed("writing"):
        write_features(features)
        write_results(clustered, args.features, label)
        log_run({"timestamp": datetime.now().isoformat(timespec="seconds"), "feature_set": args.features, "k": args.k,
                 "wssse": round(wssse, 3), "silhouette": round(sil, 4), "purity": round(purity, 4),
                 "fit_seconds": round(fit_seconds, 2)})


if __name__ == "__main__":
    main()
