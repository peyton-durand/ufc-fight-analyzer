"""
graph_analysis.py: win-chain analysis of the UFC fight graph with Spark + GraphFrames (runs locally).

Graph: one vertex per fighter, one directed edge winner -> loser for every fight that has a winner.
  1. BFS shortest win chain between two fighters   (reachability)
  2. Connected components                          (how the sport splits into clusters)
  3. Strongly connected components                 (groups where everyone can reach everyone via win chains)

Usage (from the project root, with your venv active):
    python graph_analysis.py --prep-only                       # builds vertices/edges, no GraphFrames needed
    python graph_analysis.py --from "Frankie Edgar" --to "Jon Jones"
    python graph_analysis.py                                   # components only (no BFS)

GraphFrames needs a Spark jar that matches your PySpark major version. The default below is the Spark 4.x
coordinate from the GraphFrames installation docs (graphframes.io, "Installation"); the 0.12.3 jar is built
against Spark 4.1.2, so it is paired here with pyspark==4.1.4 and graphframes-py==0.12.3. If you run
Spark 3.x, set it without editing this file:
    $env:GRAPHFRAMES_PACKAGE = "io.graphframes:graphframes-spark3_2.12:<version>"   (PowerShell)
Check the exact coordinates on the GraphFrames installation page for your Spark/Scala version.

Windows without winutils.exe / HADOOP_HOME: Spark's spark.jars.packages (and spark.jars) copy every jar through
Hadoop's FileUtil.chmod, which needs winutils.exe, so SparkContext fails to start. In that case this script
downloads the same jars from Maven Central (into ~/.cache/graphframes-jars) and puts them on the driver
classpath instead; in local[*] mode the executors run in the driver JVM, so nothing has to be shipped.
If HADOOP_HOME is set (or you are not on Windows) the normal spark.jars.packages route is used.
Checkpoints use Spark local checkpoints for the same reason (sc.setCheckpointDir also needs winutils).
"""
import argparse
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

# Run Spark's Python workers with this interpreter (the venv), not whatever "python" is first on PATH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, Window, functions as F

CLEAN = "data/clean"
OUT = "data/graph_output"
GRAPHFRAMES_PACKAGE = os.environ.get("GRAPHFRAMES_PACKAGE", "io.graphframes:graphframes-spark4_2.13:0.12.3")
MAVEN = "https://repo1.maven.org/maven2"
JAR_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "graphframes-jars")


def _pom_deps(group, artifact, version):
    """Runtime dependencies of one Maven artifact, read from its POM (compile/runtime scope, not optional)."""
    url = f"{MAVEN}/{group.replace('.', '/')}/{artifact}/{version}/{artifact}-{version}.pom"
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    props = {"project.version": version}
    props.update({p.tag.replace(ns, ""): (p.text or "").strip() for p in root.findall(f"{ns}properties/*")})
    deps = []
    for d in root.findall(f"{ns}dependencies/{ns}dependency"):
        g, a, v = (d.findtext(f"{ns}{t}") or "" for t in ("groupId", "artifactId", "version"))
        scope, optional = d.findtext(f"{ns}scope") or "compile", d.findtext(f"{ns}optional") == "true"
        if scope not in ("compile", "runtime") or optional:
            continue
        if g.startswith(("org.scala-lang", "org.apache.spark")):   # already shipped with PySpark
            continue
        for k, val in props.items():
            v = v.replace("${" + k + "}", val)
        if "${" in v or not v:
            sys.exit(f"Cannot resolve version of {g}:{a} from {url}. Set HADOOP_HOME (winutils) to use "
                     "spark.jars.packages instead.")
        deps.append((g, a, v))
    return deps


def local_graphframes_jars(coordinate):
    """Download `coordinate` and its runtime dependencies from Maven Central; return the local jar paths."""
    todo, seen, paths = [tuple(coordinate.split(":"))], set(), []
    os.makedirs(JAR_CACHE, exist_ok=True)
    while todo:
        g, a, v = todo.pop()
        if (g, a) in seen:
            continue
        seen.add((g, a))
        path = os.path.join(JAR_CACHE, f"{g}_{a}-{v}.jar")
        if not os.path.exists(path):
            print(f"downloading {g}:{a}:{v}")
            urllib.request.urlretrieve(f"{MAVEN}/{g.replace('.', '/')}/{a}/{v}/{a}-{v}.jar", path)
        paths.append(path)
        todo.extend(_pom_deps(g, a, v))
    return paths


def get_spark(use_graphframes=True):
    builder = (SparkSession.builder.master("local[*]").appName("ufc-graph")
               .config("spark.sql.shuffle.partitions", "8")   # the data is small; fewer partitions is faster
               .config("spark.ui.showConsoleProgress", "false"))
    if use_graphframes:
        if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
            jars = local_graphframes_jars(GRAPHFRAMES_PACKAGE)
            builder = builder.config("spark.driver.extraClassPath", os.pathsep.join(jars))
        else:
            builder = builder.config("spark.jars.packages", GRAPHFRAMES_PACKAGE)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_vertices_and_edges(spark):
    """Spark-only step: no GraphFrames required."""
    fighters = spark.read.csv(f"{CLEAN}/clean_fighter.csv", header=True, inferSchema=False)
    fights = spark.read.csv(f"{CLEAN}/clean_fight.csv", header=True, inferSchema=False, multiLine=True, escape='"')

    vertices = fighters.select(F.col("fighter_id").alias("id"), F.col("fighter_name").alias("name"))

    # Only fights with a winner make an edge (draws and no-contests have no winner_id).
    decided = fights.filter(F.col("winner_id").isNotNull() & (F.col("winner_id") != ""))
    edges = decided.select(
        F.col("winner_id").alias("src"),
        F.when(F.col("winner_id") == F.col("r_id"), F.col("b_id")).otherwise(F.col("r_id")).alias("dst"),
        "fight_id", "method", "event_id",
    )
    return vertices, edges, fights


def prep_report(vertices, edges, fights):
    n_v, n_fights, n_edges = vertices.count(), fights.count(), edges.count()
    distinct_pairs = edges.select("src", "dst").distinct().count()
    ids = vertices.select("id").distinct()
    dangling = (edges.join(ids, edges.src == ids.id, "left_anti").count()
                + edges.join(ids, edges.dst == ids.id, "left_anti").count())
    print(f"fighters (vertices):            {n_v}")
    print(f"fights:                         {n_fights}")
    print(f"fights with a winner (edges):   {n_edges}")
    print(f"distinct winner->loser pairs:   {distinct_pairs}")
    print(f"edges pointing at unknown IDs:  {dangling} (should be 0)")


def format_path(row):
    """GraphFrames BFS returns columns: from, e0, v1, e1, ..., to. Turn a row into 'A > B > C'."""
    names = [row["from"]["name"]]
    i = 1
    while f"v{i}" in row.asDict():
        names.append(row[f"v{i}"]["name"])
        i += 1
    names.append(row["to"]["name"])
    return " > ".join(names)


def resolve(vertices, name):
    ids = [r["id"] for r in vertices.filter(F.col("name") == name).select("id").collect()]
    if not ids:
        sys.exit(f"No fighter named '{name}'. Check the spelling against clean_fighter.csv.")
    if len(ids) > 1:
        print(f"Note: {len(ids)} fighters are named '{name}'; searching all of them.")
    return ids


def sizes(df, col, label):
    s = df.groupBy(col).count().orderBy(F.desc("count"))
    total = s.count()
    multi = s.filter(F.col("count") > 1).count()
    top = s.limit(3).collect()
    print(f"{label}: {total} groups, {multi} with more than one fighter, largest sizes: {[r['count'] for r in top]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-only", action="store_true", help="build vertices/edges and print counts, without GraphFrames")
    ap.add_argument("--from", dest="src_name")
    ap.add_argument("--to", dest="dst_name")
    ap.add_argument("--max-path", type=int, default=10)
    args = ap.parse_args()

    spark = get_spark(use_graphframes=not args.prep_only)
    vertices, edges, fights = build_vertices_and_edges(spark)
    prep_report(vertices, edges, fights)
    if args.prep_only:
        return

    from graphframes import GraphFrame   # imported late so --prep-only works without it
    g = GraphFrame(vertices, edges)
    os.makedirs(OUT, exist_ok=True)

    # 1. BFS: shortest chain of wins from one fighter to another
    if args.src_name and args.dst_name:
        src_ids, dst_ids = resolve(vertices, args.src_name), resolve(vertices, args.dst_name)
        q = lambda ids: "id IN (" + ",".join(f"'{i}'" for i in ids) + ")"
        paths = g.bfs(q(src_ids), q(dst_ids), maxPathLength=args.max_path)
        rows = paths.limit(1).collect()
        if rows:
            print(f"\nShortest win chain {args.src_name} -> {args.dst_name}:\n  {format_path(rows[0])}")
        else:
            print(f"\nNo win chain from {args.src_name} to {args.dst_name} within {args.max_path} wins.")

    # 2. Connected components (ignores edge direction)
    # Local checkpoints: no checkpoint directory, so no winutils needed on Windows.
    cc = g.connectedComponents(use_local_checkpoints=True)
    sizes(cc, "component", "\nconnected components")

    # 3. Strongly connected components (respects direction)
    scc = g.stronglyConnectedComponents(maxIter=10)
    sizes(scc, "component", "strongly connected components")

    # Save one row per fighter. GraphFrames' component numbers are arbitrary and change between runs, so each
    # group is relabelled with its smallest fighter id (stable while membership is unchanged) plus its size.
    # Written through pandas so it also works on Windows without winutils.
    def stable(df, prefix):
        w = Window.partitionBy("component")
        return df.select("id", F.min("id").over(w).alias(f"{prefix}_id"),
                         F.count("*").over(w).alias(f"{prefix}_size"))

    result = (vertices.join(stable(cc, "component"), "id").join(stable(scc, "scc"), "id")
                      .orderBy(F.desc("component_size"), "component_id", F.desc("scc_size"), "scc_id", "id"))
    result.toPandas().to_csv(f"{OUT}/fighter_graph_results.csv", index=False)
    print(f"\nWrote {OUT}/fighter_graph_results.csv")


if __name__ == "__main__":
    main()
