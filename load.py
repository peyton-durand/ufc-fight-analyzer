"""
Loads the cleaned data from the data/clean directory into MongoDB (events, fighters, fights collections).
Per-round stats from clean_round.csv are embedded in each fight as a "rounds" array.
If graph_analysis.py has been run, each fighter also gets a "graph" sub-document from
data/graph_output/fighter_graph_results.csv (component and strongly connected component: id and size).
If style_analysis.py --k has been run, each clustered fighter also gets a "style" sub-document from
data/style_output/fighter_style_results.csv (cluster, label, fights used and the style features).
Re-running is safe: documents are upserted by _id, and documents no longer in the clean data are removed.
"""
import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne, ASCENDING, DESCENDING
from pymongo.server_api import ServerApi

CLEAN = "data/clean"
GRAPH = "data/graph_output/fighter_graph_results.csv"
STYLE = "data/style_output/fighter_style_results.csv"


# Rows as dicts, leaving out null fields (a missing field is the Mongo way to say "no value")
def records(df):
    return [{k: v for k, v in r.items() if pd.notna(v)} for r in df.to_dict("records")]


# events ----------------------------------------------------------------------------------------------
def build_events():
    df = pd.read_csv(f'{CLEAN}/clean_event.csv', parse_dates=["date"])
    df = df.rename(columns={"event_id": "_id", "event_name": "name"})
    return records(df)


# fighters --------------------------------------------------------------------------------------------
def build_fighters():
    int_cols = ["height_in", "weight_lbs", "reach_inches", "str_acc", "str_def", "td_acc", "td_def"]
    df = pd.read_csv(f'{CLEAN}/clean_fighter.csv', parse_dates=["dob"], dtype={c: "Int64" for c in int_cols})
    df = df.rename(columns={"fighter_id": "_id", "fighter_name": "name", "fighter_nick_name": "nickname"})

    # Group the ufcstats career stats into one sub-document
    career_cols = ['slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']
    docs = records(df.drop(columns=career_cols))
    for doc, career in zip(docs, records(df[career_cols])):
        if career:
            doc["career"] = career

    graph = build_graph()
    if graph is not None:
        missing = {d["_id"] for d in docs} ^ graph.keys()
        if missing:
            raise SystemExit(f"{GRAPH} does not match {CLEAN}/clean_fighter.csv ({len(missing)} ids differ). "
                             "Re-run graph_analysis.py.")
        for doc in docs:
            doc["graph"] = graph[doc["_id"]]

    # Only fighters with enough tracked fights are clustered, so missing style fields are expected
    style = build_style()
    if style is not None:
        unknown = style.keys() - {d["_id"] for d in docs}
        if unknown:
            raise SystemExit(f"{STYLE} has {len(unknown)} fighter ids not in {CLEAN}/clean_fighter.csv. "
                             "Re-run style_analysis.py.")
        for doc in docs:
            if doc["_id"] in style:
                doc["style"] = style[doc["_id"]]
    return docs


# Win-graph results from graph_analysis.py, keyed by fighter id. Optional: fighters load without it.
def build_graph():
    if not os.path.exists(GRAPH):
        print(f"{GRAPH} not found: loading fighters without graph fields (run graph_analysis.py first)")
        return None
    df = pd.read_csv(GRAPH, dtype={"id": str, "component_id": str, "scc_id": str})
    return {
        r["id"]: {
            "component": {"id": r["component_id"], "size": int(r["component_size"])},  # weakly connected
            "scc": {"id": r["scc_id"], "size": int(r["scc_size"])},                    # strongly connected
        }
        for r in df.to_dict("records")
    }


# Style clusters from style_analysis.py, keyed by fighter id. Optional: fighters load without it.
# Every column after these is a style feature, so new features need no change here
STYLE_COLUMNS = ["fighter_id", "feature_set", "cluster", "label", "fights"]


def build_style():
    if not os.path.exists(STYLE):
        print(f"{STYLE} not found: loading fighters without style fields (run style_analysis.py --k N first)")
        return None
    df = pd.read_csv(STYLE, dtype={"fighter_id": str})
    features = [c for c in df.columns if c not in STYLE_COLUMNS]
    return {
        r["fighter_id"]: {
            "feature_set": r["feature_set"],
            "cluster": int(r["cluster"]),
            "label": r["label"],
            "fights_used": int(r["fights"]),
            "features": {f: round(float(r[f]), 3) for f in features},
        }
        for r in df.to_dict("records")
    }


# fights ----------------------------------------------------------------------------------------------
def outcome(fighter_id, fight):
    if fight["result_status"] == "draw":
        return "draw"
    if fight["result_status"] == "no_contest":
        return "nc"
    return "win" if fighter_id == fight["winner_id"] else "loss"


# One corner's stats for one round, from a clean_round.csv row and its column prefix ("r" or "b")
def corner_stats(row, p):
    def pair(landed, atmp):
        return {"landed": int(row[f"{p}_{landed}"]), "attempted": int(row[f"{p}_{atmp}"])}

    def sig_by(part):
        return pair(f"sig_str_landed_{part}", f"sig_str_atmp_{part}")

    stats = {
        "kd": int(row[f"{p}_kd"]),
        "sig": pair("sig_landed", "sig_atmp"),
        "total": pair("total_str_landed", "total_str_atmp"),
        "td": pair("td_success", "td_atmp"),
        "sub_att": int(row[f"{p}_sub_att"]),
        "rev": int(row[f"{p}_rev"]),
        "target": {part: sig_by(part) for part in ["head", "body", "leg"]},
        "position": {part: sig_by(part) for part in ["distance", "clinch", "ground"]},
    }
    # Missing field = no value: control time wasn't tracked in 1993-1999
    if pd.notna(row[f"{p}_ctrl_sec"]):
        stats["ctrl_sec"] = int(row[f"{p}_ctrl_sec"])
    return stats


# Rounds keyed by fight id, each list sorted by round number. Corners are on the fight, so r_id/b_id are dropped
def build_rounds():
    df = pd.read_csv(f'{CLEAN}/clean_round.csv', dtype={"r_ctrl_sec": "Int64", "b_ctrl_sec": "Int64"})
    df = df.sort_values(["fight_id", "round_no"])
    rounds = {}
    for row in df.to_dict("records"):
        rounds.setdefault(row["fight_id"], []).append({
            "round": int(row["round_no"]),
            "red": corner_stats(row, "r"),
            "blue": corner_stats(row, "b"),
        })
    return rounds


def build_fights():
    df = pd.read_csv(f'{CLEAN}/clean_fight.csv')
    events = pd.read_csv(f'{CLEAN}/clean_event.csv', parse_dates=["date"]).set_index("event_id")
    names = pd.read_csv(f'{CLEAN}/clean_fighter.csv').set_index("fighter_id")["fighter_name"]
    rounds = build_rounds()

    df["title_fight"] = df["title_fight"].astype(bool)
    df["is_tournament"] = df["is_tournament"].astype(bool)
    df = df.rename(columns={"fight_id": "_id"})

    docs = []
    for fight in records(df):
        event_id = fight.pop("event_id")
        fight["event"] = {
            "id": event_id,
            "name": events.at[event_id, "event_name"],
            "date": events.at[event_id, "date"],
        }

        # Red and blue corners become one array, so a fighter's fights are a single indexed query
        fight["fighters"] = [
            {
                "corner": corner,
                "fighter_id": fight[f"{prefix}_id"],
                "name": names[fight[f"{prefix}_id"]],
                "outcome": outcome(fight[f"{prefix}_id"], fight),
            }
            for corner, prefix in [("red", "r"), ("blue", "b")]
        ]
        del fight["r_id"], fight["b_id"]

        # Fights with no round data get no rounds field
        if fight["_id"] in rounds:
            fight["rounds"] = rounds[fight["_id"]]
        docs.append(fight)
    return docs


# load ------------------------------------------------------------------------------------------------
def upsert(collection, docs):
    collection.bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs])
    removed = collection.delete_many({"_id": {"$nin": [d["_id"] for d in docs]}}).deleted_count
    print(f"{collection.name}: {len(docs)} upserted, {removed} removed")


def connect():
    load_dotenv()
    client = MongoClient(os.environ["MONGODB_URI"], server_api=ServerApi(version="1", strict=True, deprecation_errors=True))
    client.admin.command("ping")
    print("Connected to MongoDB")
    return client


def load():
    with connect() as client:
        db = client[os.environ.get("MONGODB_DB", "ufc")]

        upsert(db.events, build_events())
        upsert(db.fighters, build_fighters())
        upsert(db.fights, build_fights())

        db.fights.create_index([("fighters.fighter_id", ASCENDING)])
        db.fights.create_index([("event.id", ASCENDING)])
        db.fights.create_index([("event.date", DESCENDING)])
        db.fights.create_index([("weight_class", ASCENDING)])
        db.fighters.create_index([("name", ASCENDING)])
        db.fighters.create_index([("graph.component.id", ASCENDING)])
        db.fighters.create_index([("graph.scc.id", ASCENDING)])
        db.fighters.create_index([("style.cluster", ASCENDING)])
        print("Indexes created")


if __name__ == "__main__":
    load()
