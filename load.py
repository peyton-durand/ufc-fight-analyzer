"""
Loads the cleaned data from the data/clean directory into MongoDB (events, fighters, fights collections).
If graph_analysis.py has been run, each fighter also gets a "graph" sub-document from
data/graph_output/fighter_graph_results.csv (component and strongly connected component: id and size).
Re-running is safe: documents are upserted by _id, and documents no longer in the clean data are removed.
"""
import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne, ASCENDING, DESCENDING
from pymongo.server_api import ServerApi

CLEAN = "data/clean"
GRAPH = "data/graph_output/fighter_graph_results.csv"


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


# fights ----------------------------------------------------------------------------------------------
def outcome(fighter_id, fight):
    if fight["result_status"] == "draw":
        return "draw"
    if fight["result_status"] == "no_contest":
        return "nc"
    return "win" if fighter_id == fight["winner_id"] else "loss"


def build_fights():
    df = pd.read_csv(f'{CLEAN}/clean_fight.csv')
    events = pd.read_csv(f'{CLEAN}/clean_event.csv', parse_dates=["date"]).set_index("event_id")
    names = pd.read_csv(f'{CLEAN}/clean_fighter.csv').set_index("fighter_id")["fighter_name"]

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
        print("Indexes created")


if __name__ == "__main__":
    load()
