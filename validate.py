"""
Runs validation queries against the MongoDB database to check the load was successful.
Exits with a non-zero status if any check fails.
"""
import os
import sys
from datetime import datetime
import pandas as pd
from load import connect, CLEAN

WEIGHT_CLASSES = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight", "Middleweight",
    "Light Heavyweight", "Heavyweight", "Super Heavyweight", "Open Weight", "Catch Weight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
]

failures = 0


def check(name, bad_count):
    global failures
    status = "PASS" if bad_count == 0 else "FAIL"
    failures += bad_count != 0
    print(f"  [{status}] {name}" + (f" ({bad_count} bad)" if bad_count else ""))


def count(collection, pipeline):
    result = list(collection.aggregate(pipeline + [{"$count": "n"}]))
    return result[0]["n"] if result else 0


# Document counts match the clean CSVs ----------------------------------------------------------------
def check_counts(db):
    print("\nCounts")
    for collection, file in [("events", "clean_event"), ("fighters", "clean_fighter"), ("fights", "clean_fight")]:
        expected = len(pd.read_csv(f'{CLEAN}/{file}.csv'))
        actual = db[collection].count_documents({})
        check(f"{collection}: {actual} docs, expected {expected}", abs(actual - expected))


# Every reference points to a document that exists, and nothing is orphaned ---------------------------
def check_references(db):
    print("\nReferences")
    check("every fight's event exists", count(db.fights, [
        {"$lookup": {"from": "events", "localField": "event.id", "foreignField": "_id", "as": "e"}},
        {"$match": {"e": {"$size": 0}}},
    ]))
    check("every fight's fighters exist", count(db.fights, [
        {"$unwind": "$fighters"},
        {"$lookup": {"from": "fighters", "localField": "fighters.fighter_id", "foreignField": "_id", "as": "f"}},
        {"$match": {"f": {"$size": 0}}},
    ]))
    check("every event has fights", count(db.events, [
        {"$lookup": {"from": "fights", "localField": "_id", "foreignField": "event.id", "as": "f"}},
        {"$match": {"f": {"$size": 0}}},
    ]))
    check("every fighter has fights", count(db.fighters, [
        {"$lookup": {"from": "fights", "localField": "_id", "foreignField": "fighters.fighter_id", "as": "f"}},
        {"$match": {"f": {"$size": 0}}},
    ]))


# Fight documents are internally consistent -----------------------------------------------------------
def check_fights(db):
    print("\nFights")
    check("every fight has exactly 2 fighters", db.fights.count_documents(
        {"$expr": {"$ne": [{"$size": "$fighters"}, 2]}}
    ))
    check("no fighter fights themselves", db.fights.count_documents(
        {"$expr": {"$eq": [{"$arrayElemAt": ["$fighters.fighter_id", 0]}, {"$arrayElemAt": ["$fighters.fighter_id", 1]}]}}
    ))

    # Wins: exactly one fighter has outcome "win", and it's the winner_id
    winners = {"$filter": {"input": "$fighters", "cond": {"$eq": ["$$this.outcome", "win"]}}}
    check("wins have one winner matching winner_id", db.fights.count_documents({
        "result_status": "win",
        "$expr": {"$or": [
            {"$ne": [{"$size": winners}, 1]},
            {"$ne": [{"$arrayElemAt": [{"$map": {"input": winners, "in": "$$this.fighter_id"}}, 0]}, "$winner_id"]},
        ]},
    }))
    check("wins have one loser", db.fights.count_documents({
        "result_status": "win",
        "$expr": {"$ne": [{"$size": {"$filter": {"input": "$fighters", "cond": {"$eq": ["$$this.outcome", "loss"]}}}}, 1]},
    }))

    # Draws and no contests: no winner, and both outcomes match the result
    check("draws/no contests have no winner_id", db.fights.count_documents(
        {"result_status": {"$in": ["draw", "no_contest"]}, "winner_id": {"$exists": True}}
    ))
    check("draw outcomes are all 'draw'", db.fights.count_documents(
        {"result_status": "draw", "fighters": {"$elemMatch": {"outcome": {"$ne": "draw"}}}}
    ))
    check("no contest outcomes are all 'nc'", db.fights.count_documents(
        {"result_status": "no_contest", "fighters": {"$elemMatch": {"outcome": {"$ne": "nc"}}}}
    ))

    check("fight_seconds is positive", db.fights.count_documents(
        {"$or": [{"fight_seconds": {"$lte": 0}}, {"fight_seconds": {"$exists": False}}]}
    ))
    check("weight_class is a known class", db.fights.count_documents(
        {"weight_class": {"$nin": WEIGHT_CLASSES}}
    ))
    check("method_group is a known group", db.fights.count_documents(
        {"method_group": {"$nin": ["KO/TKO", "Submission", "Decision", "Other"]}}
    ))
    check("decisions have no finish_detail", db.fights.count_documents(
        {"method_group": "Decision", "finish_detail": {"$exists": True}}
    ))


# Per-round stats embedded in fights ------------------------------------------------------------------
def check_rounds(db):
    print("\nRounds")
    df = pd.read_csv(f'{CLEAN}/clean_round.csv')

    expected = df["fight_id"].nunique()
    actual = db.fights.count_documents({"rounds": {"$exists": True}})
    check(f"fights with rounds: {actual}, expected {expected}", abs(actual - expected))

    result = list(db.fights.aggregate([{"$group": {"_id": None, "n": {"$sum": {"$size": {"$ifNull": ["$rounds", []]}}}}}]))
    actual = result[0]["n"] if result else 0
    check(f"round entries: {actual}, expected {len(df)}", abs(actual - len(df)))

    check("rounds array length equals the fight's round", db.fights.count_documents(
        {"rounds": {"$exists": True}, "$expr": {"$ne": [{"$size": "$rounds"}, "$round"]}}
    ))

    # One unwound row per round per corner
    per_corner = [
        {"$unwind": "$rounds"},
        {"$project": {"stats": ["$rounds.red", "$rounds.blue"]}},
        {"$unwind": "$stats"},
    ]
    check("no landed > attempted (sig, total, td, target, position)", count(db.fights, per_corner + [
        {"$match": {"$expr": {"$or": [
            {"$gt": [f"$stats.{s}.landed", f"$stats.{s}.attempted"]}
            for s in ["sig", "total", "td", "target.head", "target.body", "target.leg",
                      "position.distance", "position.clinch", "position.ground"]
        ]}}},
    ]))
    check("ctrl_sec is >= 0 where present", count(db.fights, per_corner + [
        {"$match": {"stats.ctrl_sec": {"$exists": True, "$not": {"$gte": 0}}}},
    ]))


# Types and value ranges ------------------------------------------------------------------------------
def check_types(db):
    print("\nTypes and ranges")
    check("event dates are Dates", db.events.count_documents({"date": {"$not": {"$type": "date"}}}))
    check("event dates are between UFC 1 and today", db.events.count_documents(
        {"$or": [{"date": {"$lt": datetime(1993, 11, 12)}}, {"date": {"$gt": datetime.now()}}]}
    ))
    check("fight event dates are Dates", db.fights.count_documents({"event.date": {"$not": {"$type": "date"}}}))
    check("title_fight/is_tournament are bools", db.fights.count_documents(
        {"$or": [{"title_fight": {"$not": {"$type": "bool"}}}, {"is_tournament": {"$not": {"$type": "bool"}}}]}
    ))
    check("no NaN values stored", db.fighters.count_documents(
        {"$or": [{f: float("nan")} for f in ["height_in", "weight_lbs", "reach_inches", "career.slpm"]]}
    ) + db.fights.count_documents({"fight_seconds": float("nan")}))
    check("fighter heights are 4'0\" to 8'0\"", db.fighters.count_documents(
        {"height_in": {"$exists": True, "$not": {"$gte": 48, "$lte": 96}}}
    ))
    check("career percentages are 0-100", db.fighters.count_documents({"$or": [
        {f"career.{c}": {"$exists": True, "$not": {"$gte": 0, "$lte": 100}}}
        for c in ["str_acc", "str_def", "td_acc", "td_def"]
    ]}))


# Indexes from load.py exist --------------------------------------------------------------------------
def check_indexes(db):
    print("\nIndexes")
    fight_keys = {tuple(ix["key"].keys()) for ix in db.fights.list_indexes()}
    fighter_keys = {tuple(ix["key"].keys()) for ix in db.fighters.list_indexes()}
    for key in ["fighters.fighter_id", "event.id", "event.date", "weight_class"]:
        check(f"fights.{key}", int((key,) not in fight_keys))
    check("fighters.name", int(("name",) not in fighter_keys))



# Database queries ------------------------------------------------------------------------------------
def show_examples(db):
    print("\nQueries")

    latest = db.events.find_one(sort=[("date", -1)])
    card = db.fights.count_documents({"event.id": latest["_id"]})
    print(f"  Latest event: {latest['name']} ({latest['date']:%Y-%m-%d}, {latest['city']}) - {card} fights")

    top = list(db.fights.aggregate([
        {"$unwind": "$fighters"},
        {"$group": {
            "_id": "$fighters.name",
            "wins": {"$sum": {"$cond": [{"$eq": ["$fighters.outcome", "win"]}, 1, 0]}},
            "losses": {"$sum": {"$cond": [{"$eq": ["$fighters.outcome", "loss"]}, 1, 0]}},
        }},
        {"$sort": {"wins": -1}},
        {"$limit": 5},
    ]))
    print("  Most UFC wins: " + ", ".join(f"{t['_id']} ({t['wins']}-{t['losses']})" for t in top))

    finish_rates = db.fights.aggregate([
        {"$match": {"weight_class": {"$nin": ["Open Weight", "Catch Weight", "Super Heavyweight"]}}},
        {"$group": {"_id": "$weight_class", "fights": {"$sum": 1},
                    "finishes": {"$sum": {"$cond": [{"$in": ["$method_group", ["KO/TKO", "Submission"]]}, 1, 0]}}}},
        {"$project": {"rate": {"$divide": ["$finishes", "$fights"]}}},
        {"$sort": {"rate": -1}},
    ])
    print("  Finish rate by weight class:")
    for r in finish_rates:
        print(f"    {r['_id']:<22} {r['rate']:.0%}")

    # Red stats go to fighters[0] (red corner), blue stats to fighters[1]
    strikers = db.fights.aggregate([
        {"$match": {"rounds": {"$exists": True}}},
        {"$unwind": "$rounds"},
        {"$project": {"corners": [
            {"fighter": {"$arrayElemAt": ["$fighters", 0]}, "landed": "$rounds.red.sig.landed"},
            {"fighter": {"$arrayElemAt": ["$fighters", 1]}, "landed": "$rounds.blue.sig.landed"},
        ]}},
        {"$unwind": "$corners"},
        {"$group": {"_id": "$corners.fighter.fighter_id", "name": {"$first": "$corners.fighter.name"},
                    "landed": {"$sum": "$corners.landed"}}},
        {"$sort": {"landed": -1}},
        {"$limit": 5},
    ])
    print("  Most significant strikes landed: " + ", ".join(f"{s['name']} ({s['landed']})" for s in strikers))


if __name__ == "__main__":
    with connect() as client:
        db = client[os.environ.get("MONGODB_DB", "ufc")]
        check_counts(db)
        check_references(db)
        check_fights(db)
        check_rounds(db)
        check_types(db)
        check_indexes(db)
        show_examples(db)

    print(f"\n{'All checks passed' if failures == 0 else f'{failures} check(s) failed'}")
    sys.exit(1 if failures else 0)
