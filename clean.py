"""
Takes raw data from the data/raw directory and cleans it, saving the cleaned data to the data/clean directory.
"""
import pandas as pd
import re

RAW = "data/raw"
CLEAN = "data/clean"


# event.csv -------------------------------------------------------------------------------------------
# Locations missing from the scraped data
LOCATION_FIXES = {
    "6420efac0578988b": "Denver, Colorado, USA",  # UFC 1: The Beginning
}


def load_events():
    events = pd.read_csv(f'{RAW}/event.csv')
    for event_id, location in LOCATION_FIXES.items():
        events.loc[events["event_id"] == event_id, "location"] = location
    return events


# Only keep events that count towards a UFC record
# Non-UFC events (PRIDE, WEC, Strikeforce, DWCS, Road to UFC, ...) have no location
def is_ufc_event(events):
    return events["location"].notna()


def ufc_event_ids():
    events = load_events()
    return set(events.loc[is_ufc_event(events), "event_id"])


def clean_event():
    print("\nCleaning event.csv...")
    df = load_events()

    print(f"\nPre-cleaning checks:")
    print(f"Number of events: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Drop non-UFC events (PRIDE, WEC, Strikeforce, DWCS, Road to UFC, ...)
    keep = is_ufc_event(df)
    print(f"\nDropping {(~keep).sum()} non-UFC events")
    df = df[keep]

    # Locations are "City, Country" or "City, Region, Country"
    # Split the location into city, region, and country columns
    parts = df["location"].str.split(", ")
    df["city"] = parts.str[0]
    df["region"] = parts.where(parts.str.len() == 3).str[1]
    df["country"] = parts.str[-1]
    df = df.drop(columns=["location"])

    print(f"\nPost-cleaning checks:")
    print(f"Number of events: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(f'{CLEAN}/clean_event.csv', index=False)


# fight.csv -------------------------------------------------------------------------------------------
def clean_fight():
    print("\nCleaning fight.csv...")
    df = pd.read_csv(f'{RAW}/fight.csv')

    print(f"\nPre-cleaning checks:")
    print(f"Number of fights: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Drop non-UFC fights (PRIDE, WEC, Strikeforce, DWCS, Road to UFC, ...)
    keep = df["event_id"].isin(ufc_event_ids())
    print(f"\nDropping {(~keep).sum()} non-UFC fights")
    df = df[keep].copy()

    # Weight class: tournament labels ("Ultimate Fighter 9 Welterweight Tournament", "13 Lightweight Tournament")
    # are reduced to their base class, and labels with no weight class ("3 Tournament", "Superfight") become Open Weight
    classes = r"(Women's \w+weight|Light Heavyweight|Super Heavyweight|\w+weight|Open Weight|Catch Weight)"
    df["is_tournament"] = df["weight_class"].str.contains("Tournament").astype(int)
    df["weight_class"] = df["weight_class"].str.extract(classes)[0].fillna("Open Weight")
    print(f"\nWeight classes: {sorted(df['weight_class'].unique())}")

    # Fight duration in seconds: full rounds before the last one plus the time in the last round
    # Round lengths come from time_format, e.g. "3 Rnd (5-5-5)" or "1 Rnd + OT (12-3)"; "No Time Limit" fights all ended in round 1
    def fight_seconds(row):
        m = re.search(r"\(([\d-]+)\)", row["time_format"])
        lengths = [int(x) for x in m.group(1).split("-")] if m else []
        mins, secs = map(int, row["time"].split(":"))
        return sum(lengths[:row["round"] - 1]) * 60 + mins * 60 + secs

    df["fight_seconds"] = df.apply(fight_seconds, axis=1)

    # Details: collapse stray whitespace ("Injury  \n      Neck Injury"), then split by what it holds
    # Decisions hold judges' scorecards; everything else holds the finish technique or reason
    df["details"] = df["details"].str.split().str.join(" ")
    is_decision = df["method"].str.startswith("Decision")
    df["scorecards"] = df["details"].where(is_decision)
    df["finish_detail"] = df["details"].where(~is_decision)
    df = df.drop(columns=["details"])

    # Coarse method grouping for analysis
    method_groups = {
        "KO/TKO": "KO/TKO",
        "TKO - Doctor's Stoppage": "KO/TKO",
        "Submission": "Submission",
        "Decision - Unanimous": "Decision",
        "Decision - Split": "Decision",
        "Decision - Majority": "Decision",
    }
    df["method_group"] = df["method"].map(method_groups).fillna("Other")

    # Sanity checks
    assert df["fight_id"].is_unique, "Duplicate fight_id"
    assert df["fight_seconds"].gt(0).all(), "Non-positive fight duration"
    assert df["winner_id"].isna().equals(df["result_status"].ne("win")), "winner_id nulls don't match draws/no contests"

    print(f"\nPost-cleaning checks:")
    print(f"Number of fights: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(f'{CLEAN}/clean_fight.csv', index=False)


# fighter.csv -----------------------------------------------------------------------------------------
# Fighters who have fought in at least one official UFC fight
def ufc_fighter_ids():
    fights = pd.read_csv(f'{RAW}/fight.csv')
    fights = fights[fights["event_id"].isin(ufc_event_ids())]
    return set(fights["r_id"]) | set(fights["b_id"])


def clean_fighter():
    print("\nCleaning fighter.csv...")
    df = pd.read_csv(f'{RAW}/fighter.csv')

    print(f"\nPre-cleaning checks:")
    print(f"Number of fighters: {len(df)}")

    # Drop fighters with no UFC fights
    keep = df["fighter_id"].isin(ufc_fighter_ids())
    print(f"\nDropping {(~keep).sum()} fighters with no UFC fights")
    df = df[keep].copy()

    # Fighters with no fight stats (all zeros) do not have scraped fight data
    # Convert the stat columns to nullable integers and set all-zero rows to NA
    stat_cols = ['slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']
    all_zero = df[stat_cols].eq(0).all(axis=1)
    print(f"Number of fighters with no fight stats: {all_zero.sum()}")

    # Keep integer columns as integers but allow nulls
    int_cols = ["str_acc", "str_def", "td_acc", "td_def"]
    for c in int_cols:
        df[c] = df[c].astype('Int64')

    # Set all-zero rows to NA for the stat columns
    df.loc[all_zero, stat_cols] = pd.NA

    # Convert height to inches
    def to_inches(h):
        if pd.isna(h):
            return pd.NA
        m = re.match(r"^\s*(\d+)'\s*(\d+)", str(h))
        return int(m.group(1)) * 12 + int(m.group(2)) if m else pd.NA

    df["height_in"] = df["height"].apply(to_inches).astype('Int64')
    df = df.drop(columns=["height"])

    # Convert weight and reach to nullable integers
    for c in ["weight_lbs", "reach_inches"]:
        if (df[c].dropna() % 1 == 0).all():
            df[c] = df[c].astype('Int64')

    print(f"\nPost-cleaning checks:")
    print(f"Number of fighters: {len(df)}")

    df.to_csv(f'{CLEAN}/clean_fighter.csv', index=False)



if __name__ == "__main__":
    clean_event()
    clean_fight()
    clean_fighter()
