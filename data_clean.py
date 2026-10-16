import pandas as pd
import re

df = pd.read_csv('data/raw/fighter.csv')
stat_cols = ['slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']

all_zero = df[stat_cols].eq(0).all(axis=1)
print(f"Number of rows with all zero stats: {all_zero.sum()}")

# Keep integer columns as integers but allow nulls
int_cols = ["str_acc", "str_def", "td_acc", "td_def"]
for c in int_cols:
    df[c] = df[c].astype('Int64')

# Set all-zero rows to NA for the stat columns
df.loc[all_zero, stat_cols] = pd.NA

print("\nStats after setting all-zero rows to NA:")
print(df[stat_cols].describe())


# Convert height to inches
def to_inches(h):
    if pd.isna(h):
        return pd.NA
    m = re.match(r"^\s*(\d+)'\s*(\d+)", str(h))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else pd.NA

df["height_in"] = df["height"].apply(to_inches).astype('Int64')
df = df.drop(columns=["height"])


# Statistics for checking data quality
print("\nheight_in stats:")
print(df["height_in"].describe())
print("height_in nulls:", df["height_in"].isna().sum())

print("\nColumn names:")
print(df.columns.tolist())


# Save the cleaned data to a new CSV file
df.to_csv('data/clean/clean_fighter.csv', index=False)