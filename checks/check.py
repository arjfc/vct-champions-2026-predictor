import pandas as pd

df = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
print("Rows:", len(df))
print("Date range:", df["datetime"].min().date(), "to", df["datetime"].max().date())
print("Missing winner:", df["winner_id"].isna().sum())
print("Duplicate match_ids:", df["match_id"].duplicated().sum())

ids = {120: "100 Thieves", 11058: "G2", 2059: "Vitality", 11060: "Nongshim",
       1034: "NRG", 14: "T1", 624: "Paper Rex", 6961: "LOUD"}
print("\nMatches per playoff team:")
for i, name in ids.items():
    n = ((df.team1_id == i) | (df.team2_id == i)).sum()
    print(f"  {name}: {n}")

mine = df[df.involves_playoff_team]
print("\nEvents involving your teams:")
print(mine["event"].value_counts().to_string())