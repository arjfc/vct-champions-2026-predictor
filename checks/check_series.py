import numpy as np
import pandas as pd

PLAYOFF = {120: "100 Thieves", 11058: "G2 Esports", 2059: "Team Vitality",
           11060: "Nongshim RedForce", 1034: "NRG", 14: "T1",
           624: "Paper Rex", 6961: "LOUD"}

m = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
events = set(m[m.involves_playoff_team]["event"])
m = m[m.event.isin(events)].dropna(subset=["winner_id"]).copy()
games = pd.read_csv("data/series_games.csv")
pl = pd.read_csv("data/series_players.csv")
pl = pl.dropna(subset=["team_id", "player_id"]).copy()
pl["team_id"] = pl.team_id.astype(int)
pl["player_id"] = pl.player_id.astype(int)

# ---------- 1. coverage ----------
ids = set(m.match_id)
print("=== 1. Coverage ===")
print(f"Matches in matches.csv (filtered): {len(ids)}")
print(f"  with map data:    {games.match_id.nunique()}")
print(f"  with player data: {pl.match_id.nunique()}")
print(f"  missing map data:    {len(ids - set(games.match_id))}")
print(f"  missing player data: {len(ids - set(pl.match_id))}")

# ---------- 2. players per team per match ----------
print("\n=== 2. Players per team per match (expect 5) ===")
cnt = pl.groupby(["match_id", "team_id"]).player_id.nunique()
print(cnt.value_counts().sort_index().to_string())
teams_per_match = pl.groupby("match_id").team_id.nunique()
print("Teams per match (expect 2):", teams_per_match.value_counts().sort_index().to_dict())

# ---------- 3. missing and odd values ----------
print("\n=== 3. Missing values in player stats ===")
cols = ["rating", "acs", "kast", "adr", "first_kills", "first_deaths", "attack_rating", "defend_rating"]
print(pl[cols].isna().sum().to_string())
print("\nRating summary (typical pro rating is about 0.8 to 1.3):")
print(pl.rating.describe().round(3).to_string())

# ---------- 4. do map scores agree with series scores? ----------
print("\n=== 4. Map results vs series score ===")
g = games[games.played.astype(str).str.lower() == "true"].dropna(subset=["team1_score", "team2_score"]).copy()
g["w"] = np.where(g.team1_score > g.team2_score, g.team1_id,
                  np.where(g.team2_score > g.team1_score, g.team2_id, np.nan))
g = g[g.w.notna()]
won = {(r.match_id, int(r.w)): r.n for r in
       g.groupby(["match_id", "w"]).size().rename("n").reset_index().itertuples()}
mm = m[m.match_id.isin(set(games.match_id))].copy()
mm["maps1"] = [won.get((i, int(t)), 0) for i, t in zip(mm.match_id, mm.team1_id)]
mm["maps2"] = [won.get((i, int(t)), 0) for i, t in zip(mm.match_id, mm.team2_id)]
bad = mm[(mm.maps1 != mm.team1_score) | (mm.maps2 != mm.team2_score)]
print(f"Matches checked: {len(mm)} | mismatches: {len(bad)}")
if len(bad):
    print(bad[["match_id", "team1_name", "team1_score", "maps1",
               "team2_name", "team2_score", "maps2"]].head(8).to_string(index=False))

print("\nMaps played (map rows):")
print(g.map_name.value_counts().to_string())
print("\nBest-of formats:", games.drop_duplicates("match_id").best_of.value_counts().to_dict())

# ---------- 5. how much history does each player have? ----------
print("\n=== 5. Series per player ===")
spp = pl.groupby("player_id").match_id.nunique()
print(f"Distinct players: {len(spp)}")
print(f"  with 5+ series:  {(spp >= 5).sum()}")
print(f"  with 10+ series: {(spp >= 10).sum()}")
print(f"  with only 1-2 series: {(spp <= 2).sum()}")

# ---------- 6. latest lineup of each playoff team ----------
print("\n=== 6. Latest lineup per playoff team (series in dataset, avg rating) ===")
last = pl.merge(m[["match_id", "datetime"]], on="match_id")
avg = pl.groupby("player_id").rating.mean()
for tid, name in PLAYOFF.items():
    t = last[last.team_id == tid]
    if t.empty:
        print(f"{name}: no player data")
        continue
    latest = t.sort_values("datetime").match_id.iloc[-1]
    row = t[t.match_id == latest]
    print(f"\n{name} (latest match {latest}):")
    for r in row.itertuples():
        print(f"   {r.player:<14} series={spp[r.player_id]:<4} avg rating={avg[r.player_id]:.2f}")