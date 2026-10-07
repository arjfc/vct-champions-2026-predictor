import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import vlrdevapi as vlr

PLAYOFF = [120, 11058, 2059, 11060, 1034, 14, 624, 6961]
MIN_MATCHES = 8     # teams with fewer matches than this are skipped (treated as average later)
PAUSE = 2.0         # seconds between requests
CACHE = Path("data/cache/team_stats")
CACHE.mkdir(parents=True, exist_ok=True)

# calendar quarters from 2025 Q1 to 2026 Q3
WINDOWS = []
for y in (2025, 2026):
    for sm, em, ed in [(1, 3, 31), (4, 6, 30), (7, 9, 30), (10, 12, 31)]:
        WINDOWS.append((date(y, sm, 1), date(y, em, ed)))
WINDOWS = [w for w in WINDOWS if w[0] <= date(2026, 9, 30)]

# ---------- which teams and windows do we need? ----------
df = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
events = set(df[df.involves_playoff_team]["event"])
df = df[df.event.isin(events)].dropna(subset=["winner_id"]).copy()
df["day"] = df["datetime"].dt.date

a = df[["team1_id", "day"]].rename(columns={"team1_id": "team_id"})
b = df[["team2_id", "day"]].rename(columns={"team2_id": "team_id"})
long = pd.concat([a, b])
counts = long.team_id.value_counts()
team_days = long.groupby("team_id")["day"].apply(list)

teams = [t for t in PLAYOFF] + [t for t in counts[counts >= MIN_MATCHES].index
                                if t not in PLAYOFF]

jobs = []
for t in teams:
    for s, e in WINDOWS:
        if any(s <= d <= e for d in team_days.get(t, [])):   # only windows where the team played
            jobs.append((int(t), s, e))

def path_for(t, s, e):
    return CACHE / f"{t}_{s.isoformat()}_{e.isoformat()}.json"

todo = [j for j in jobs if not path_for(*j).exists()]
print(f"Teams: {len(teams)} | windows per team: up to {len(WINDOWS)}")
print(f"Requests needed: {len(jobs)} | already cached: {len(jobs) - len(todo)} | remaining: {len(todo)}")
print(f"Estimated time: about {len(todo) * (PAUSE + 1) / 60:.0f} minutes")
input("Press Enter to start (Ctrl+C to cancel)... ")

# ---------- download ----------
def fetch(t, s, e):
    for attempt in range(3):
        try:
            return vlr.team.stats(t, date_start=s, date_end=e).model_dump(mode="json")
        except Exception as ex:
            print(f"   failed ({type(ex).__name__}: {ex}), retrying...")
            time.sleep(3)
    return None

failed = []
try:
    for i, (t, s, e) in enumerate(todo, 1):
        data = fetch(t, s, e)
        if data is None:
            failed.append((t, s, e))
            print(f"[{i}/{len(todo)}] team {t} {s} to {e}: FAILED, skipped")
        else:
            path_for(t, s, e).write_text(json.dumps(data), encoding="utf-8")
            print(f"[{i}/{len(todo)}] team {t} {s} to {e}: {len(data.get('maps', []))} maps")
        time.sleep(PAUSE)
finally:
    # ---------- combine everything cached into one CSV ----------
    rows = []
    for f in CACHE.glob("*.json"):
        t, s, e = f.stem.split("_")
        for m in json.loads(f.read_text(encoding="utf-8")).get("maps", []):
            rows.append({
                "team_id": int(t), "window_start": s, "window_end": e,
                "map_name": m.get("map_name"), "games_played": m.get("games_played"),
                "wins": m.get("wins"), "losses": m.get("losses"),
                "attack_rounds_won": m.get("attack_rounds_won"),
                "attack_rounds_lost": m.get("attack_rounds_lost"),
                "defense_rounds_won": m.get("defense_rounds_won"),
                "defense_rounds_lost": m.get("defense_rounds_lost"),
            })
    pd.DataFrame(rows).to_csv("data/team_map_stats.csv", index=False)
    print(f"\nSaved {len(rows)} rows to data/team_map_stats.csv")
    if failed:
        print(f"{len(failed)} requests failed. Run the script again to retry them.")