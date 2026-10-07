import json
import time
from pathlib import Path

import pandas as pd
import vlrdevapi as vlr

PAUSE = 2.0                      # seconds between requests (be polite to VLR)
CACHE = Path("data/cache/series")
CACHE.mkdir(parents=True, exist_ok=True)

# ---------- which matches? (same filter as elo.py) ----------
df = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
events = set(df[df.involves_playoff_team]["event"])
df = df[df.event.isin(events)].dropna(subset=["winner_id"]).copy()
df = df.sort_values("datetime", ascending=False)

# matches involving your 8 teams first, so a partial run is already useful
ids = (list(df[df.involves_playoff_team].match_id) +
       list(df[~df.involves_playoff_team].match_id))
ids = [int(i) for i in ids]

def path_for(kind, mid):
    return CACHE / f"{kind}_{mid}.json"

jobs = [(kind, mid) for mid in ids for kind in ("info", "players")]
todo = [j for j in jobs if not path_for(*j).exists()]
print(f"Matches: {len(ids)} | requests needed: {len(jobs)} | "
      f"already cached: {len(jobs) - len(todo)} | remaining: {len(todo)}")
print(f"Estimated time: about {len(todo) * (PAUSE + 1) / 60:.0f} minutes")
print("Tip: keep your laptop plugged in and stop it from sleeping while this runs.")
input("Press Enter to start (Ctrl+C to stop at any time, progress is saved)... ")

# ---------- download ----------
def fetch(kind, mid):
    fn = vlr.series.info if kind == "info" else vlr.series.players
    for attempt in range(3):
        try:
            return fn(mid).model_dump(mode="json")
        except Exception as ex:
            print(f"   {kind} {mid} failed ({type(ex).__name__}: {str(ex)[:80]}), retrying...")
            time.sleep(3)
    return None

def g(d, *keys):
    """Safe nested lookup: g(data, 'team1', 'team_id')."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

failed = []
try:
    for i, (kind, mid) in enumerate(todo, 1):
        data = fetch(kind, mid)
        if data is None:
            failed.append((kind, mid))
            print(f"[{i}/{len(todo)}] {kind} {mid}: FAILED, skipped")
        else:
            path_for(kind, mid).write_text(json.dumps(data), encoding="utf-8")
            if i <= 5 or i % 25 == 0:
                print(f"[{i}/{len(todo)}] {kind} {mid}: ok")
        time.sleep(PAUSE)
finally:
    # ---------- combine everything cached into two CSVs ----------
    try:
        games, players = [], []
        for mid in ids:
            p = path_for("info", mid)
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                for gm in d.get("games") or []:
                    games.append({
                        "match_id": mid, "datetime": d.get("datetime"),
                        "event_name": d.get("event_name"), "best_of": d.get("best_of"),
                        "patch": d.get("patch"),
                        "team1_id": g(d, "team1", "id"), "team1_tag": g(d, "team1", "tag"),
                        "team2_id": g(d, "team2", "id"), "team2_tag": g(d, "team2", "tag"),
                        "game_id": gm.get("game_id"), "order": gm.get("order"),
                        "map_name": gm.get("map_name"), "picked_by": gm.get("picked_by"),
                        "played": gm.get("played"),
                        "team1_score": gm.get("team1_score"), "team2_score": gm.get("team2_score"),
                        "team1_attack_rounds": gm.get("team1_attack_rounds"),
                        "team1_defense_rounds": gm.get("team1_defense_rounds"),
                        "team2_attack_rounds": gm.get("team2_attack_rounds"),
                        "team2_defense_rounds": gm.get("team2_defense_rounds"),
                    })
            p = path_for("players", mid)
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                for side in ("team1", "team2"):
                    for pl in g(d, side, "players") or []:
                        st = pl.get("stats") or {}
                        ov = st.get("overall") or {}
                        players.append({
                            "match_id": mid, "team_id": g(d, side, "team_id"),
                            "team_name": g(d, side, "team_name"),
                            "player_id": pl.get("player_id"), "player": pl.get("name"),
                            "agents": "|".join(pl.get("agents") or []),
                            "rating": ov.get("rating"), "acs": ov.get("acs"),
                            "kills": ov.get("kills"), "deaths": ov.get("deaths"),
                            "assists": ov.get("assists"), "kast": ov.get("kast"),
                            "adr": ov.get("adr"),
                            "first_kills": ov.get("first_kills"),
                            "first_deaths": ov.get("first_deaths"),
                            "attack_rating": g(st, "attack", "rating"),
                            "defend_rating": g(st, "defend", "rating"),
                        })
        pd.DataFrame(games).to_csv("data/series_games.csv", index=False)
        pd.DataFrame(players).to_csv("data/series_players.csv", index=False)
        print(f"\nSaved {len(games)} map rows to data/series_games.csv")
        print(f"Saved {len(players)} player rows to data/series_players.csv")
    except Exception as ex:
        print(f"\nCombining step failed ({type(ex).__name__}: {ex}). "
              "Your downloaded files are safe in data/cache/series/.")
    if failed:
        print(f"{len(failed)} requests failed. Run the script again to retry them.")