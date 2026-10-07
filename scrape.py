import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import vlrdevapi as vlr

PLAYOFF_IDS = {120, 11058, 2059, 11060, 1034, 14, 624, 6961}
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)  # stop going further back than this
MAX_PAGES = 100   # safety limit
PAUSE = 1.5       # seconds between requests (be polite to VLR)

FIELDS = ["match_id", "datetime", "event", "stage",
          "team1_id", "team1_name", "team1_score",
          "team2_id", "team2_name", "team2_score",
          "winner_id", "involves_playoff_team", "url"]

def fetch_page(n):
    for attempt in range(3):
        try:
            return vlr.matches.completed(page=n)
        except Exception as e:
            print(f"  page {n} failed ({e}), retrying...")
            time.sleep(5)
    return None

rows, seen = [], set()
try:
    for n in range(1, MAX_PAGES + 1):
        page = fetch_page(n)
        if page is None or not page.matches:
            print("No more data, stopping.")
            break

        for m in page.matches:
            if m.match_id in seen:
                continue

            seen.add(m.match_id)
            t1, t2 = m.team1, m.team2
            
            if t1 is None or t2 is None:
                continue
            winner = t1.id if t1.is_winner else (t2.id if t2.is_winner else None)
            rows.append({
                "match_id": m.match_id,
                "datetime": m.datetime.isoformat() if m.datetime else "",
                "event": m.event, "stage": m.stage,
                "team1_id": t1.id, "team1_name": t1.name, "team1_score": t1.score,
                "team2_id": t2.id, "team2_name": t2.name, "team2_score": t2.score,
                "winner_id": winner,
                "involves_playoff_team": t1.id in PLAYOFF_IDS or t2.id in PLAYOFF_IDS,
                "url": m.url,
            })

        dates = [m.datetime for m in page.matches if m.datetime]
        print(f"Page {n}: {len(page.matches)} matches (total {len(rows)})")

        if dates and min(dates) < START_DATE:
            print("Reached start date, stopping.")
            break
        if not page.has_next_page:
            print("Last page reached.")
            break
        time.sleep(PAUSE)
finally:
    Path("data").mkdir(exist_ok=True)
    with open("data/matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    mine = sum(r["involves_playoff_team"] for r in rows)
    print(f"Saved {len(rows)} matches to data/matches.csv ({mine} involve your 8 teams)")