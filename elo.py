import numpy as np
import pandas as pd

PLAYOFF = {120: "100 Thieves", 11058: "G2 Esports", 2059: "Team Vitality",
           11060: "Nongshim RedForce", 1034: "NRG", 14: "T1",
           624: "Paper Rex", 6961: "LOUD"}

# ---------- load and filter ----------
df = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
events = set(df[df.involves_playoff_team]["event"])        # events your 8 teams played in
df = df[df.event.isin(events)].dropna(subset=["winner_id"]).copy()
df["winner_id"] = df["winner_id"].astype(int)
df = df.sort_values("datetime").reset_index(drop=True)
print(f"Using {len(df)} matches from {len(events)} events\n")

# ---------- Elo ----------
def win_prob(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def run_elo(df, k):
    ratings, rows = {}, []
    for r in df.itertuples():
        a, b = r.team1_id, r.team2_id
        ra, rb = ratings.get(a, 1500), ratings.get(b, 1500)
        p = win_prob(ra, rb)                       # prediction made BEFORE the result
        a_won = 1 if r.winner_id == a else 0
        rows.append((a, b, p, a_won))
        ratings[a] = ra + k * (a_won - p)          # update AFTER
        ratings[b] = rb - k * (a_won - p)
    return ratings, pd.DataFrame(rows, columns=["a", "b", "p", "a_won"])

def evaluate(preds, warmup=0.3):
    ev = preds.iloc[int(len(preds) * warmup):]     # skip early matches while ratings settle
    p = ev.p.clip(0.001, 0.999)
    acc = ((ev.p > 0.5) == (ev.a_won == 1)).mean()
    ll = -(ev.a_won * np.log(p) + (1 - ev.a_won) * np.log(1 - p)).mean()
    mask = ev.a.isin(PLAYOFF) | ev.b.isin(PLAYOFF)
    acc_mine = ((ev.p[mask] > 0.5) == (ev.a_won[mask] == 1)).mean()
    return acc, ll, acc_mine, len(ev)

# ---------- try a few K values ----------
print("K     accuracy   log-loss   acc (your 8 teams)")
best = None
for k in [16, 24, 32, 48, 64]:
    ratings, preds = run_elo(df, k)
    acc, ll, acc_mine, n = evaluate(preds)
    print(f"{k:<5} {acc:.1%}      {ll:.3f}      {acc_mine:.1%}")
    if best is None or ll < best[1]:
        best = (k, ll, ratings)
print(f"\n(tested on {n} matches after a warm-up period; coin flip = 50%, log-loss 0.693)")

k, _, ratings = best
print(f"Best K by log-loss: {k}\n")

# ---------- ratings and first-round predictions ----------
table = pd.DataFrame(
    [(name, round(ratings.get(i, 1500))) for i, name in PLAYOFF.items()],
    columns=["team", "elo"]).sort_values("elo", ascending=False)
print(table.to_string(index=False))
table.to_csv("data/elo_ratings.csv", index=False)

print("\nUpper Quarterfinal win chances:")
for a, b in [(120, 11058), (2059, 11060), (1034, 14), (624, 6961)]:
    p = win_prob(ratings.get(a, 1500), ratings.get(b, 1500))
    print(f"  {PLAYOFF[a]} {p:.0%}  vs  {PLAYOFF[b]} {1 - p:.0%}")