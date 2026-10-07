import numpy as np
import pandas as pd
from pathlib import Path

PLAYOFF = {120: "100 Thieves", 11058: "G2 Esports", 2059: "Team Vitality",
           11060: "Nongshim RedForce", 1034: "NRG", 14: "T1",
           624: "Paper Rex", 6961: "LOUD"}
WARMUP = 0.30

# ---------- load (same filtering as elo.py) ----------
df = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
events = set(df[df.involves_playoff_team]["event"])
df = df[df.event.isin(events)].dropna(subset=["winner_id"]).copy()
df["winner_id"] = df["winner_id"].astype(int)
df = df.sort_values("datetime").reset_index(drop=True)
df["a_won"] = (df.winner_id == df.team1_id).astype(int)

# how decisive was the series? 2-0 -> 1.0, 2-1 -> 0.33, 3-2 -> 0.2
s1, s2 = df.team1_score.fillna(0), df.team2_score.fillna(0)
w, l = np.maximum(s1, s2), np.minimum(s1, s2)
df["margin_frac"] = np.where(w + l > 0, (w - l) / (w + l), 0.5)

# ---------- Elo with two optional upgrades ----------
def win_prob(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def run_elo(data, k, margin, half_life):
    """margin: 0 = ignore score margin. half_life: None = no decay, else days.
    Decay pulls a team's rating back toward 1500 while it is inactive."""
    ratings, last_seen, probs = {}, {}, []
    for r in data.itertuples():
        a, b = r.team1_id, r.team2_id
        if half_life:
            for t in (a, b):
                if t in last_seen:
                    gap = (r.datetime - last_seen[t]).total_seconds() / 86400
                    ratings[t] = 1500 + (ratings[t] - 1500) * 0.5 ** (gap / half_life)
        ra, rb = ratings.get(a, 1500), ratings.get(b, 1500)
        p = win_prob(ra, rb)                      # prediction BEFORE the result
        probs.append(p)
        mult = 1 + margin * (r.margin_frac - 0.5)  # bigger win -> bigger update
        delta = k * mult * (r.a_won - p)
        ratings[a], ratings[b] = ra + delta, rb - delta
        last_seen[a] = last_seen[b] = r.datetime
    return ratings, np.array(probs)

def losses(p, y):
    p = np.clip(p, 0.001, 0.999)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# ---------- evaluation window (same 620 matches as model.py) ----------
start = int(len(df) * WARMUP)
ev0 = start + (len(df) - start) // 5
mid = ev0 + (len(df) - ev0) // 2
y = df.a_won.to_numpy()
print(f"Matches: {len(df)} | evaluation window: {len(df) - ev0}")
print(f"  tune half: {mid - ev0} matches | confirm half: {len(df) - mid} matches\n")

def label(c):
    k, m, hl = c
    return f"K={k:<2} margin={m:<3} half-life={hl if hl else 'off'}"

BASE = (24, 0.0, None)                     # plain Elo, as before
grid = [(k, m, hl) for k in (16, 24, 32) for m in (0.0, 0.5, 1.0)
        for hl in (None, 365, 180, 90)]

results = {}
for c in grid:
    _, p = run_elo(df, *c)
    results[c] = losses(p, y)              # per-match log-loss for every match

def tune(c):    return results[c][ev0:mid].mean()
def confirm(c): return results[c][mid:].mean()

# ---------- pick on the tune half, check on the confirm half ----------
print("Top 5 settings on the TUNE half (plus the plain-Elo baseline):")
print(f"{'setting':<38}{'tune':>8}{'confirm':>9}")
top = sorted(grid, key=tune)[:5]
for c in top + ([BASE] if BASE not in top else []):
    tag = "  <- baseline" if c == BASE else ""
    print(f"{label(c):<38}{tune(c):>8.4f}{confirm(c):>9.4f}{tag}")

best = top[0]
print(f"\nChosen on the tune half: {label(best)}")
if best == BASE:
    print("Plain Elo was already the best setting. The upgrades don't help.")
    use = BASE
else:
    gain = results[BASE][mid:] - results[best][mid:]      # positive = upgrade is better
    m_, se = gain.mean(), gain.std(ddof=1) / np.sqrt(len(gain))
    print(f"On the confirm half (data it was NOT chosen on), log-loss changes by "
          f"{m_:+.4f} per match (margin of error +/-{1.96 * se:.4f})")
    if m_ - 1.96 * se > 0:
        print("Verdict: the upgrade clearly helps.")
    elif m_ > 0:
        print("Verdict: slightly better, but within the noise. Not proven.")
    else:
        print("Verdict: the upgrade does NOT help on fresh data.")
    use = best if m_ > 0 else BASE

# ---------- final ratings ----------
print(f"\nUsing: {label(use)}\n")
ratings, _ = run_elo(df, *use)
table = pd.DataFrame([(name, round(ratings.get(i, 1500))) for i, name in PLAYOFF.items()],
                     columns=["team", "elo"]).sort_values("elo", ascending=False)
old = Path("data/elo_ratings.csv")
if old.exists():
    table = table.merge(pd.read_csv(old).rename(columns={"elo": "old_elo"}), on="team", how="left")
    table = table.sort_values("elo", ascending=False)
print(table.to_string(index=False))
table.to_csv("data/elo2_ratings.csv", index=False)

pairs = [{"team_a": PLAYOFF[a], "team_b": PLAYOFF[b],
          "p_a_wins": win_prob(ratings.get(a, 1500), ratings.get(b, 1500))}
         for a in PLAYOFF for b in PLAYOFF if a != b]
pd.DataFrame(pairs).to_csv("data/pair_probs_elo2.csv", index=False)

print("\nUpper Quarterfinal win chances:")
for a, b in [(120, 11058), (2059, 11060), (1034, 14), (624, 6961)]:
    p = win_prob(ratings.get(a, 1500), ratings.get(b, 1500))
    print(f"  {PLAYOFF[a]} {p:.0%}  vs  {PLAYOFF[b]} {1 - p:.0%}")
print("\nSaved data/elo2_ratings.csv and data/pair_probs_elo2.csv")