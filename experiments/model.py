import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PLAYOFF = {120: "100 Thieves", 11058: "G2 Esports", 2059: "Team Vitality",
           11060: "Nongshim RedForce", 1034: "NRG", 14: "T1",
           624: "Paper Rex", 6961: "LOUD"}
K_ELO = 24       # best value from elo.py
DECAY = 0.6      # map results from one quarter earlier count 60% as much
SHRINK = 4       # pseudo-games that pull small samples toward the team's average
MIN_GAMES = 5    # less map history than this -> map features are neutral (0)
WARMUP = 0.30    # skip the first 30% of matches while Elo settles
N_CHUNKS = 5

# ---------- 1. matches + pre-match Elo (same logic as elo.py) ----------
df = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
events = set(df[df.involves_playoff_team]["event"])
df = df[df.event.isin(events)].dropna(subset=["winner_id"]).copy()
df["winner_id"] = df["winner_id"].astype(int)
df = df.sort_values("datetime").reset_index(drop=True)

def win_prob(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

ratings, elo_diff, elo_p = {}, [], []
for r in df.itertuples():
    a, b = r.team1_id, r.team2_id
    ra, rb = ratings.get(a, 1500), ratings.get(b, 1500)
    p = win_prob(ra, rb)
    elo_diff.append(ra - rb)
    elo_p.append(p)
    a_won = 1 if r.winner_id == a else 0
    ratings[a] = ra + K_ELO * (a_won - p)
    ratings[b] = rb - K_ELO * (a_won - p)
df["elo_diff"] = elo_diff
df["elo_p"] = elo_p
df["a_won"] = (df.winner_id == df.team1_id).astype(int)

# ---------- 2. map-pool features (only from quarters BEFORE the match) ----------
ms = pd.read_csv("data/team_map_stats.csv", parse_dates=["window_start"])
ms = ms.dropna(subset=["map_name"]).copy()
ms["q"] = ms.window_start.dt.year * 4 + (ms.window_start.dt.month - 1) // 3
by_team = {t: g for t, g in ms.groupby("team_id")}
_cache = {}

def profile(team, q):
    """Shrunk, recency-weighted win rate per map, using only quarters before q."""
    key = (int(team), int(q))
    if key in _cache:
        return _cache[key]
    out = None
    g = by_team.get(int(team))
    if g is not None:
        g = g[g.q < q]
        if len(g):
            w = DECAY ** (q - 1 - g.q)
            wins = (g.wins.fillna(0) * w).groupby(g.map_name).sum()
            games = (g.games_played.fillna(0) * w).groupby(g.map_name).sum()
            if games.sum() >= MIN_GAMES:
                overall = wins.sum() / games.sum()
                rate = (wins + SHRINK * overall) / (games + SHRINK)
                out = (overall, rate.to_dict())
    _cache[key] = out
    return out

def map_features(a, b, q):
    pa, pb = profile(a, q), profile(b, q)
    if pa is None or pb is None:
        return 0.0, 0.0, 0
    (oa, ra), (ob, rb) = pa, pb
    maps = set(ra) | set(rb)
    mean_diff = np.mean([ra.get(m, oa) - rb.get(m, ob) for m in maps])
    best3_diff = (np.mean(sorted(ra.values(), reverse=True)[:3])
                  - np.mean(sorted(rb.values(), reverse=True)[:3]))
    return mean_diff, best3_diff, 1

q = df.datetime.dt.year * 4 + (df.datetime.dt.month - 1) // 3
feats = [map_features(a, b, qq) for a, b, qq in zip(df.team1_id, df.team2_id, q)]
df["map_mean_diff"], df["map_best3_diff"], df["has_map"] = zip(*feats)

d = df.iloc[int(len(df) * WARMUP):].reset_index(drop=True)
print(f"Matches: {len(df)} | used after warm-up: {len(d)}")
print(f"Matches where both teams have map history: {d.has_map.mean():.0%}\n")

# ---------- 3. models ----------
ELO_COLS = ["elo_diff"]
FULL_COLS = ["elo_diff", "map_mean_diff", "map_best3_diff"]

def fit(train, cols):
    X, y = train[cols].to_numpy(float), train.a_won.to_numpy()
    # add each match with the teams swapped, so team order can't bias the model
    X2, y2 = np.vstack([X, -X]), np.concatenate([y, 1 - y])
    return make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(fit_intercept=False)).fit(X2, y2)

def predict(model, test, cols):
    return model.predict_proba(test[cols].to_numpy(float))[:, 1]

def losses(p, y):
    p = np.clip(p, 0.001, 0.999)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# walk-forward test: always train on the past, test on the next chunk
edges = np.linspace(0, len(d), N_CHUNKS + 1).astype(int)
ys, p_raw, p_elo, p_full = [], [], [], []
for k in range(1, N_CHUNKS):
    train, test = d.iloc[:edges[k]], d.iloc[edges[k]:edges[k + 1]]
    ys.append(test.a_won.to_numpy())
    p_raw.append(test.elo_p.to_numpy())
    p_elo.append(predict(fit(train, ELO_COLS), test, ELO_COLS))
    p_full.append(predict(fit(train, FULL_COLS), test, FULL_COLS))
y, p_raw, p_elo, p_full = map(np.concatenate, (ys, p_raw, p_elo, p_full))

print(f"Tested on {len(y)} matches (always trained on earlier matches only)")
print(f"{'model':<24}{'accuracy':>10}{'log-loss':>10}")
for name, p in [("Elo (raw)", p_raw), ("Elo only (logistic)", p_elo),
                ("Elo + map features", p_full)]:
    acc = ((p > 0.5) == (y == 1)).mean()
    print(f"{name:<24}{acc:>10.1%}{losses(p, y).mean():>10.3f}")

gain = losses(p_elo, y) - losses(p_full, y)       # positive = map features help
m, se = gain.mean(), gain.std(ddof=1) / np.sqrt(len(gain))
print(f"\nMap features change log-loss by {m:+.4f} per match (margin of error ±{1.96 * se:.4f})")
if m - 1.96 * se > 0:
    print("Verdict: map features clearly help.")
elif m > 0:
    print("Verdict: slightly better, but within the noise. Not proven.")
else:
    print("Verdict: map features do NOT help. Elo alone is as good or better.")

# ---------- 4. final model + win probabilities for the 8 playoff teams ----------
cols = FULL_COLS if m > 0 else ELO_COLS
final = fit(d, cols)
print(f"\nFinal model uses: {cols}")
print("Standardized coefficients:", dict(zip(cols, final[-1].coef_[0].round(3))))

q_now = int(df.datetime.max().year * 4 + (df.datetime.max().month - 1) // 3)
print("\nBest maps per team (shrunk win rate, recent quarters weighted more):")
for t, name in PLAYOFF.items():
    pr = profile(t, q_now)
    if pr is None:
        print(f"  {name}: no map history")
    else:
        top = sorted(pr[1].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {name}: " + ", ".join(f"{mp} {r:.0%}" for mp, r in top))

pairs = []
for a in PLAYOFF:
    for b in PLAYOFF:
        if a != b:
            md, b3, _ = map_features(a, b, q_now)
            pairs.append({"team_a": PLAYOFF[a], "team_b": PLAYOFF[b], "elo_diff": ratings[a] - ratings[b],
                          "map_mean_diff": md, "map_best3_diff": b3})
pairs = pd.DataFrame(pairs)
pairs["p_a_wins"] = predict(final, pairs, cols)
pairs[["team_a", "team_b", "p_a_wins"]].to_csv("data/pair_probs.csv", index=False)

print("\nUpper Quarterfinal win chances (new model):")
for a, b in [("100 Thieves", "G2 Esports"), ("Team Vitality", "Nongshim RedForce"),
             ("NRG", "T1"), ("Paper Rex", "LOUD")]:
    p = pairs[(pairs.team_a == a) & (pairs.team_b == b)].p_a_wins.iloc[0]
    print(f"  {a} {p:.0%}  vs  {b} {1 - p:.0%}")
print("\nSaved data/pair_probs.csv")