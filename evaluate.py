import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import load_matches, build_features

WARMUP = 0.30
N_CHUNKS = 5

# Every model to test: a name and the feature columns it uses.
# New experiments are added here, nothing else changes.
MODELS = {
    "Elo only (logistic)": ["elo_diff"],
    "Elo + player rating": ["elo_diff", "player_diff"],
    "Map Elo only": ["map_elo_diff"],
    "Elo + map adjustment": ["elo_diff", "map_adj_diff"],
    "Elo + player + map adj": ["elo_diff", "player_diff", "map_adj_diff"],
}
# MODELS = {
#     "Elo only (logistic)": ["elo_diff"],
#     "Player rating only": ["player_diff"],
#     "Elo + player rating": ["elo_diff", "player_diff"],
# }
# MODELS = {
#     "Elo only (logistic)": ["elo_diff"],
#     # later: "Elo + player rating": ["elo_diff", "player_diff"],
# }


def fit(train, cols):
    X, y = train[cols].to_numpy(float), train.a_won.to_numpy()
    X2, y2 = np.vstack([X, -X]), np.concatenate([y, 1 - y])   # swap teams so order can't bias it
    return make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(fit_intercept=False)).fit(X2, y2)


def losses(p, y):
    p = np.clip(p, 0.001, 0.999)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def walk_forward(d, cols):
    """Always train on the past, test on the next chunk."""
    edges = np.linspace(0, len(d), N_CHUNKS + 1).astype(int)
    tests, preds = [], []
    for k in range(1, N_CHUNKS):
        train, test = d.iloc[:edges[k]], d.iloc[edges[k]:edges[k + 1]]
        preds.append(fit(train, cols).predict_proba(test[cols].to_numpy(float))[:, 1])
        tests.append(test)
    return pd.concat(tests), np.concatenate(preds)


# ---------- build features, then test ----------
df = load_matches()
df, ratings = build_features(df)
d = df.iloc[int(len(df) * WARMUP):].reset_index(drop=True)
print(f"Matches with a usable map feature: {(d.map_adj_diff != 0).mean():.0%}")
print(f"Correlation of map adjustment with Elo diff: {d.elo_diff.corr(d.map_adj_diff):.2f}")
print(f"Matches with a usable player feature: {(d.player_diff != 0).mean():.0%}")
print(f"Correlation between Elo diff and player diff: {d.elo_diff.corr(d.player_diff):.2f}")
print(f"Matches: {len(df)} | after warm-up: {len(d)} | Bo1 matches overall: {(df.best_of == 1).sum()}")
print("Events counted as international:")
for ev, n in df[df.is_international].event.value_counts().items():
    print(f"   {ev}: {n} matches")

t, preds = None, {}
for name, cols in MODELS.items():
    t, preds[name] = walk_forward(d, cols)
y = t.a_won.to_numpy()
preds = {"Elo (raw, baseline)": t.elo_p.to_numpy(), **preds}
base_loss = losses(preds["Elo (raw, baseline)"], y)

subsets = {
    "ALL test matches": np.ones(len(t), bool),
    "INTERNATIONAL events only": t.is_international.to_numpy(),
    "Bo3/Bo5 only (no Bo1)": t.best_of.isin([3, 5]).to_numpy(),
}

for label, mask in subsets.items():
    print(f"\n=== {label} ({mask.sum()} matches) ===")
    print(f"{'model':<24}{'accuracy':>10}{'log-loss':>10}   vs baseline (+ = better)")
    for name, p in preds.items():
        ll = losses(p[mask], y[mask])
        acc = ((p[mask] > 0.5) == (y[mask] == 1)).mean()
        if name.startswith("Elo (raw"):
            cmp = "-"
        else:
            gain = base_loss[mask] - ll
            se = gain.std(ddof=1) / np.sqrt(len(gain))
            cmp = f"{gain.mean():+.4f} ±{1.96 * se:.4f}"
        print(f"{name:<24}{acc:>10.1%}{ll.mean():>10.3f}   {cmp}")

print("\nA new model only counts as better if its 'vs baseline' number is positive AND larger than its ±margin.")
print("\n=== Does the map feature add anything ON TOP of Elo + player rating? ===")
a, b = "Elo + player + map adj", "Elo + player rating"
for label, mask in subsets.items():
    gain = losses(preds[b][mask], y[mask]) - losses(preds[a][mask], y[mask])
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    print(f"{label:<28}{gain.mean():+.4f} ±{1.96 * se:.4f}   (+ = map feature helps)")

print("\n=== 50/50 blend of 'Elo only' and 'Elo + player rating' ===")
blend = (preds["Elo only (logistic)"] + preds["Elo + player rating"]) / 2
for label, mask in subsets.items():
    ll = losses(blend[mask], y[mask])
    acc = ((blend[mask] > 0.5) == (y[mask] == 1)).mean()
    gain = base_loss[mask] - ll
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    print(f"{label:<28} accuracy {acc:.1%}  log-loss {ll.mean():.3f}  vs baseline {gain.mean():+.4f} ±{1.96 * se:.4f}")