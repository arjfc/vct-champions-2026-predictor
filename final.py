import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import PLAYOFF, load_matches, build_features, win_prob

WARMUP = 0.30     # same as evaluate.py
SHRINK = 10       # same as add_player_rating in features.py
N = 20_000
ID = {name: tid for tid, name in PLAYOFF.items()}

# ---------- 1. features for every past match, then train both models ----------
df = load_matches()
df, ratings = build_features(df)          # ratings = Elo after the last match in the data
d = df.iloc[int(len(df) * WARMUP):].reset_index(drop=True)

def fit(train, cols):
    X, y = train[cols].to_numpy(float), train.a_won.to_numpy()
    X2, y2 = np.vstack([X, -X]), np.concatenate([y, 1 - y])    # swap teams so order can't bias it
    return make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(fit_intercept=False)).fit(X2, y2)

m_elo = fit(d, ["elo_diff"])
m_full = fit(d, ["elo_diff", "player_diff"])
print("Standardized coefficients (Elo + player model):",
      dict(zip(["elo_diff", "player_diff"], m_full[-1].coef_[0].round(3))))

# ---------- 2. current lineups and player ratings ----------
pl = pd.read_csv("data/series_players.csv",
                 usecols=["match_id", "team_id", "player_id", "player", "rating"])
pl = pl.dropna(subset=["team_id", "player_id"]).copy()
pl["team_id"] = pl.team_id.astype(int)
pl["player_id"] = pl.player_id.astype(int)
gm = pl.rating.mean()
st = pl.dropna(subset=["rating"]).groupby("player_id").rating.agg(["sum", "count"])
shrunk = (st["sum"] + SHRINK * gm) / (st["count"] + SHRINK)      # each player's full-history rating
pl = pl.merge(df[["match_id", "datetime"]], on="match_id")

team_rating, lineup = {}, {}
for tid, name in PLAYOFF.items():
    t = pl[pl.team_id == tid].sort_values("datetime")
    row = t[t.match_id == t.match_id.iloc[-1]]                    # the team's most recent match
    team_rating[tid] = float(np.mean([shrunk.get(p, gm) for p in row.player_id]))
    lineup[tid] = ", ".join(f"{n} {shrunk.get(p, gm):.2f}" for n, p in zip(row.player, row.player_id))

print("\nLineups used (latest match in the data, shrunk rating per player):")
for tid, name in PLAYOFF.items():
    print(f"  {name}: {lineup[tid]}")

# ---------- 3. win chance for every pair, under three models ----------
P = {"Elo (raw)": {}, "Elo (logistic)": {}, "Elo + player": {}}
for a in PLAYOFF:
    for b in PLAYOFF:
        if a == b:
            continue
        ed = ratings[a] - ratings[b]
        pdif = team_rating[a] - team_rating[b]
        P["Elo (raw)"][(a, b)] = win_prob(ratings[a], ratings[b])
        P["Elo (logistic)"][(a, b)] = m_elo.predict_proba(np.array([[ed]]))[0, 1]
        P["Elo + player"][(a, b)] = m_full.predict_proba(np.array([[ed, pdif]]))[0, 1]
P["Blend 50/50"] = {k: (P["Elo (logistic)"][k] + P["Elo + player"][k]) / 2 for k in P["Elo (logistic)"]}
pd.DataFrame([{"team_a": PLAYOFF[a], "team_b": PLAYOFF[b],
               "p_elo_logistic": P["Elo (logistic)"][(a, b)],
               "p_elo_player": P["Elo + player"][(a, b)],
               "p_blend": P["Blend 50/50"][(a, b)]}
              for (a, b) in P["Elo (raw)"]]).to_csv("data/pair_probs_final.csv", index=False)

# ---------- 4. bracket simulation (same random draws for every model) ----------
def simulate(p, seed=42):
    rng = np.random.default_rng(seed)
    def play(a, b):
        return (a, b) if rng.random() < p[(a, b)] else (b, a)
    titles = dict.fromkeys(PLAYOFF, 0)
    finals = dict.fromkeys(PLAYOFF, 0)
    for _i in range(N):
        w1, l1 = play(ID["100 Thieves"], ID["G2 Esports"])
        w2, l2 = play(ID["Team Vitality"], ID["Nongshim RedForce"])
        w3, l3 = play(ID["NRG"], ID["T1"])
        w4, l4 = play(ID["Paper Rex"], ID["LOUD"])
        wsf1, lsf1 = play(w1, w2)
        wsf2, lsf2 = play(w3, w4)
        wu, lu = play(wsf1, wsf2)
        wl1, _x = play(l1, l2)
        wl2, _x = play(l3, l4)
        wa, _x = play(wl1, lsf2)
        wb, _x = play(wl2, lsf1)
        wr3, _x = play(wa, wb)
        wlf, _x = play(lu, wr3)
        champ, _x = play(wu, wlf)
        titles[champ] += 1
        finals[wu] += 1
        finals[wlf] += 1
    return {t: titles[t] / N for t in PLAYOFF}, {t: finals[t] / N for t in PLAYOFF}

res = {name: simulate(p) for name, p in P.items()}

rows = []
for tid, name in PLAYOFF.items():
    rows.append({
        "team": name, "elo": round(ratings[tid]), "player_rating": round(team_rating[tid], 3),
        "title_elo_raw": res["Elo (raw)"][0][tid],
        "title_elo": res["Elo (logistic)"][0][tid],
        "title_elo_player": res["Elo + player"][0][tid],
        "final_elo": res["Elo (logistic)"][1][tid],
        "final_elo_player": res["Elo + player"][1][tid],
        "title_blend": res["Blend 50/50"][0][tid],
        "final_blend": res["Blend 50/50"][1][tid],
    })
out = pd.DataFrame(rows).sort_values("title_blend", ascending=False).reset_index(drop=True)
out.to_csv("data/title_odds_compare.csv", index=False)

pct = lambda s: (s * 100).round(1).astype(str) + "%"
show = pd.DataFrame({
    "team": out.team, "elo": out.elo, "player rtg": out.player_rating,
    "Elo raw": pct(out.title_elo_raw),
    "Elo": pct(out.title_elo),
    "Elo+player": pct(out.title_elo_player),
    "Blend": pct(out.title_blend),
})
print(f"\n=== Title odds ({N:,} simulated brackets per model) ===")
print("'Elo raw' should land within a few tenths of your earlier table (sanity check).")
print("Blend = average of the Elo and Elo+player win chances; the table is sorted by it.\n")
print(show.to_string(index=False))

print("\nUpper Quarterfinal win chances (Elo | Elo+player | Blend):")
for a, b in [("100 Thieves", "G2 Esports"), ("Team Vitality", "Nongshim RedForce"),
             ("NRG", "T1"), ("Paper Rex", "LOUD")]:
    k = (ID[a], ID[b])
    print(f"  {a} vs {b}: {P['Elo (logistic)'][k]:.0%} | {P['Elo + player'][k]:.0%} | {P['Blend 50/50'][k]:.0%}")

# ---------- 5. chart ----------
o = out.sort_values("title_elo_player")
fig = go.Figure()
fig.add_bar(y=o.team, x=o.title_elo * 100, orientation="h", name="Elo only", marker_color="#b7c4d6")
fig.add_bar(y=o.team, x=o.title_elo_player * 100, orientation="h", name="Elo + player rating",
            marker_color="#e0485a", text=(o.title_elo_player * 100).round(1).astype(str) + "%",
            textposition="outside")
fig.update_layout(barmode="group", title="Predicted title odds: Elo vs Elo + player rating",
                  xaxis_title="Probability (%)", legend_orientation="h", legend_y=-0.15,
                  height=520, margin=dict(l=150))
fig.write_html("data/title_odds_compare.html", include_plotlyjs="cdn")
print("\nSaved data/title_odds_compare.csv, data/pair_probs_final.csv, data/title_odds_compare.html")