"""Shared data loading and feature building. One small function per feature."""
import numpy as np
import pandas as pd

PLAYOFF = {120: "100 Thieves", 11058: "G2 Esports", 2059: "Team Vitality",
           11060: "Nongshim RedForce", 1034: "NRG", 14: "T1",
           624: "Paper Rex", 6961: "LOUD"}
K_ELO = 24
INTERNATIONAL_KEYWORDS = ["Masters", "Valorant Champions", "Esports World Cup"]


def is_international(event):
    return any(k in event for k in INTERNATIONAL_KEYWORDS) and "Qualifier" not in event


def load_matches():
    """Same filtering as elo.py and model.py, plus best-of format and an international flag."""
    df = pd.read_csv("data/matches.csv", parse_dates=["datetime"])
    events = set(df[df.involves_playoff_team]["event"])
    df = df[df.event.isin(events)].dropna(subset=["winner_id"]).copy()
    df["winner_id"] = df["winner_id"].astype(int)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["a_won"] = (df.winner_id == df.team1_id).astype(int)

    try:
        bo = (pd.read_csv("data/series_games.csv", usecols=["match_id", "best_of"])
              .drop_duplicates("match_id"))
        df = df.merge(bo, on="match_id", how="left")
    except FileNotFoundError:
        df["best_of"] = np.nan
    df["is_international"] = [is_international(e) for e in df.event]
    return df


def win_prob(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def add_elo(df, k=K_ELO):
    """Feature: Elo difference, using ratings from BEFORE each match."""
    ratings, diffs, probs = {}, [], []
    for r in df.itertuples():
        a, b = r.team1_id, r.team2_id
        ra, rb = ratings.get(a, 1500), ratings.get(b, 1500)
        p = win_prob(ra, rb)
        diffs.append(ra - rb)
        probs.append(p)
        a_won = 1 if r.winner_id == a else 0
        ratings[a] = ra + k * (a_won - p)
        ratings[b] = rb - k * (a_won - p)
    out = df.copy()
    out["elo_diff"] = diffs
    out["elo_p"] = probs
    return out, ratings

def add_player_rating(df, shrink=10):
    """Feature: average pre-match rating of the 5 players who played, Team A minus Team B.
    Each player's rating uses only PAST series and is pulled toward the overall average
    (as if every player started with `shrink` average series). Missing data -> 0 (neutral)."""
    pl = pd.read_csv("data/series_players.csv",
                     usecols=["match_id", "team_id", "player_id", "rating"])
    pl = pl.dropna(subset=["team_id", "player_id"]).copy()
    pl["team_id"] = pl.team_id.astype(int)
    pl["player_id"] = pl.player_id.astype(int)
    by_match = {mid: g for mid, g in pl.groupby("match_id")}

    psum, pcnt = {}, {}          # running rating sum and series count per player
    gsum, gcnt = 0.0, 0          # running overall average
    diffs = []
    for r in df.itertuples():
        g = by_match.get(r.match_id)
        if g is None:
            diffs.append(0.0)
            continue
        gm = gsum / gcnt if gcnt >= 50 else 1.0
        vals = {}
        for tid in (r.team1_id, r.team2_id):
            t = g[g.team_id == tid]
            if t.empty or t.rating.notna().sum() == 0:
                vals[tid] = None                       # no stats for this match
                continue
            vals[tid] = np.mean([(psum.get(p, 0.0) + shrink * gm) / (pcnt.get(p, 0) + shrink)
                                 for p in t.player_id])
        a, b = vals[r.team1_id], vals[r.team2_id]
        diffs.append(a - b if a is not None and b is not None else 0.0)

        for row in g.itertuples():                     # update AFTER the prediction
            if pd.notna(row.rating):
                psum[row.player_id] = psum.get(row.player_id, 0.0) + row.rating
                pcnt[row.player_id] = pcnt.get(row.player_id, 0) + 1
                gsum += row.rating
                gcnt += 1
    out = df.copy()
    out["player_diff"] = diffs
    return out
def add_map_elo(df, k_overall=16, k_map=8, window_days=60, min_pool=20):
    """Two features built from individual MAP results (series_games.csv):
      map_adj_diff: the map-specific part only. Each team has a rating per map, stored as an
                    adjustment on top of its overall map rating. It moves slowly (small K), so
                    it is heavily shrunk toward 0. Averaged over the maps played recently
                    (a proxy for the current map pool), Team A minus Team B.
      map_elo_diff: overall map-level Elo difference plus that same adjustment.
    Everything uses only maps from BEFORE the match. Too little recent data -> 0 (neutral)."""
    g = pd.read_csv("data/series_games.csv",
                    usecols=["match_id", "order", "map_name", "played", "team1_id",
                             "team2_id", "team1_score", "team2_score"])
    g = g[g.played.astype(str).str.lower() == "true"]
    g = g.dropna(subset=["map_name", "team1_id", "team2_id", "team1_score", "team2_score"])
    g = g[g.team1_score != g.team2_score].copy()
    g["team1_id"] = g.team1_id.astype(int)
    g["team2_id"] = g.team2_id.astype(int)
    g = g.sort_values(["match_id", "order"])
    by_match = {mid: grp for mid, grp in g.groupby("match_id")}

    O, D = {}, {}                 # O[team] overall map rating; D[(team, map)] map adjustment
    recent = []                   # (datetime, map) of recently played maps
    window = pd.Timedelta(days=window_days)
    adj, full = [], []
    for r in df.itertuples():
        recent = [(t, m) for t, m in recent if r.datetime - t <= window]
        a, b = r.team1_id, r.team2_id
        if len(recent) >= min_pool:
            counts = {}
            for _, m in recent:
                counts[m] = counts.get(m, 0) + 1
            tot = sum(counts.values())
            adj_v = sum(c * (D.get((a, m), 0.0) - D.get((b, m), 0.0))
                        for m, c in counts.items()) / tot
            adj.append(adj_v)
            full.append(O.get(a, 1500) - O.get(b, 1500) + adj_v)
        else:
            adj.append(0.0)
            full.append(0.0)

        grp = by_match.get(r.match_id)          # update AFTER the prediction
        if grp is None:
            continue
        for row in grp.itertuples():
            t1, t2, m = row.team1_id, row.team2_id, row.map_name
            r1 = O.get(t1, 1500) + D.get((t1, m), 0.0)
            r2 = O.get(t2, 1500) + D.get((t2, m), 0.0)
            p = win_prob(r1, r2)
            res = 1 if row.team1_score > row.team2_score else 0
            O[t1] = O.get(t1, 1500) + k_overall * (res - p)
            O[t2] = O.get(t2, 1500) - k_overall * (res - p)
            D[(t1, m)] = D.get((t1, m), 0.0) + k_map * (res - p)
            D[(t2, m)] = D.get((t2, m), 0.0) - k_map * (res - p)
            recent.append((r.datetime, m))
    out = df.copy()
    out["map_adj_diff"] = adj
    out["map_elo_diff"] = full
    return out


def build_features(df):
    """Runs every feature function. New features get added here."""
    df, ratings = add_elo(df)
    df = add_player_rating(df)
    df = add_map_elo(df)
    return df, ratings