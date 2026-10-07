"""live.py: bracket simulation that locks in the results you type into data/results.csv.

Run after final.py (it reads data/pair_probs_final.csv). Usage:  python live.py
Outputs: data/live_odds.csv and data/live_bracket.html (open the HTML in your browser).
"""
from pathlib import Path

import numpy as np
import pandas as pd

N = 20_000
RESULTS = Path("data/results.csv")
TEAMS = ["100 Thieves", "G2 Esports", "Team Vitality", "Nongshim RedForce",
         "NRG", "T1", "Paper Rex", "LOUD"]
ALIASES = {
    "100 thieves": "100 Thieves", "100t": "100 Thieves",
    "g2": "G2 Esports", "g2 esports": "G2 Esports",
    "vitality": "Team Vitality", "vit": "Team Vitality", "team vitality": "Team Vitality",
    "nongshim": "Nongshim RedForce", "ns": "Nongshim RedForce",
    "nongshim redforce": "Nongshim RedForce",
    "nrg": "NRG", "t1": "T1", "prx": "Paper Rex", "paper rex": "Paper Rex", "loud": "LOUD",
}

# match code -> date (in the order the matches are played)
SCHEDULE = {
    "UQF3": "Oct 7, 5 pm PST", "UQF4": "Oct 7, 8 pm PST",
    "UQF1": "Oct 8, 5 pm PST", "UQF2": "Oct 8, 8 pm PST",
    "LR1A": "Oct 9, 5 pm PST", "LR1B": "Oct 9, 8 pm PST",
    "USF1": "Oct 10, 5 pm PST", "USF2": "Oct 10, 8 pm PST",
    "LR2A": "Oct 11, 5 pm PST", "LR2B": "Oct 11, 8 pm PST",
    "UF": "Oct 16, 2 pm PST", "LR3": "Oct 16, 5 pm PST",
    "LF": "Oct 17, 3 pm PST, Bo5", "GF": "Oct 18, 2 pm PST, Bo5",
}
# matches that must be finished before a match's teams are certain
FEEDERS = {
    "UQF1": [], "UQF2": [], "UQF3": [], "UQF4": [],
    "USF1": ["UQF1", "UQF2"], "USF2": ["UQF3", "UQF4"], "UF": ["USF1", "USF2"],
    "LR1A": ["UQF1", "UQF2"], "LR1B": ["UQF3", "UQF4"],
    "LR2A": ["LR1A", "USF2"], "LR2B": ["LR1B", "USF1"],
    "LR3": ["LR2A", "LR2B"], "LF": ["UF", "LR3"], "GF": ["UF", "LF"],
}


# ---------- 1. results you typed ----------
def load_results():
    if not RESULTS.exists():
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text("match,winner\n", encoding="utf-8")
        print(f"Created {RESULTS} (empty). Add one line per finished match, e.g.  UQF3,NRG\n")
    df = pd.read_csv(RESULTS, dtype=str).dropna(how="all")
    known = {}
    for code_raw, win_raw in zip(df["match"], df["winner"]):
        code = str(code_raw).strip().upper()
        winner = ALIASES.get(str(win_raw).strip().lower())
        if code not in SCHEDULE:
            raise SystemExit(f"Unknown match code '{code_raw}' in {RESULTS}. "
                             f"Valid codes: {', '.join(SCHEDULE)}")
        if winner is None:
            raise SystemExit(f"Unknown team '{win_raw}' in {RESULTS}. Use: {', '.join(TEAMS)}")
        known[code] = winner
    return known


# ---------- 2. win chances from final.py ----------
pp = pd.read_csv("data/pair_probs_final.csv")
P = {"Elo only": {}, "Elo + player": {}, "Blend": {}}
for r in pp.itertuples():
    key = (r.team_a, r.team_b)
    P["Elo only"][key] = r.p_elo_logistic
    P["Elo + player"][key] = r.p_elo_player
    P["Blend"][key] = r.p_blend


# ---------- 3. the bracket (one function, played two ways) ----------
def bracket(play):
    m = {}

    def g(code, a, b):
        w, l = play(code, a, b)
        m[code] = (a, b, w, l)
        return w, l

    w1, l1 = g("UQF1", "100 Thieves", "G2 Esports")
    w2, l2 = g("UQF2", "Team Vitality", "Nongshim RedForce")
    w3, l3 = g("UQF3", "NRG", "T1")
    w4, l4 = g("UQF4", "Paper Rex", "LOUD")
    ws1, ls1 = g("USF1", w1, w2)
    ws2, ls2 = g("USF2", w3, w4)
    wu, lu = g("UF", ws1, ws2)
    wl1, _x = g("LR1A", l1, l2)
    wl2, _x = g("LR1B", l3, l4)
    wa, _x = g("LR2A", ls2, wl1)       # assumed crossover: USF2 loser meets LR1A winner
    wb, _x = g("LR2B", ls1, wl2)       # USF1 loser meets LR1B winner
    wr3, _x = g("LR3", wa, wb)
    wlf, _x = g("LF", lu, wr3)
    g("GF", wu, wlf)
    return m


def apply_known(code, a, b, known):
    w = known[code]
    if w not in (a, b):
        raise SystemExit(f"Result '{code},{w}' is impossible: {code} is {a} vs {b}. "
                         "Check data/results.csv, or enter the earlier rounds first.")
    return (a, b) if w == a else (b, a)


def simulate(p, known, seed=42):
    rng = np.random.default_rng(seed)

    def play(code, a, b):
        if code in known:
            return apply_known(code, a, b, known)
        return (a, b) if rng.random() < p[(a, b)] else (b, a)

    titles, finals = dict.fromkeys(TEAMS, 0), dict.fromkeys(TEAMS, 0)
    for _i in range(N):
        a, b, champ, _l = bracket(play)["GF"]
        titles[champ] += 1
        finals[a] += 1
        finals[b] += 1
    return {t: titles[t] / N for t in TEAMS}, {t: finals[t] / N for t in TEAMS}


def chalk(p, known):
    """The single most likely path: the favorite wins every match not yet played."""
    def play(code, a, b):
        if code in known:
            return apply_known(code, a, b, known)
        return (a, b) if p[(a, b)] >= 0.5 else (b, a)
    return bracket(play)


# ---------- 4. run ----------
known = load_results()
print(f"Results locked in: {len(known)}")
for code in SCHEDULE:
    if code in known:
        print(f"   {code}: {known[code]}")

cm = chalk(P["Blend"], known)
print("\nMatch guide (blend win chances; * = teams are predicted, not confirmed yet):")
for code in SCHEDULE:
    a, b, w, l = cm[code]
    star = "" if all(f in known for f in FEEDERS[code]) else "*"
    pa = P["Blend"][(a, b)]
    note = f"DONE, {w} won" if code in known else f"{a} {pa:.0%} | {b} {1 - pa:.0%}"
    print(f"  {code:<5}{SCHEDULE[code]:<24}{a} vs {b}{star}  ->  {note}")

sim = {name: simulate(p, known) for name, p in P.items()}
out = pd.DataFrame({
    "team": TEAMS,
    "title_elo": [sim["Elo only"][0][t] for t in TEAMS],
    "title_player": [sim["Elo + player"][0][t] for t in TEAMS],
    "title_blend": [sim["Blend"][0][t] for t in TEAMS],
    "final_blend": [sim["Blend"][1][t] for t in TEAMS],
}).sort_values("title_blend", ascending=False).reset_index(drop=True)
out.to_csv("data/live_odds.csv", index=False)

pct = lambda s: (s * 100).round(1).astype(str) + "%"
show = pd.DataFrame({"team": out.team, "Elo only": pct(out.title_elo),
                     "Elo+player": pct(out.title_player), "Blend": pct(out.title_blend),
                     "Grand Final (blend)": pct(out.final_blend)})
print(f"\n=== Title odds ({N:,} simulated brackets per model, played results locked in) ===\n")
print(show.to_string(index=False))
print(f"\nMost likely single path (favorite wins every unplayed match): champion = {cm['GF'][2]}")

# ---------- 5. bracket page ----------
COLW, CARDW, OFF = 300, 240, 36
POS = {  # code: (column, top)
    "UQF1": (0, 0), "UQF2": (0, 92), "UQF3": (0, 210), "UQF4": (0, 302),
    "USF1": (1, 38), "USF2": (1, 248), "UF": (2, 135), "GF": (3, 135),
    "LR1A": (0, 440), "LR1B": (0, 570), "LR2A": (1, 417), "LR2B": (1, 547),
    "LR3": (2, 474), "LF": (3, 451),
}
SLOT_TAG = {("LR1A", 0): "loser UQF1", ("LR1A", 1): "loser UQF2",
            ("LR1B", 0): "loser UQF3", ("LR1B", 1): "loser UQF4",
            ("LR2A", 0): "loser USF2", ("LR2B", 0): "loser USF1",
            ("LF", 0): "loser UF", ("GF", 1): "winner LF"}
EDGES = [("UQF1", "USF1", 0), ("UQF2", "USF1", 1), ("UQF3", "USF2", 0), ("UQF4", "USF2", 1),
         ("USF1", "UF", 0), ("USF2", "UF", 1), ("UF", "GF", 0),
         ("LR1A", "LR2A", 1), ("LR1B", "LR2B", 1), ("LR2A", "LR3", 0),
         ("LR2B", "LR3", 1), ("LR3", "LF", 1)]

CSS = """
:root{--ink:#0f1923;--paper:#eef1f4;--card:#fbfcfd;--line:#c3cad2;--red:#ff4655;--mute:#6b7785;--done:#0f1923}
*{box-sizing:border-box}
body{margin:24px;background:var(--paper);color:var(--ink);font-family:Bahnschrift,"Segoe UI",Arial,sans-serif}
h1{font-size:26px;font-weight:600;margin:0 0 6px;letter-spacing:.2px}
h2{font-size:18px;font-weight:600;margin:28px 0 8px}
.sub{max-width:760px;color:var(--mute);font-size:14px;line-height:1.5;margin-bottom:18px}
.wrap{overflow-x:auto;padding-bottom:8px}
.board{position:relative;width:1230px;height:710px}
svg{position:absolute;left:0;top:0}
path{fill:none;stroke:var(--line);stroke-width:1.5}
.lab{position:absolute;left:10px;font-size:13px;font-weight:600;color:var(--mute)}
.card{position:absolute;width:240px;background:var(--card);border:1px solid var(--line);border-radius:2px}
.bar{height:16px;display:flex;justify-content:space-between;padding:2px 8px;font-size:11px;color:var(--mute);border-bottom:1px solid #e4e8ec}
.row{height:30px;display:flex;align-items:center;gap:6px;padding:0 8px;font-size:13.5px;border-left:3px solid transparent}
.row+.row{border-top:1px solid #e4e8ec}
.row .t{flex:1;white-space:nowrap}
.row .tag{font-size:10.5px;color:var(--mute);white-space:nowrap}
.row .p{min-width:34px;text-align:right;font-variant-numeric:tabular-nums;color:var(--mute)}
.pick{border-left-color:var(--red);font-weight:600}
.pick .p{color:var(--ink)}
.win{background:var(--done);color:#fff;font-weight:600}
.win .p,.win .tag{color:#fff}
.lose{color:#98a2ad}
.guess .t{font-style:italic}
table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:6px 16px 6px 0;text-align:left;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--mute)}
td.b{font-weight:600}
"""


def card(code):
    a, b, w, l = cm[code]
    pa = P["Blend"][(a, b)]
    done = code in known
    guess = not all(f in known for f in FEEDERS[code])
    rows = ""
    for i, t in enumerate((a, b)):
        p = pa if i == 0 else 1 - pa
        cls = "row"
        if done:
            cls += " win" if t == w else " lose"
        elif t == w:
            cls += " pick"
        if guess:
            cls += " guess"
        shown = ("&#10003;" if t == w else "") if done else f"{p:.0%}"
        rows += (f'<div class="{cls}"><span class="t">{t}</span>'
                 f'<span class="tag">{SLOT_TAG.get((code, i), "")}</span>'
                 f'<span class="p">{shown}</span></div>')
    x, y = 10 + POS[code][0] * COLW, POS[code][1] + OFF
    return (f'<div class="card" style="left:{x}px;top:{y}px">'
            f'<div class="bar"><span>{code}</span><span>{SCHEDULE[code]}</span></div>{rows}</div>')


def edge(src, dst, slot):
    x1 = 10 + POS[src][0] * COLW + CARDW + 2
    y1 = POS[src][1] + OFF + 39
    x2 = 10 + POS[dst][0] * COLW
    y2 = POS[dst][1] + OFF + (32 if slot == 0 else 62)
    return f'<path d="M{x1} {y1} H{(x1 + x2) / 2} V{y2} H{x2}"/>'


table = "<table><tr><th>Team</th><th>Blend</th><th>Grand Final (blend)</th></tr>"
for r in out.itertuples():
    table += (f"<tr><td>{r.team}</td>"
              f"<td class='b'>{r.title_blend:.1%}</td><td>{r.final_blend:.1%}</td></tr>")
table += "</table>"
# <td>{r.title_elo:.1%}</td><td>{r.title_player:.1%}</td> <th>Elo only</th><th>Elo + player</th>
html = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Valorant Champions 2026 playoffs: predicted bracket</title>"
    f"<style>{CSS}</style></head><body>"
    "<h1>Valorant Champions 2026 playoffs: predicted bracket</h1>"
    "<div class='sub'>Each match shows the blend model's single-series win chances. "
    "A red edge marks the team the model sends through; dark rows are played results; "
    "italic names are teams that depend on earlier predictions. "
    f"On the most likely path the champion is <b>{cm['GF'][2]}</b>, but one path is only a guide. "
    "The odds table below averages 20,000 simulated brackets and is the better summary.</div>"
    "<div class='wrap'><div class='board'>"
    f"<svg width='1230' height='710'>{''.join(edge(*e) for e in EDGES)}</svg>"
    "<div class='lab' style='top:6px'>Upper bracket</div>"
    "<div class='lab' style='top:430px'>Lower bracket</div>"
    f"{''.join(card(c) for c in SCHEDULE)}"
    "</div></div>"
    "<h2>Title odds with played results locked in</h2>"
    f"{table}</body></html>"
)
Path("data/live_bracket.html").write_text(html, encoding="utf-8")
print("\nSaved data/live_odds.csv and data/live_bracket.html (open the HTML in your browser)")