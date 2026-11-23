"""Read-only view of the evolving attacker, trial by trial. Safe to run while the run is going.

python3 diagnose_evolving.py RUN_DIR                  # one line per pair: T/F per trial, flips marked
python3 diagnose_evolving.py RUN_DIR --pair PAIR_ID   # every trial of one pair: source, strategy, score, text, reply
python3 diagnose_evolving.py RUN_DIR --pair PAIR_ID --full   # same, without cutting texts short (adds victim trace)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

argv = sys.argv[1:]
full = "--full" in argv
pair_id = argv[argv.index("--pair") + 1] if "--pair" in argv else None
run = Path(next(a for a in argv if not a.startswith("--") and a != pair_id))


def cut(s, n):
    s = "" if s is None else str(s)
    return s if full or len(s) <= n else s[:n] + " [...]"


trials = defaultdict(dict)
for line in (run / "trials.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue  # line still being written
    if r.get("pair"):
        trials[r["pair"]][r["trial"]] = r
archive = defaultdict(dict)
state_path = run / "state.json"
if state_path.exists():
    for e in json.loads(state_path.read_text()).get("attacker", {}).get("archive", []):
        archive[e["pair"]][e["trial"]] = e

LETTER = {"generated": "n", "kept_success": "k", "goal_fallback": "f"}
SRC = {"generated": "new", "kept_success": "kept", "goal_fallback": "fallback"}

if pair_id is None:
    print(
        "T = attack worked, F = blocked. Second column: 0 = original text, n = new text, k = kept the last winning text, f = fallback."
    )
    print("FLIP = a kept winning text failed on a later trial (same text, different victim answer).\n")
    for p in sorted(trials):
        ts = sorted(trials[p])
        sec = "".join("T" if trials[p][t].get("security") else "F" for t in ts)
        src = "".join(LETTER.get(archive[p].get(t, {}).get("source"), "?") if t else "0" for t in ts)
        flips = [
            t
            for t in ts
            if t and archive[p].get(t, {}).get("source") == "kept_success" and not trials[p][t].get("security")
        ]
        print(f"{p:60s} {sec:10s} {src:10s} {'FLIP at trial ' + ','.join(map(str, flips)) if flips else ''}")
    sys.exit()

if pair_id not in trials:
    sys.exit(f"no trials for {pair_id!r}; run without --pair to list the pair ids")
prev_text = None
for t in sorted(trials[pair_id]):
    r, e = trials[pair_id][t], archive[pair_id].get(t, {})
    same = " (SAME text as previous trial)" if r.get("injection_text") == prev_text else ""
    prev_text = r.get("injection_text")
    print(f"\n========== trial {t} | source: {SRC.get(e.get('source'), e.get('source', 'trial 0 / original'))}{same}")
    print(
        f"security (attack worked): {r.get('security')} | user task done: {r.get('utility')} | "
        f"progress score 0-4: {r.get('progress')}"
    )
    if e.get("strategy"):
        print(f"strategy: {e['strategy']}   ranking: {', '.join(e.get('strategy_ranking') or [])}")
    if e.get("diagnosis"):
        print(f"attacker's diagnosis: {cut(e['diagnosis'], 400)}")
    print(f"--- injection text:\n{cut(r.get('injection_text'), 700)}")
    print(f"--- victim's final answer:\n{cut(r.get('final_answer'), 500)}")
    if full and r.get("victim_trace"):
        print(f"--- victim trace (shown to the attacker, S0 redacted):\n{r['victim_trace']}")
