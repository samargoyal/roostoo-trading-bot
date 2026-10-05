"""H75: short swings only in downtrends, in the rotation's idle money (the user: "long-short
aggressive swings").

h74's nine short swing setups lost in most years, often in bull phases (failed breakouts lost 96%
in 2020-21): in an uptrend dips are bought, and shorts get squeezed. Written down before running:
the same nine, entering only while BTC's 7-day EMA is below its 28-day EMA (the live rotation's
filter off) and covered when it turns back on. Stage A as before. Then, as the aggressive option,
books run with the rotation's idle 70% while the filter is off (in PAXG or cash today), against
the model of the live bot and h73's long-only idle book:

  short    the gated short survivors alone
  both     the long survivors (h61) and the gated short survivors, equally weighted

    python -m research.h75_gated_short_swings
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import research.h74_long_short_swings as h74
from research.h60_swing_mft import costs, ema, stage_a
from research.h61_swing import OUT as H61, REGISTRY as LONGS, Daily, load, run

OUT = os.path.join("runs", "research", "h75")
_ENGINE = h74.engine_short
GATE = {}


def gated_engine(D, entry, **kw):
    """h74's short engine with entries only while BTC's filter is off, covered when it turns on."""
    off, on = GATE["off"], GATE["on"]
    kw["exit_when"] = on if kw.get("exit_when") is None else (kw["exit_when"].fillna(False).astype(bool) | on)
    return _ENGINE(D, entry.fillna(False).astype(bool) & off, **kw)


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    D = load()
    Y = Daily(D)
    cost = costs(list(D.c.columns))
    btc = D.c["BTC/USD"]
    off = (ema(btc, 168) < ema(btc, 672)).values[:, None]
    GATE["off"] = pd.DataFrame(np.repeat(off, D.c.shape[1], axis=1), index=D.idx, columns=D.c.columns)
    GATE["on"] = ~GATE["off"]
    h74.engine_short = gated_engine
    path = os.path.join(OUT, "stage_a.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    print("Stage A, the short setups only while BTC's filter is off (net return by fold):")
    for name, fn, params, neighbours in h74.SHORTS:
        if name not in results:
            by = run(D, Y, cost, fn, params)
            ok, good, have = stage_a(by)
            nb = []
            if ok:
                for p in neighbours:
                    nok, ngood, nhave = stage_a(run(D, Y, cost, fn, p))
                    nb.append("%s %d/%d" % ("holds" if nok else "breaks", ngood, nhave))
            results[name] = {"folds": by, "stage_a": ok, "good": good, "have": have, "neighbours": nb}
            json.dump(results, open(path, "w"))
        r = results[name]
        row = " ".join("%+6.0f%%" % (r["folds"][y]["ret"] * 100) if r["folds"][y] else "    --" for y in sorted(r["folds"]))
        print("  %-42s %s | %s %d/%d%s" % (name, row, "PASS" if r["stage_a"] else "fail", r["good"], r["have"],
                                            " | neighbours: " + ", ".join(r["neighbours"]) if r["neighbours"] else ""),
              flush=True)
    longs = json.load(open(os.path.join(H61, "stage_a.json")))
    long_names = [n for n, *_ in LONGS if longs.get(n, {}).get("stage_a")]
    short_names = [n for n, *_ in h74.SHORTS if results[n]["stage_a"]]
    print("\ngated short setups passing Stage A: %d (%s)" % (len(short_names), ", ".join(n.split(" ")[0] for n in short_names) or "none"))
    L = h74.net_returns(D, Y, cost, LONGS, long_names)
    books = {"long-only swings (h73)": L.mean(axis=1)}
    if short_names:
        S = h74.net_returns(D, Y, cost, h74.SHORTS, short_names)
        books["gated short swings"] = S.mean(axis=1)
        books["long and gated short swings"] = pd.concat([L, S], axis=1).mean(axis=1)
    h74.sleeves(books, D)


if __name__ == "__main__":
    main()
