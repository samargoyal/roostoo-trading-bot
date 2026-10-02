"""H1: which signals predict the next 24 hours?

For every candidate feature, once a day at 00:00 UTC, the Spearman rank correlation (IC)
between the feature and the next 24-hour return across that month's universe. A useful
signal has a mean IC reliably above zero in both development years. Also reports the
pooled IC, which mixes timing (all coins together) with selection.

    python -m research.h1_signals
"""
import numpy as np
import pandas as pd

from bot.config import UniverseConfig
from research.panel import cached_pairs, features, load_panel, monthly_universe, targets

PERIODS = {
    "Y0 Oct24-Sep25": ("2024-10-01", "2025-10-01"),
    "Y1 Oct25-Jun26": ("2025-10-01", "2026-07-01"),
}
HOLD_OUT_START = "2026-07-01"  # never looked at during development


def daily_ic(feature: pd.DataFrame, target: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    rows = feature.index[feature.index.hour == 0]
    f = feature.loc[rows].where(mask.loc[rows])
    t = target.loc[rows].where(mask.loc[rows])
    fr = f.rank(axis=1)
    tr = t.rank(axis=1)
    valid = fr.notna() & tr.notna()
    fr, tr = fr.where(valid), tr.where(valid)
    fd = fr.sub(fr.mean(axis=1), axis=0)
    td = tr.sub(tr.mean(axis=1), axis=0)
    ic = (fd * td).sum(axis=1) / np.sqrt((fd ** 2).sum(axis=1) * (td ** 2).sum(axis=1))
    return ic[valid.sum(axis=1) >= 8]


def main() -> None:
    pairs = cached_pairs()
    panel = load_panel(pairs)
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    feats = features(panel)
    fwd = targets(panel)["fwd"]
    rows = []
    for name, f in feats.items():
        if name.startswith("btc_") or name == "breadth":
            continue  # identical across pairs: no cross-sectional information
        row = {"feature": name}
        for label, (start, end) in PERIODS.items():
            ic = daily_ic(f, fwd, mask).loc[start:end]
            ic = ic[ic.index < HOLD_OUT_START]
            row[label + " IC"] = ic.mean()
            row[label + " t"] = ic.mean() / ic.std() * np.sqrt(len(ic))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("feature")
    table["both years same sign"] = np.sign(table.iloc[:, 0]) == np.sign(table.iloc[:, 2])
    pd.set_option("display.width", 160)
    print("Cross-sectional rank IC with the next 24h return (daily, point-in-time top 20 + PAXG)")
    print(table.round(3).sort_values(table.columns[0], ascending=False).to_string())

    # Timing: does market-wide context predict the average coin's next 24 hours?
    print("\nTiming: correlation of market features with the universe's mean next-24h return")
    mean_fwd = fwd.where(mask).mean(axis=1)
    for name in ("btc_mom_24", "btc_mom_168", "btc_trend", "breadth"):
        x = feats[name]["BTC/USD"].astype(float)  # market-wide: the same for every pair
        out = []
        for label, (start, end) in PERIODS.items():
            sel = (x.index >= start) & (x.index < min(end, HOLD_OUT_START)) & (x.index.hour == 0)
            both = pd.concat([x[sel], mean_fwd[sel]], axis=1).dropna()
            out.append("%s %.3f" % (label, both.corr().iloc[0, 1]))
        print("  %-12s %s" % (name, "   ".join(out)))


if __name__ == "__main__":
    main()
