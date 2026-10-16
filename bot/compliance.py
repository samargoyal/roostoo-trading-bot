"""Compliance report: the account's own records checked against the strategy and the repository.

    python -m bot.compliance --account comp                # from runs/comp/ and this git checkout
    python -m bot.compliance --account comp --exchange     # also against Roostoo's order history (keys)

The competition asks that the trade logs match the declared strategy, that every change is in the
commit history and that no trade is made by hand. This reads what the live bot wrote (logs/bot.log*,
journal/decisions.jsonl, journal/orders.csv, journal/equity.csv) and checks, writing the result to
runs/<account>/compliance.md:

  1. every start of the bot ran a committed config: the commit in its start line exists, had no
     local changes, and its config/<account>.json (or the --config file) holds the settings the bot
     logged;
  2. every order came from a decision: the latest cycle decision before it set that pair's target
     weight to the order's target, and the order moves the holding towards it (the activity rule's
     small orders and funding sales are accounted for separately);
  3. every order's reason is one the strategy or planner gives;
  4. the activity rule: the UTC days with at least one filled order (the competition asks for 8 of 14);
  5. with --exchange: every order in Roostoo's own history is one the bot recorded (an order the bot
     did not record would be a manual trade);
  6. the account's figures: return, drawdown and the competition's score from equity.csv.
Read-only: it sends no orders. With --exchange it makes a few signed queries.
"""
import argparse
import bisect
import csv
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bot.metrics import summarize
from bot.planner import ACTIVITY, BUY, COVER, EXIT_UNIVERSE, FUNDING, REBALANCE, SELL, SHORT
from bot import reasons as R

KNOWN_REASONS = {getattr(R, n) for n in dir(R) if n.isupper() and isinstance(getattr(R, n), str)} | {
    ACTIVITY, REBALANCE, FUNDING, EXIT_UNIVERSE}
START = re.compile(r"^(\S+)Z INFO bot\.live: starting: account=(\S+) dry_run=(\S+) once=\S+ commit=(.+?) config=(.+)$")
CONFIG = re.compile(r"^(\S+)Z INFO bot\.live: config: (\{.*\})$")
TOLERANCE = 1e-3


def _time(text: str) -> datetime:
    text = text.rstrip("Z").split(".")[0]
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def read_starts(log_dir: str) -> List[Dict[str, Any]]:
    """Every start of the bot in the (rotated) logs, oldest first, with the config it logged."""
    starts: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(log_dir, "bot.log*")):          # each file in order; sorted by time below
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = START.match(line.rstrip("\n"))
                if m:
                    commit = m.group(4)
                    starts.append({"time": _time(m.group(1)), "account": m.group(2), "dry_run": m.group(3) == "True",
                                   "commit": commit.split()[0], "dirty": "local changes" in commit,
                                   "config_path": m.group(5), "config": None})
                    continue
                m = CONFIG.match(line.rstrip("\n"))
                if m and starts and starts[-1]["config"] is None:
                    starts[-1]["config"] = json.loads(m.group(2))
    starts.sort(key=lambda s: s["time"])
    return starts


def committed_file(repo: str, commit: str, path: str) -> Optional[Dict[str, Any]]:
    """The JSON file at `path` as of `commit`, or None if it is not in that commit."""
    if path in ("code defaults", ""):
        return {}
    try:
        out = subprocess.run(["git", "show", "%s:%s" % (commit, path.replace("\\", "/"))], cwd=repo,
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def overrides_match(logged: Dict[str, Any], overrides: Dict[str, Any], prefix: str = "") -> List[str]:
    """Settings in the committed file that differ from what the bot logged."""
    diffs = []
    for key, value in overrides.items():
        have = logged.get(key) if isinstance(logged, dict) else None
        if isinstance(value, dict):
            diffs += overrides_match(have or {}, value, prefix + key + ".")
        elif have != value and not (isinstance(value, (int, float)) and isinstance(have, (int, float))
                                     and abs(have - value) < 1e-12):
            diffs.append("%s%s: committed %r, logged %r" % (prefix, key, value, have))
    return diffs


def check_starts(starts: List[Dict[str, Any]], repo: str) -> List[Dict[str, Any]]:
    rows = []
    for s in starts:
        problems = []
        if s["dirty"]:
            problems.append("ran with uncommitted changes")
        if s["commit"] == "unknown":
            problems.append("commit unknown")
        committed = committed_file(repo, s["commit"], s["config_path"]) if s["commit"] != "unknown" else None
        if committed is None:
            problems.append("%s is not in commit %s" % (s["config_path"], s["commit"]))
        elif s["config"] is None:
            problems.append("no config line logged")
        else:
            problems += overrides_match(s["config"], committed)
        rows.append(dict(s, ok=not problems, problems=problems))
    return rows


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_decisions(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["_time"] = _time(d["time"])
                out.append(d)
    out.sort(key=lambda d: d["_time"])
    return out


def check_orders(orders: List[Dict[str, str]], decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Each order against the latest decision made before it."""
    times = [d["_time"] for d in decisions]
    unmatched, wrong_side, unknown_reason, matched, activity = [], [], [], 0, 0
    for o in orders:
        if o.get("status") == "ERROR" and not float(o.get("filled") or 0):
            continue
        reason = o.get("reason", "")
        if reason not in KNOWN_REASONS:
            unknown_reason.append(o)
        if reason == ACTIVITY:
            activity += 1
            continue
        t = _time(o["time"])
        i = bisect.bisect_right(times, t) - 1
        d = decisions[i] if i >= 0 else None
        if d is None or (t - d["_time"]).total_seconds() > 2 * 3600:
            unmatched.append(o)
            continue
        target = float(o.get("target_weight") or 0)
        current = float(o.get("current_weight") or 0)
        decided = float(d.get("targets", {}).get(o["pair"], 0.0))
        if reason != FUNDING and abs(decided - target) > TOLERANCE:
            unmatched.append(dict(o, decided=decided))
            continue
        side = o.get("side") or ""
        up = side in (BUY, COVER) or o.get("type") == "SHORT_CLOSE"
        down = side in (SELL, SHORT) or o.get("type") == "SHORT_OPEN"
        if (up and target < current - TOLERANCE) or (down and target > current + TOLERANCE):
            wrong_side.append(o)
            continue
        matched += 1
    return {"orders": len(orders), "matched": matched, "activity": activity, "unmatched": unmatched,
            "wrong_side": wrong_side, "unknown_reason": unknown_reason}


def active_days(orders: List[Dict[str, str]], equity: List[Dict[str, str]]) -> Dict[str, Any]:
    days = sorted({e["time"][:10] for e in equity})
    traded = {o["time"][:10] for o in orders if float(o.get("filled") or 0) > 0}
    return {"days": days, "traded": sorted(traded & set(days)) if days else sorted(traded),
            "missed": [d for d in days if d not in traded]}


def exchange_orders(client) -> List[Dict[str, Any]]:
    """Every order in Roostoo's history for the account, oldest page last."""
    out, offset = [], 0
    while True:
        page = client.query_order(pending_only=False, offset=offset, limit=100)
        if not page:
            break
        out += page
        if len(page) < 100:
            break
        offset += len(page)
    return out


def check_exchange(recorded: List[Dict[str, str]], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Orders in the exchange's history that the bot did not record (none means no manual trades)."""
    ids = {str(o.get("order_id")) for o in recorded if o.get("order_id")}
    return [h for h in history if str(h.get("OrderID")) not in ids]


def performance(equity: List[Dict[str, str]]) -> Dict[str, float]:
    curve = []
    for e in equity:
        if e.get("equity"):
            curve.append((int(_time(e["time"]).timestamp() * 1000), float(e["equity"])))
    if len(curve) < 2:
        return {}
    return summarize(curve, curve[0][1])


def build(run_dir: str, repo: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    orders = read_csv(os.path.join(run_dir, "journal", "orders.csv"))
    equity = read_csv(os.path.join(run_dir, "journal", "equity.csv"))
    report = {
        "starts": check_starts(read_starts(os.path.join(run_dir, "logs")), repo),
        "orders": check_orders(orders, read_decisions(os.path.join(run_dir, "journal", "decisions.jsonl"))),
        "activity": active_days(orders, equity),
        "performance": performance(equity),
        "exchange": None if history is None else check_exchange(orders, history),
        "exchange_count": None if history is None else len(history),
    }
    o = report["orders"]
    report["passed"] = (all(s["ok"] for s in report["starts"]) and not o["unmatched"] and not o["wrong_side"]
                        and not o["unknown_reason"] and (history is None or not report["exchange"]))
    return report


def render(report: Dict[str, Any], account: str) -> str:
    o, a, p = report["orders"], report["activity"], report["performance"]
    mark = lambda ok: "PASS" if ok else "CHECK"
    lines = ["# Compliance report: %s account" % account, "",
             "Made %s by `python -m bot.compliance` from the bot's own records. Overall: **%s**." % (
                 datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), mark(report["passed"])), "",
             "## 1. Every start ran a committed config: %s" % mark(all(s["ok"] for s in report["starts"])), "",
             "| Started (UTC) | Commit | Config | Result |", "|---|---|---|---|"]
    for s in report["starts"]:
        lines.append("| %s | `%s`%s | `%s` | %s |" % (s["time"].strftime("%Y-%m-%d %H:%M"), s["commit"],
                                                    " (dirty)" if s["dirty"] else "", s["config_path"],
                                                    "matches the commit" if s["ok"] else "; ".join(s["problems"])))
    lines += ["", "## 2. Every order came from a strategy decision: %s" % mark(not o["unmatched"] and not o["wrong_side"]),
              "", "%d orders: %d matched to the decision before them, %d placed by the activity rule, %d unmatched, "
              "%d against the decided direction." % (o["orders"], o["matched"], o["activity"], len(o["unmatched"]),
                                                     len(o["wrong_side"]))]
    for x in (o["unmatched"] + o["wrong_side"])[:20]:
        lines.append("- %s %s %s (%s): target %s, current %s%s" % (x["time"], x["side"], x["pair"], x["reason"],
                                                                 x.get("target_weight"), x.get("current_weight"),
                                                                 ", decided %.4f" % x["decided"] if "decided" in x else ""))
    lines += ["", "## 3. Every reason is the strategy's: %s" % mark(not o["unknown_reason"]), "",
              "Reasons seen: %s." % ", ".join(sorted({x.get("reason", "") for x in o["unknown_reason"]})) if o["unknown_reason"]
              else "All order reasons are ones the strategy or planner gives."]
    lines += ["", "## 4. Trading days: %d of %d" % (len(a["traded"]), len(a["days"])), "",
              "Days with at least one filled order: %s." % ", ".join(a["traded"]) if a["traded"] else "None."]
    if a["missed"]:
        lines.append("Days without one: %s." % ", ".join(a["missed"]))
    lines += ["", "## 5. No manual trades"]
    if report["exchange"] is None:
        lines.append("Not checked (run with `--exchange` on the machine with the account's keys).")
    else:
        lines.append("%s: %d orders in Roostoo's history, %d not recorded by the bot." % (
            mark(not report["exchange"]), report["exchange_count"], len(report["exchange"])))
        for h in report["exchange"][:20]:
            lines.append("- order %s %s %s at %s" % (h.get("OrderID"), h.get("Side"), h.get("Pair"), h.get("CreateTimestamp")))
    if p:
        lines += ["", "## 6. The account", "",
                  "| Return | Worst drawdown | Sharpe | Sortino | Calmar | Score (0.4 So + 0.3 Sh + 0.3 Ca) |",
                  "|---|---|---|---|---|---|",
                  "| %+.2f%% | %.2f%% | %.2f | %.2f | %.2f | %.2f |" % (100 * p["total_return"], 100 * p["max_drawdown"],
                                                                 p["sharpe"], p["sortino"], p["calmar"], p["composite"])]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the account's records against the strategy and the repo.")
    parser.add_argument("--account", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--exchange", action="store_true", help="also compare with Roostoo's order history (keys)")
    args = parser.parse_args(argv)
    run_dir = os.path.join(args.runs, args.account)
    history = None
    if args.exchange:
        from bot.roostoo import RoostooClient
        history = exchange_orders(RoostooClient.from_env(max_calls_per_minute=5))
    report = build(run_dir, args.repo, history)
    text = render(report, args.account)
    path = os.path.join(run_dir, "compliance.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("written to", path)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
