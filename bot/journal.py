"""Log set-up, the append-only trade and equity records, and the saved strategy state.

Each account's files live under runs/<account>/:
  logs/bot.log            what the bot did and why (rotated)
  logs/api.log            every API request and response, without keys (rotated)
  journal/orders.csv      every order: status, fill, fee and the reason behind it
  journal/equity.csv      portfolio value and positions after every hourly cycle
  journal/decisions.jsonl signals, scores and target weights behind every cycle
  state.json              what the strategy remembers between restarts
"""
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from bot.strategy import StrategyState

log = logging.getLogger(__name__)

ORDER_FIELDS = ["time", "pair", "side", "type", "order_id", "status", "role", "quantity",
                "limit_price", "filled", "avg_price", "fee", "fee_coin", "reason",
                "current_weight", "target_weight", "error", "collateral", "realized_pnl"]
EQUITY_FIELDS = ["time", "equity", "cash", "invested", "drawdown", "peak_equity", "risk_on",
                 "brake_on", "positions"]


def utc_iso(ts_ms: Optional[int] = None) -> str:
    ts = time.time() if ts_ms is None else ts_ms / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s: %(message)s",
                                  "%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime

    def rotating(name: str, megabytes: int) -> logging.Handler:
        handler = RotatingFileHandler(os.path.join(log_dir, name), maxBytes=megabytes * 1024 * 1024,
                                      backupCount=10, encoding="utf-8")
        handler.setFormatter(formatter)
        return handler

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(rotating("bot.log", 10))

    api = logging.getLogger("roostoo.api")
    api.setLevel(logging.INFO)
    api.propagate = False
    for handler in list(api.handlers):
        api.removeHandler(handler)
    api.addHandler(rotating("api.log", 20))


class Journal:
    """Append-only CSV and JSON-lines records, flushed to disk after every row."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def order(self, row: Dict[str, Any]) -> None:
        self._append_csv("orders.csv", ORDER_FIELDS, dict(row, time=row.get("time") or utc_iso()))

    def equity(self, row: Dict[str, Any]) -> None:
        self._append_csv("equity.csv", EQUITY_FIELDS, dict(row, time=row.get("time") or utc_iso()))

    def decision(self, record: Dict[str, Any]) -> None:
        self._append_line("decisions.jsonl", json.dumps(record, sort_keys=True))

    def peak_equity(self) -> float:
        """Highest equity ever recorded, so the drawdown brake survives a lost state file."""
        path = os.path.join(self.directory, "equity.csv")
        if not os.path.exists(path):
            return 0.0
        with open(path, newline="") as f:
            values = [float(r["equity"]) for r in csv.DictReader(f) if r.get("equity")]
        return max(values) if values else 0.0

    def _append_csv(self, name: str, fields: List[str], row: Dict[str, Any]) -> None:
        path = os.path.join(self.directory, name)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def _append_line(self, name: str, line: str) -> None:
        with open(os.path.join(self.directory, name), "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


class StateStore:
    """Strategy state as JSON, replaced atomically so a crash never leaves half a file."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> Optional[StrategyState]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path) as f:
                return StrategyState.from_dict(json.load(f))
        except (ValueError, KeyError, TypeError) as exc:
            log.error("ignoring unreadable state file %s: %s", self.path, exc)
            return None

    def save(self, state: StrategyState) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
