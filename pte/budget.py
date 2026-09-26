"""Spend tracking with a hard cap, so an experiment can never surprise your wallet."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_usd: float = 5.0
    max_calls: int = 10_000
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def check(self) -> None:
        """Raise before a new call if the cap has been reached."""
        with self._lock:
            if self.spent_usd >= self.max_usd:
                raise BudgetExceeded(f"budget of ${self.max_usd:.2f} reached (spent ${self.spent_usd:.4f})")
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"call cap of {self.max_calls} reached")

    def record(self, input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
        cost = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.spent_usd += cost
        return cost

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "spent_usd": round(self.spent_usd, 6),
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }

    def restore(self, snap: dict) -> None:
        with self._lock:
            self.spent_usd = float(snap.get("spent_usd", 0.0))
            self.calls = int(snap.get("calls", 0))
            self.input_tokens = int(snap.get("input_tokens", 0))
            self.output_tokens = int(snap.get("output_tokens", 0))
