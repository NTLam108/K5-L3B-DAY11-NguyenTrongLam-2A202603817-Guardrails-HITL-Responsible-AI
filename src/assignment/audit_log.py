"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.

Each finished request becomes one entry:

    {
      "request_id": "req-0001",
      "user_id": "alice",
      "input": "...",
      "output": "...",
      "blocked": true,
      "layer": "input_guardrail",
      "issues": [...],
      "started_at": "2026-...Z",
      "finished_at": "2026-...Z",
      "latency_ms": 12.3
    }
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # Inputs waiting for their output, keyed by request_id (or user_id).
        # A FIFO per key so two in-flight requests from the same user without
        # a request_id are matched in order instead of overwriting each other.
        self._open: dict[str, deque[dict]] = defaultdict(deque)
        self._seq = 0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def _next_request_id(self) -> str:
        self._seq += 1
        return f"req-{self._seq:04d}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store the input and its start time. Returns the request_id used."""
        rid = request_id or self._next_request_id()
        # Key by the explicit request_id when given; otherwise by user_id, so a
        # later record_output(user_id=...) without an id still finds it.
        key = request_id or user_id
        self._open[key].append(
            {
                "request_id": rid,
                "user_id": user_id,
                "input": text,
                "started_at": utc_now_iso(),
                "_t0": time.perf_counter(),
            }
        )
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        issues: list[str] | None = None,
    ) -> dict:
        """Close the matching input, compute latency, append one entry to ``self.logs``."""
        key = request_id or user_id
        queue = self._open.get(key)
        pending = queue.popleft() if queue else None
        if queue is not None and not queue:
            self._open.pop(key, None)

        if pending is None:
            # Output without a recorded input: still log it — a gap in the
            # trail is itself something an investigator needs to see.
            pending = {
                "request_id": request_id or self._next_request_id(),
                "user_id": user_id,
                "input": None,
                "started_at": None,
                "_t0": None,
            }

        t0 = pending.pop("_t0")
        entry = {
            **pending,
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "issues": list(issues or []),
            "finished_at": utc_now_iso(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2) if t0 is not None else None,
        }
        self.logs.append(entry)
        return entry

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def pending(self) -> list[dict]:
        """Inputs that never got an output (crash / timeout)."""
        out = []
        for queue in self._open.values():
            for p in queue:
                out.append({k: v for k, v in p.items() if k != "_t0"} | {"status": "no_output"})
        return out

    def summary(self) -> dict:
        total = len(self.logs)
        blocked = sum(1 for e in self.logs if e["blocked"])
        by_layer: dict[str, int] = {}
        for e in self.logs:
            if e["blocked"]:
                name = e["layer"] or "unknown"
                by_layer[name] = by_layer.get(name, 0) + 1
        latencies = [e["latency_ms"] for e in self.logs if e["latency_ms"] is not None]
        return {
            "total": total,
            "blocked": blocked,
            "passed": total - blocked,
            "blocked_by_layer": by_layer,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        }

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default.

        Requests still waiting for an output are appended with
        ``"status": "no_output"`` so nothing silently disappears.
        Returns the path written.
        """
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.logs + self.pending()
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
