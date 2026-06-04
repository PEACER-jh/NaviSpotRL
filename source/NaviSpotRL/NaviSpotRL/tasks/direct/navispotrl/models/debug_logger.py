"""File-based debug logger for model diagnostics.

Writes to ``models/logs/train_debug.log`` or ``models/logs/play_debug.log``
depending on NAVISPOTRL_MODE env var (defaults to ``train_debug.log``).
Each launch overwrites the corresponding file.
"""

from __future__ import annotations

import os
import atexit
import datetime


class DebugLogger:
    """Singleton file logger that flushes every write."""

    _instance: DebugLogger | None = None
    _fd = None

    def __new__(cls) -> DebugLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        mode = os.environ.get("NAVISPOTRL_MODE", "train")
        fname = f"{mode}_debug.log"
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, fname)
        self._fd = open(path, "w")  # overwrite
        atexit.register(self.close)
        self._writeln(f"=== Debug log started at {datetime.datetime.now()} ===")

    def log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._writeln(f"[{ts}] {msg}")

    def _writeln(self, line: str) -> None:
        assert self._fd is not None
        self._fd.write(line + "\n")
        self._fd.flush()

    def close(self) -> None:
        if self._fd is not None:
            self._writeln("=== Debug log ended ===")
            self._fd.close()
            self._fd = None


# Global convenience
def dbg(msg: str) -> None:
    DebugLogger().log(msg)
