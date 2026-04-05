from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeStats:
    processed: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.processed == 0:
            return 0.0
        return self.total_latency_ms / self.processed

    def record_success(self, latency_ms: float) -> None:
        self.processed += 1
        self.total_latency_ms += latency_ms

    def record_failure(self) -> None:
        self.failed += 1
