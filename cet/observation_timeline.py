"""Timestamp-aware North observation buffering for real-time rollouts.

North provides a producer timestamp in each observation bundle. The callback
also records local wall and monotonic receive times. This module maps a
clock-synchronised producer timestamp onto the local monotonic control timeline,
falls back safely when the producer clock/unit is not credible, and suppresses
repeated polls of the same observation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class ObservationStamp:
    """One observation's source and local timing metadata."""

    source_timestamp_raw: Any
    source_wall_time_s: Optional[float]
    receive_wall_time_s: float
    receive_monotonic_s: float
    control_time_s: float
    used_source_timestamp: bool
    observation: Dict[str, Any]

    @property
    def transport_age_s(self) -> Optional[float]:
        if self.source_wall_time_s is None:
            return None
        return self.receive_wall_time_s - self.source_wall_time_s


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def producer_timestamp_to_wall_seconds(
    value: Any,
    receive_wall_time_s: float,
    max_clock_skew_s: float,
) -> Optional[float]:
    """Convert a North producer timestamp to epoch seconds when credible.

    Numeric North deployments have used seconds, milliseconds, microseconds,
    and nanoseconds. Select the scale closest to the local receive wall time.
    A protobuf Timestamp-like object with seconds/nanos is also accepted. A
    value is rejected when the resulting clock skew is too large; scheduling
    then falls back to the callback's monotonic receive time.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        seconds_attr = value.get("seconds", value.get("sec"))
        nanos_attr = value.get("nanos", value.get("nanosec", 0.0))
    else:
        seconds_attr = getattr(
            value, "seconds", getattr(value, "sec", None)
        )
        nanos_attr = getattr(
            value, "nanos", getattr(value, "nanosec", 0.0)
        )
    if seconds_attr is not None:
        sec = _finite_float(seconds_attr)
        nanos = _finite_float(nanos_attr)
        if sec is not None and nanos is not None:
            candidate = sec + nanos * 1e-9
            if abs(candidate - receive_wall_time_s) <= max_clock_skew_s:
                return candidate
        return None

    raw = _finite_float(value)
    if raw is None or raw <= 0.0:
        return None
    candidates = (raw, raw * 1e-3, raw * 1e-6, raw * 1e-9)
    candidate = min(candidates, key=lambda x: abs(x - receive_wall_time_s))
    if abs(candidate - receive_wall_time_s) > max_clock_skew_s:
        return None
    return candidate


class ObservationTimestampBuffer:
    """Bounded, monotonic buffer of unique North observations.

    append returns None when the control loop has merely polled the same
    callback result again. This prevents a 100-Hz policy loop from filling the
    model history with duplicates from a roughly 30-Hz North stream.
    """

    def __init__(
        self,
        capacity: int = 256,
        max_clock_skew_s: float = 5.0,
        max_transport_age_s: float = 0.5,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_clock_skew_s < 0 or max_transport_age_s < 0:
            raise ValueError("timestamp tolerances must be non-negative")
        self._records: Deque[ObservationStamp] = deque(maxlen=int(capacity))
        self.max_clock_skew_s = float(max_clock_skew_s)
        self.max_transport_age_s = float(max_transport_age_s)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def latest(self) -> Optional[ObservationStamp]:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> Tuple[ObservationStamp, ...]:
        return tuple(self._records)

    def append(self, observation: Dict[str, Any]) -> Optional[ObservationStamp]:
        receive_wall = _finite_float(observation.get("receive_wall_time_s"))
        receive_mono = _finite_float(observation.get("receive_monotonic_s"))
        if receive_wall is None:
            receive_wall = time.time()
        if receive_mono is None:
            receive_mono = time.monotonic()

        # Polling get_latest_observation repeatedly returns the same callback
        # receive timestamp. Do not insert or re-run inference history for it.
        if (
            self._records
            and receive_mono <= self._records[-1].receive_monotonic_s + 1e-9
        ):
            return None

        raw_source = observation.get("timestamp")
        source_wall = producer_timestamp_to_wall_seconds(
            raw_source,
            receive_wall_time_s=receive_wall,
            max_clock_skew_s=self.max_clock_skew_s,
        )
        if source_wall is not None:
            transport_age = receive_wall - source_wall
            if (
                transport_age < -0.05
                or transport_age > self.max_transport_age_s
            ):
                source_wall = None
        if source_wall is None:
            control_time = receive_mono
            used_source = False
        else:
            # Both deltas are local to the callback, so this maps epoch wall
            # time to the monotonic clock without retaining an NTP-sensitive
            # wall-clock timeline.
            control_time = receive_mono + (source_wall - receive_wall)
            used_source = True
            if (
                self._records
                and self._records[-1].source_wall_time_s is not None
                and source_wall
                <= self._records[-1].source_wall_time_s + 1e-9
            ):
                return None

        if self._records and control_time <= self._records[-1].control_time_s:
            # A small producer-clock regression should never make the command
            # timeline go backwards. The receive clock is authoritative.
            control_time = max(
                receive_mono,
                self._records[-1].control_time_s + 1e-6,
            )
            used_source = False

        stamp = ObservationStamp(
            source_timestamp_raw=raw_source,
            source_wall_time_s=source_wall,
            receive_wall_time_s=receive_wall,
            receive_monotonic_s=receive_mono,
            control_time_s=control_time,
            used_source_timestamp=used_source,
            observation=observation,
        )
        self._records.append(stamp)
        return stamp

    def age_s(self, now_monotonic_s: Optional[float] = None) -> float:
        if not self._records:
            return float("inf")
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        return float(now - self._records[-1].control_time_s)
