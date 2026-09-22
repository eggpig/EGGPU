#!/usr/bin/env python3
"""Strict parser and boundary contract for instrumented Gunrock executables."""

from __future__ import annotations

import re


PROTOCOL_VERSION = "gunrock_standalone_three_phase_v1"
PROTOCOL_TOKENS = (
    "matrix_market_load_to_device_graph_v1",
    "matrix_market_load_to_complete_host_result_v1",
    "processing_native_device_interval_v1",
)


class GunrockTimingProtocolError(ValueError):
    """Raised when a native log cannot prove all required timing boundaries."""


def _single_float(label: str, text: str) -> float:
    matches = re.findall(
        rf"^{re.escape(label)}\s*:\s*([0-9.eE+-]+)\s*\(ms\)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise GunrockTimingProtocolError(
            f"expected exactly one {label!r} label, observed {len(matches)}"
        )
    value = float(matches[0])
    if not value > 0.0:
        raise GunrockTimingProtocolError(f"{label} must be positive, got {value}")
    return value


def _processing_ms(text: str) -> tuple[float, str]:
    """Select the native device interval without using external process wall time."""

    candidates = (
        ("GPU Total Elapsed Time", r"^GPU Total Elapsed Time\s*:\s*([0-9.eE+-]+)\s*\(ms\)\s*$"),
        ("GPU Elapsed Time", r"^GPU Elapsed Time\s*:\s*([0-9.eE+-]+)\s*\(ms\)\s*$"),
        ("Gunrock avg. elapsed", r"avg\.\s+elapsed:\s*([0-9.eE+-]+)\s*ms"),
        ("Gunrock run elapsed", r"Run\s+\d+\s+elapsed:\s*([0-9.eE+-]+)\s*ms"),
    )
    for label, pattern in candidates:
        matches = re.findall(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if matches:
            # Maintained multi-source applications emit a single total line in
            # addition to the last-source line, so the first candidate wins.
            # Legacy LCC may report several per-run lines plus one average; the
            # average candidate precedes the per-run fallback.
            if label in {"GPU Total Elapsed Time", "GPU Elapsed Time"} and len(matches) != 1:
                raise GunrockTimingProtocolError(
                    f"expected exactly one {label!r} label, observed {len(matches)}"
                )
            value = float(matches[0])
            if not value > 0.0:
                raise GunrockTimingProtocolError(
                    f"native processing interval must be positive, got {value}"
                )
            return value, label
    raise GunrockTimingProtocolError(
        "native output contains no device processing interval"
    )


def parse_strict_gunrock_timing(text: str) -> dict:
    """Parse and validate construction, processing, and standalone E2E.

    Construction starts immediately before MatrixMarket loading and ends after
    host CSR/device graph materialization. Standalone E2E has the same start and
    ends once the complete result is host-resident, before validation, printing,
    or result-file I/O. Processing is Gunrock's native GPU device interval.
    """

    construction_ms = _single_float("Aligned Construction Time", text)
    e2e_ms = _single_float("Aligned E2E Time", text)
    processing_ms, processing_label = _processing_ms(text)

    protocol_matches = re.findall(
        r"^Aligned Timing Protocol\s*:\s*(\S.*?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if len(protocol_matches) != 1:
        raise GunrockTimingProtocolError(
            "expected exactly one 'Aligned Timing Protocol' label, "
            f"observed {len(protocol_matches)}"
        )
    protocol_tokens = tuple(
        token.strip() for token in protocol_matches[0].split(";") if token.strip()
    )
    if protocol_tokens != PROTOCOL_TOKENS:
        raise GunrockTimingProtocolError(
            f"unexpected timing protocol tokens: {protocol_tokens!r}"
        )
    tolerance_ms = max(1.0e-6, e2e_ms * 1.0e-6)
    if construction_ms > e2e_ms + tolerance_ms:
        raise GunrockTimingProtocolError(
            f"construction ({construction_ms} ms) exceeds E2E ({e2e_ms} ms)"
        )
    if processing_ms > e2e_ms + tolerance_ms:
        raise GunrockTimingProtocolError(
            f"processing ({processing_ms} ms) exceeds E2E ({e2e_ms} ms)"
        )

    return {
        "construction_seconds": construction_ms / 1000.0,
        "processing_seconds": processing_ms / 1000.0,
        "e2e_seconds": e2e_ms / 1000.0,
        "processing_source_label": processing_label,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_tokens": list(protocol_tokens),
        "timer_kind": {
            "construction": "steady_clock_wall",
            "processing": "gunrock_native_device_interval",
            "e2e": "steady_clock_wall",
        },
        "measurement_scope": "standalone_gpu_function",
        "construction_window": "matrix_market_load_to_device_graph",
        "processing_window": "native_device_interval",
        "e2e_window": "matrix_market_load_to_complete_host_result",
        "validation_outside_timer": True,
        "printing_outside_timer": True,
        "result_file_io_outside_timer": True,
        "external_cli_wall_used": False,
    }
