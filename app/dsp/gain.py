from __future__ import annotations

import numpy as np


TARGET_RMS_DBFS = -14.0
MAX_GAIN_DB = 3.0
MIN_GAIN_DB = -6.0


def calculate_safe_gain(
    current_rms_dbfs: float,
    current_peak_dbfs: float,
) -> float:
    """
    Calculate conservative gain matching.

    We intentionally do NOT normalize everything
    aggressively. Retail playback benefits from
    consistency without destroying dynamics.
    """

    desired_gain = (
        TARGET_RMS_DBFS
        - current_rms_dbfs
    )

    desired_gain = max(
        MIN_GAIN_DB,
        min(MAX_GAIN_DB, desired_gain),
    )

    # Keep at least ~1 dB peak headroom.
    peak_after_gain = (
        current_peak_dbfs
        + desired_gain
    )

    if peak_after_gain > -1.0:
        desired_gain -= (
            peak_after_gain + 1.0
        )

    return float(
        max(
            MIN_GAIN_DB,
            min(MAX_GAIN_DB, desired_gain),
        )
    )


def apply_gain_envelope(
    audio: np.ndarray,
    start_gain_db: float,
    end_gain_db: float,
) -> np.ndarray:
    """
    Smooth gain automation across the whole buffer.
    """

    if audio.shape[1] == 0:
        return audio

    envelope_db = np.linspace(
        start_gain_db,
        end_gain_db,
        audio.shape[1],
        dtype=np.float32,
    )

    envelope = (
        10.0 ** (envelope_db / 20.0)
    )

    return (
        audio * envelope
    ).astype(np.float32)