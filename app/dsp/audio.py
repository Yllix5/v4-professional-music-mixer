from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(
    path: Path,
    target_sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load audio as float32 stereo.

    Returns:
        audio: shape (channels, samples)
        sample_rate: integer sample rate
    """

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # soundfile returns (samples, channels)
    audio = audio.T

    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])

    elif audio.shape[0] > 2:
        audio = audio[:2]

    if target_sample_rate is not None:
        if sample_rate != target_sample_rate:
            audio = resample_audio(
                audio,
                sample_rate,
                target_sample_rate,
            )
            sample_rate = target_sample_rate

    return np.asarray(audio, dtype=np.float32), sample_rate


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """
    High-quality rational-factor resampling.
    """

    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    gcd = np.gcd(
        source_rate,
        target_rate,
    )

    up = target_rate // gcd
    down = source_rate // gcd

    channels = []

    for channel in audio:
        resampled = resample_poly(
            channel,
            up,
            down,
        )
        channels.append(resampled)

    return np.asarray(
        channels,
        dtype=np.float32,
    )


def ensure_finite(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Replace NaN/Inf values safely.
    """

    if np.all(np.isfinite(audio)):
        return audio

    return np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def apply_gain(
    audio: np.ndarray,
    gain_db: float,
) -> np.ndarray:
    """
    Apply linear gain to audio.
    """

    if gain_db == 0.0:
        return audio

    gain = 10.0 ** (gain_db / 20.0)

    return (
        audio * gain
    ).astype(np.float32)


def fade_in(
    audio: np.ndarray,
    duration_samples: int,
) -> np.ndarray:
    if duration_samples <= 0:
        return audio

    duration_samples = min(
        duration_samples,
        audio.shape[1],
    )

    curve = np.linspace(
        0.0,
        1.0,
        duration_samples,
        dtype=np.float32,
    )

    audio[:, :duration_samples] *= curve

    return audio


def fade_out(
    audio: np.ndarray,
    duration_samples: int,
) -> np.ndarray:
    if duration_samples <= 0:
        return audio

    duration_samples = min(
        duration_samples,
        audio.shape[1],
    )

    curve = np.linspace(
        1.0,
        0.0,
        duration_samples,
        dtype=np.float32,
    )

    audio[:, -duration_samples:] *= curve

    return audio


def equal_power_crossfade(
    outgoing: np.ndarray,
    incoming: np.ndarray,
) -> np.ndarray:
    """
    Equal-power stereo crossfade.

    Both arrays must have shape:
        (channels, samples)

    and the same number of samples.
    """

    samples = min(
        outgoing.shape[1],
        incoming.shape[1],
    )

    outgoing = outgoing[:, :samples]
    incoming = incoming[:, :samples]

    position = np.linspace(
        0.0,
        1.0,
        samples,
        dtype=np.float32,
    )

    out_curve = np.cos(
        position * np.pi / 2.0
    )

    in_curve = np.sin(
        position * np.pi / 2.0
    )

    return (
        outgoing * out_curve
        + incoming * in_curve
    ).astype(np.float32)


def peak_dbfs(
    audio: np.ndarray,
) -> float:
    peak = float(
        np.max(np.abs(audio))
    )

    if peak <= 0.0:
        return -120.0

    return float(
        20.0 * np.log10(peak)
    )


def rms_dbfs(
    audio: np.ndarray,
) -> float:
    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio),
            )
        )
    )

    if rms <= 0.0:
        return -120.0

    return float(
        20.0 * np.log10(rms)
    )