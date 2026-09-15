from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .tempo import analyze_tempo


@dataclass
class TrackAnalysis:
    path: Path
    duration: float
    sample_rate: int

    # Tempo / rhythm
    bpm: float
    bpm_candidates: list[float]
    tempo_confidence: float

    beat_times: list[float]
    beat_intervals: list[float]
    downbeat_times: list[float]
    bar_times: list[float]

    # Energy / loudness
    energy_mean: float
    energy_std: float
    rms_mean: float
    rms_std: float
    peak_dbfs: float
    rms_dbfs: float

    # Spectrum
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float

    # Frequency activity
    bass_activity: float
    vocal_activity: float

    # Key estimation
    key: str
    key_confidence: float

    # Structure candidates
    intro_end: float
    outro_start: float
    structure_confidence: float


def _load_audio(path: Path, sr: int | None = None):
    y, sample_rate = librosa.load(
        str(path),
        sr=sr,
        mono=False,
    )

    if y.ndim == 1:
        mono = y
    else:
        mono = np.mean(y, axis=0)

    return y, mono.astype(np.float32), sample_rate


def _downbeats_and_bars(
    beat_times: list[float],
) -> tuple[list[float], list[float]]:
    if not beat_times:
        return [], []

    beats = np.asarray(beat_times, dtype=float)

    # First generation: assume common 4/4 phrasing.
    # We will improve meter detection later.
    downbeats = beats[::4].tolist()
    bars = downbeats.copy()

    return downbeats, bars


def _energy_analysis(
    mono: np.ndarray,
    sample_rate: int,
) -> tuple[float, float, float, float, float, float]:
    rms = librosa.feature.rms(
        y=mono,
        frame_length=2048,
        hop_length=512,
    )[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    energy = rms**2
    energy_mean = float(np.mean(energy))
    energy_std = float(np.std(energy))

    peak = float(np.max(np.abs(mono)))

    if peak <= 0:
        peak_dbfs = -120.0
    else:
        peak_dbfs = float(20.0 * np.log10(peak))

    if rms_mean <= 0:
        rms_dbfs = -120.0
    else:
        rms_dbfs = float(20.0 * np.log10(rms_mean))

    return (
        energy_mean,
        energy_std,
        rms_mean,
        rms_std,
        peak_dbfs,
        rms_dbfs,
    )


def _spectral_analysis(
    mono: np.ndarray,
    sample_rate: int,
) -> tuple[float, float, float]:
    centroid = librosa.feature.spectral_centroid(
        y=mono,
        sr=sample_rate,
        hop_length=512,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=mono,
        sr=sample_rate,
        hop_length=512,
    )[0]

    return (
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(bandwidth)),
    )


def _frequency_activity(
    mono: np.ndarray,
    sample_rate: int,
) -> tuple[float, float]:
    stft = np.abs(
        librosa.stft(
            mono,
            n_fft=2048,
            hop_length=512,
        )
    )

    frequencies = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=2048,
    )

    total_energy = float(np.sum(stft)) + 1e-12

    bass_mask = frequencies < 180.0
    vocal_mask = (frequencies >= 300.0) & (frequencies <= 4000.0)

    bass_energy = float(np.sum(stft[bass_mask]))
    vocal_energy = float(np.sum(stft[vocal_mask]))

    bass_activity = bass_energy / total_energy
    vocal_activity = vocal_energy / total_energy

    return float(bass_activity), float(vocal_activity)


def _key_analysis(
    mono: np.ndarray,
    sample_rate: int,
) -> tuple[str, float]:
    chroma = librosa.feature.chroma_cqt(
        y=mono,
        sr=sample_rate,
    )

    chroma_mean = np.mean(chroma, axis=1)

    # Krumhansl-Schmuckler style profiles.
    major_profile = np.array(
        [
            6.35,
            2.23,
            3.48,
            2.33,
            4.38,
            4.09,
            2.52,
            5.19,
            2.39,
            3.66,
            2.29,
            2.88,
        ]
    )

    minor_profile = np.array(
        [
            6.33,
            2.68,
            3.52,
            5.38,
            2.60,
            3.53,
            2.54,
            4.75,
            3.98,
            2.69,
            3.34,
            3.17,
        ]
    )

    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)

    chroma_norm = np.linalg.norm(chroma_mean)

    if chroma_norm <= 1e-12:
        return "Unknown", 0.0

    chroma_normed = chroma_mean / chroma_norm

    names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]

    best_score = -999.0
    best_key = "Unknown"

    for root in range(12):
        major_rotated = np.roll(major_profile, root)
        minor_rotated = np.roll(minor_profile, root)

        major_score = float(
            np.dot(chroma_normed, major_rotated)
        )
        minor_score = float(
            np.dot(chroma_normed, minor_rotated)
        )

        if major_score > best_score:
            best_score = major_score
            best_key = f"{names[root]} M"

        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{names[root]} m"

    confidence = float(np.clip(best_score, 0.0, 1.0))

    return best_key, confidence


def _structure_analysis(
    mono: np.ndarray,
    sample_rate: int,
    duration: float,
) -> tuple[float, float, float]:
    rms = librosa.feature.rms(
        y=mono,
        frame_length=2048,
        hop_length=512,
    )[0]

    if len(rms) < 10 or duration <= 0:
        return 0.0, max(duration - 10.0, 0.0), 0.0

    frame_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sample_rate,
        hop_length=512,
    )

    # Smooth energy curve.
    kernel_size = min(31, max(3, len(rms) // 20))

    if kernel_size % 2 == 0:
        kernel_size -= 1

    if kernel_size >= 3:
        kernel = np.ones(kernel_size) / kernel_size
        smooth = np.convolve(rms, kernel, mode="same")
    else:
        smooth = rms

    normalized = smooth / (np.max(smooth) + 1e-12)

    # Intro: first point where sustained energy reaches ~65%.
    intro_threshold = 0.65

    intro_end = 0.0

    for i in range(len(normalized)):
        if normalized[i] >= intro_threshold:
            intro_end = float(frame_times[i])
            break

    # Keep intro candidate sensible.
    intro_end = float(
        np.clip(
            intro_end,
            0.0,
            min(duration * 0.25, 30.0),
        )
    )

    # Outro: last point where sustained energy is above ~55%.
    outro_threshold = 0.55

    active_indices = np.where(
        normalized >= outro_threshold
    )[0]

    if len(active_indices):
        last_active = int(active_indices[-1])
        outro_start = float(frame_times[last_active])
    else:
        outro_start = max(duration - 15.0, 0.0)

    outro_start = float(
        np.clip(
            outro_start,
            duration * 0.65,
            max(duration - 2.0, duration * 0.65),
        )
    )

    # Conservative confidence because this is only a candidate detector.
    confidence = 0.35

    return intro_end, outro_start, confidence


def analyze_track(path: Path) -> TrackAnalysis:
    y, mono, sample_rate = _load_audio(path)

    duration = float(
        mono.shape[0] / sample_rate
    )

    # New multi-candidate tempo analyzer.
    tempo = analyze_tempo(path)

    bpm = float(tempo.bpm)
    bpm_candidates = [
        float(value)
        for value in tempo.candidates
    ]

    beat_times = [
        float(value)
        for value in tempo.beat_times
    ]

    beat_intervals = [
        float(value)
        for value in tempo.beat_intervals
    ]

    downbeat_times, bar_times = _downbeats_and_bars(
        beat_times
    )

    (
        energy_mean,
        energy_std,
        rms_mean,
        rms_std,
        peak_dbfs,
        rms_dbfs,
    ) = _energy_analysis(
        mono,
        sample_rate,
    )

    (
        spectral_centroid_mean,
        spectral_centroid_std,
        spectral_bandwidth_mean,
    ) = _spectral_analysis(
        mono,
        sample_rate,
    )

    (
        bass_activity,
        vocal_activity,
    ) = _frequency_activity(
        mono,
        sample_rate,
    )

    key, key_confidence = _key_analysis(
        mono,
        sample_rate,
    )

    (
        intro_end,
        outro_start,
        structure_confidence,
    ) = _structure_analysis(
        mono,
        sample_rate,
        duration,
    )

    return TrackAnalysis(
        path=path,
        duration=duration,
        sample_rate=sample_rate,
        bpm=bpm,
        bpm_candidates=bpm_candidates,
        tempo_confidence=float(tempo.confidence),
        beat_times=beat_times,
        beat_intervals=beat_intervals,
        downbeat_times=downbeat_times,
        bar_times=bar_times,
        energy_mean=energy_mean,
        energy_std=energy_std,
        rms_mean=rms_mean,
        rms_std=rms_std,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        spectral_centroid_mean=spectral_centroid_mean,
        spectral_centroid_std=spectral_centroid_std,
        spectral_bandwidth_mean=spectral_bandwidth_mean,
        bass_activity=bass_activity,
        vocal_activity=vocal_activity,
        key=key,
        key_confidence=key_confidence,
        intro_end=intro_end,
        outro_start=outro_start,
        structure_confidence=structure_confidence,
    )


def analyze_tracks(
    paths: list[Path],
) -> list[TrackAnalysis]:
    results: list[TrackAnalysis] = []

    total = len(paths)

    for index, path in enumerate(paths, start=1):
        print(
            f"[{index:02d}/{total:02d}] {path.name}"
        )

        try:
            analysis = analyze_track(path)

            print(
                f"     BPM: {analysis.bpm:.2f}"
                f" | Key: {analysis.key}"
                f" | Confidence:"
                f" {analysis.tempo_confidence:.2f}"
            )

            results.append(analysis)

        except Exception as exc:
            print(
                f"     [WARNING] Analysis failed:"
                f" {exc}"
            )

    return results


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60

    return f"{minutes}:{remaining:05.2f}"


def print_analysis_report(
    analyses: list[TrackAnalysis],
) -> None:
    print()
    print("=" * 70)
    print("V4 TRACK ANALYSIS")
    print("=" * 70)

    for index, track in enumerate(analyses, start=1):
        candidates_text = ", ".join(
            f"{value:.2f}"
            for value in track.bpm_candidates
        )

        print()
        print(
            f"[{index:02d}] {track.path.name}"
        )

        print(
            f"    Duration:             "
            f"{_format_seconds(track.duration)}"
        )

        print(
            f"    Sample rate:          "
            f"{track.sample_rate} Hz"
        )

        print(
            f"    BPM:                  "
            f"{track.bpm:.2f}"
        )

        print(
            f"    BPM candidates:       "
            f"{candidates_text}"
        )

        print(
            f"    Tempo confidence:     "
            f"{track.tempo_confidence:.2f}"
        )

        print(
            f"    Beats:                "
            f"{len(track.beat_times)}"
        )

        print(
            f"    Downbeats:            "
            f"{len(track.downbeat_times)}"
        )

        print(
            f"    Key:                  "
            f"{track.key}"
            f" ({track.key_confidence:.2f})"
        )

        print(
            f"    RMS:                  "
            f"{track.rms_dbfs:.2f} dBFS"
        )

        print(
            f"    Peak:                 "
            f"{track.peak_dbfs:.2f} dBFS"
        )

        print(
            f"    Bass activity:        "
            f"{track.bass_activity:.3f}"
        )

        print(
            f"    Vocal activity:       "
            f"{track.vocal_activity:.3f}"
        )

        print(
            f"    Intro candidate:      "
            f"{_format_seconds(track.intro_end)}"
        )

        print(
            f"    Outro candidate:      "
            f"{_format_seconds(track.outro_start)}"
        )

        print(
            f"    Structure confidence: "
            f"{track.structure_confidence:.2f}"
        )

    print()
    print("=" * 70)