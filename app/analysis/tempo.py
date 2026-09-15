from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class TempoAnalysis:
    bpm: float
    candidates: list[float]
    scores: list[float]
    confidence: float
    beat_times: list[float]
    beat_intervals: list[float]


MIN_BPM = 70.0
MAX_BPM = 180.0


def _normalise_bpm(bpm: float) -> float:
    """
    Bring tempo into the normal musical range.

    We deliberately do not force everything into one narrow range.
    A genuine 150+ BPM track must remain possible.
    """
    bpm = float(bpm)

    while bpm < MIN_BPM:
        bpm *= 2.0

    while bpm > MAX_BPM:
        bpm /= 2.0

    return bpm


def _tempo_candidates_from_librosa(
    onset_envelope: np.ndarray,
    sr: int,
    hop_length: int,
) -> list[float]:
    """
    Get several independent tempo estimates from librosa.

    We use:
      - global tempo
      - local tempo estimates
      - tempogram-derived peaks

    The purpose is candidate generation, not blind selection.
    """
    candidates: list[float] = []

    # Global tempo.
    try:
        global_tempo = librosa.feature.tempo(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median,
        )

        values = np.asarray(global_tempo).flatten()

        for value in values:
            if np.isfinite(value) and value > 0:
                candidates.append(float(value))
    except Exception:
        pass

    # Local tempo estimates.
    try:
        local_tempo = librosa.feature.tempo(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
            aggregate=None,
        )

        values = np.asarray(local_tempo).flatten()

        values = values[np.isfinite(values)]
        values = values[(values >= 50.0) & (values <= 220.0)]

        if len(values):
            # Percentiles are more useful than dumping hundreds
            # of nearly identical local estimates.
            for percentile in (10, 25, 50, 75, 90):
                candidates.append(float(np.percentile(values, percentile)))
    except Exception:
        pass

    # Tempogram.
    try:
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
        )

        tempo_axis = librosa.tempo_frequencies(
            n_bins=tempogram.shape[0],
            hop_length=hop_length,
            sr=sr,
        )

        strength = np.mean(np.abs(tempogram), axis=1)

        valid = (
            np.isfinite(tempo_axis)
            & np.isfinite(strength)
            & (tempo_axis >= 50.0)
            & (tempo_axis <= 220.0)
        )

        tempo_axis = tempo_axis[valid]
        strength = strength[valid]

        if len(strength):
            # Keep the strongest tempo regions.
            order = np.argsort(strength)[::-1]

            for index in order[:12]:
                candidates.append(float(tempo_axis[index]))
    except Exception:
        pass

    return candidates


def _add_harmonic_candidates(candidates: list[float]) -> list[float]:
    """
    Add musically related half/double tempos.

    Example:
        99 -> 49.5, 198

    They are generated here but later normalised/scored.
    """
    expanded = list(candidates)

    for bpm in candidates:
        expanded.append(bpm / 2.0)
        expanded.append(bpm * 2.0)

    return expanded


def _cluster_candidates(candidates: list[float]) -> list[float]:
    """
    Merge very similar tempo estimates.

    99.1, 99.4 and 99.7 should become one candidate around 99.4.
    """
    normalised = []

    for bpm in candidates:
        bpm = _normalise_bpm(bpm)

        if MIN_BPM <= bpm <= MAX_BPM and np.isfinite(bpm):
            normalised.append(float(bpm))

    if not normalised:
        return []

    normalised.sort()

    clusters: list[list[float]] = []

    for bpm in normalised:
        if not clusters:
            clusters.append([bpm])
            continue

        previous = np.median(clusters[-1])

        # Approximately 1.5% tolerance.
        if abs(bpm - previous) / max(previous, 1.0) <= 0.015:
            clusters[-1].append(bpm)
        else:
            clusters.append([bpm])

    result = []

    for cluster in clusters:
        result.append(float(np.median(cluster)))

    return result


def _beat_strength_score(
    bpm: float,
    onset_envelope: np.ndarray,
    sr: int,
    hop_length: int,
) -> float:
    """
    Estimate how strongly the onset envelope supports a particular pulse.

    We sample the onset envelope at the candidate beat period and compare
    those positions against nearby positions.
    """
    if len(onset_envelope) < 16:
        return 0.0

    beat_period_seconds = 60.0 / bpm
    beat_period_frames = beat_period_seconds * sr / hop_length

    if beat_period_frames < 1.0:
        return 0.0

    values = []

    # Test several phase offsets so we don't depend on the first sample.
    for phase in np.linspace(0.0, beat_period_frames, 12, endpoint=False):
        positions = phase + np.arange(
            0,
            len(onset_envelope),
            beat_period_frames,
        )

        indices = np.round(positions).astype(int)
        indices = indices[
            (indices >= 0) & (indices < len(onset_envelope))
        ]

        if len(indices) < 8:
            continue

        pulse_values = onset_envelope[indices]

        # Compare pulse locations with local surrounding energy.
        offsets = np.array([-2, -1, 1, 2], dtype=float)

        neighbours = []

        for offset in offsets:
            neighbour_positions = np.round(
                positions[: len(indices)] + offset
            ).astype(int)

            valid = (
                (neighbour_positions >= 0)
                & (neighbour_positions < len(onset_envelope))
            )

            if np.any(valid):
                neighbours.append(
                    onset_envelope[neighbour_positions[valid]]
                )

        if not neighbours:
            continue

        neighbour_values = np.concatenate(neighbours)

        pulse_mean = float(np.mean(pulse_values))
        neighbour_mean = float(np.mean(neighbour_values))

        if neighbour_mean <= 1e-9:
            score = 0.0
        else:
            score = pulse_mean / neighbour_mean

        values.append(score)

    if not values:
        return 0.0

    # Convert ratio into a bounded 0..1 score.
    score = float(np.median(values))

    return float(np.clip((score - 0.8) / 1.2, 0.0, 1.0))


def _stability_score(
    bpm: float,
    beat_times: np.ndarray,
) -> float:
    """
    Measure whether detected beats agree with the candidate tempo.
    """
    if len(beat_times) < 8:
        return 0.0

    intervals = np.diff(beat_times)

    intervals = intervals[
        np.isfinite(intervals)
        & (intervals > 0.15)
        & (intervals < 2.0)
    ]

    if len(intervals) < 6:
        return 0.0

    target_interval = 60.0 / bpm

    # Allow nearby multiples because beat trackers can occasionally
    # skip or insert a beat.
    ratios = intervals / target_interval

    nearest = np.minimum.reduce(
        [
            np.abs(ratios - 0.5),
            np.abs(ratios - 1.0),
            np.abs(ratios - 2.0),
        ]
    )

    agreement = np.exp(-nearest * 5.0)

    return float(np.clip(np.median(agreement), 0.0, 1.0))


def _musical_prior(bpm: float) -> float:
    """
    Mild prior favouring common DJ/mixing tempos.

    This is deliberately weak. It must NOT override actual audio evidence.
    """
    common_ranges = [
        (80.0, 130.0),
        (130.0, 175.0),
    ]

    for low, high in common_ranges:
        if low <= bpm <= high:
            return 1.0

    if 70.0 <= bpm < 80.0:
        return 0.85

    return 0.70


def _score_candidates(
    candidates: list[float],
    onset_envelope: np.ndarray,
    sr: int,
    hop_length: int,
    beat_times: np.ndarray,
) -> list[float]:
    scores = []

    for bpm in candidates:
        pulse = _beat_strength_score(
            bpm,
            onset_envelope,
            sr,
            hop_length,
        )

        stability = _stability_score(
            bpm,
            beat_times,
        )

        prior = _musical_prior(bpm)

        # Audio evidence dominates the weak prior.
        score = (
            pulse * 0.60
            + stability * 0.30
            + prior * 0.10
        )

        scores.append(float(score))

    return scores


def _confidence(scores: list[float]) -> float:
    if not scores:
        return 0.0

    ordered = sorted(scores, reverse=True)

    best = ordered[0]

    if len(ordered) == 1:
        return float(np.clip(best, 0.0, 1.0))

    second = ordered[1]

    # Confidence should decrease when two candidates are nearly tied.
    separation = max(best - second, 0.0)

    confidence = (
        best * 0.70
        + min(separation * 2.0, 0.30)
    )

    return float(np.clip(confidence, 0.0, 1.0))


def analyze_tempo(
    path: Path,
    *,
    sr: int | None = None,
    hop_length: int = 512,
) -> TempoAnalysis:
    """
    Analyze tempo and beat grid for one audio file.

    The returned BPM is the best-supported candidate, but all candidates
    and their scores are retained so the planner can make safer decisions.
    """
    y, loaded_sr = librosa.load(
        str(path),
        sr=sr,
        mono=True,
    )

    sr = loaded_sr

    if len(y) == 0:
        raise ValueError("Audio file contains no samples.")

    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )

    if len(onset_envelope) < 16:
        raise ValueError("Audio is too short for reliable tempo analysis.")

    # First beat tracker.
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        trim=False,
    )

    tempo_values = np.asarray(tempo).flatten()

    beat_frames = np.asarray(beat_frames, dtype=int)

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=hop_length,
    )

    candidates = []

    for value in tempo_values:
        if np.isfinite(value) and value > 0:
            candidates.append(float(value))

    candidates.extend(
        _tempo_candidates_from_librosa(
            onset_envelope,
            sr,
            hop_length,
        )
    )

    candidates = _add_harmonic_candidates(candidates)
    candidates = _cluster_candidates(candidates)

    if not candidates:
        raise ValueError("Could not generate any tempo candidates.")

    scores = _score_candidates(
        candidates,
        onset_envelope,
        sr,
        hop_length,
        beat_times,
    )

    order = np.argsort(scores)[::-1]

    candidates = [candidates[i] for i in order]
    scores = [scores[i] for i in order]

    selected_bpm = candidates[0]

    # Re-run beat tracking using the selected candidate.
    try:
        _, final_beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
            bpm=selected_bpm,
            trim=False,
        )

        final_beat_frames = np.asarray(
            final_beat_frames,
            dtype=int,
        )

        if len(final_beat_frames):
            beat_times = librosa.frames_to_time(
                final_beat_frames,
                sr=sr,
                hop_length=hop_length,
            )
    except Exception:
        pass

    beat_intervals = np.diff(beat_times)

    beat_intervals = beat_intervals[
        np.isfinite(beat_intervals)
        & (beat_intervals > 0)
    ]

    confidence = _confidence(scores)

    return TempoAnalysis(
        bpm=float(selected_bpm),
        candidates=[float(x) for x in candidates[:8]],
        scores=[float(x) for x in scores[:8]],
        confidence=float(confidence),
        beat_times=[float(x) for x in beat_times],
        beat_intervals=[float(x) for x in beat_intervals],
    )


def print_tempo_analysis(analysis: TempoAnalysis) -> None:
    print(f"Selected BPM:       {analysis.bpm:.2f}")
    print(f"Confidence:         {analysis.confidence:.2f}")

    print("Candidates:")

    for bpm, score in zip(
        analysis.candidates,
        analysis.scores,
    ):
        print(
            f"  {bpm:7.2f} BPM"
            f"   score={score:.3f}"
        )

    print(f"Beats:              {len(analysis.beat_times)}")

    if analysis.beat_intervals:
        median_interval = float(
            np.median(analysis.beat_intervals)
        )

        print(
            f"Median beat interval:"
            f" {median_interval:.4f}s"
        )
