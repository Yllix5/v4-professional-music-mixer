from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from app.dsp.audio import (
    apply_gain,
    equal_power_crossfade,
    ensure_finite,
    peak_dbfs,
    rms_dbfs,
)
from app.dsp.gain import calculate_safe_gain


TARGET_SAMPLE_RATE = 44100


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return max(0, int(round(seconds * sample_rate)))


def _slice_audio(
    audio: np.ndarray,
    start_seconds: float,
    end_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    start = _seconds_to_samples(start_seconds, sample_rate)
    end = _seconds_to_samples(end_seconds, sample_rate)

    start = max(0, min(start, audio.shape[1]))
    end = max(start, min(end, audio.shape[1]))

    return audio[:, start:end]


def _load_track(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    audio = audio.T

    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    if source_rate != sample_rate:
        from app.dsp.audio import resample_audio

        audio = resample_audio(
            audio,
            source_rate,
            sample_rate,
        )

    return ensure_finite(audio)


def _prepare_track(
    path: Path,
    analysis,
    sample_rate: int,
) -> tuple[np.ndarray, float]:
    audio = _load_track(
        path,
        sample_rate,
    )

    gain_db = calculate_safe_gain(
        analysis.rms_dbfs,
        analysis.peak_dbfs,
    )

    audio = apply_gain(
        audio,
        gain_db,
    )

    return ensure_finite(audio), gain_db


def _get_transition_duration(
    transition,
    sample_rate: int,
) -> int:
    return _seconds_to_samples(
        transition.duration,
        sample_rate,
    )


def render_mix(
    plan,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict:
    """
    Render a MixPlan sequentially.

    The renderer follows the planner's transition positions:

        Track A:
        [start ........ source_end-duration]
                              \
                               crossfade
                              /
        Track B:
                    target_start .... target_start+duration
                                            \
                                             rest of Track B

    For every transition we preserve the material before the
    transition, crossfade exactly once, then continue from the
    incoming track after its transition segment.
    """

    if not plan.tracks:
        raise ValueError(
            "Mix plan contains no tracks."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared: list[np.ndarray] = []
    gains: list[float] = []

    # ---------------------------------------------------------
    # LOAD + GAIN MATCH
    # ---------------------------------------------------------

    for track_plan in plan.tracks:
        if not hasattr(track_plan, "analysis"):
            raise ValueError(
                f"TrackPlan for "
                f"{track_plan.path.name} has no analysis data."
            )

        audio, gain_db = _prepare_track(
            track_plan.path,
            track_plan.analysis,
            sample_rate,
        )

        prepared.append(audio)
        gains.append(gain_db)

    # ---------------------------------------------------------
    # BUILD MIX
    # ---------------------------------------------------------

    parts: list[np.ndarray] = []

    current_start_seconds = 0.0

    for i in range(len(plan.tracks) - 1):
        track_plan = plan.tracks[i]
        transition = track_plan.transition_to_next

        if transition is None:
            raise ValueError(
                f"Track {i} has no transition to the next track."
            )

        outgoing = prepared[i]
        incoming = prepared[i + 1]

        outgoing_duration = (
            outgoing.shape[1] / sample_rate
        )

        incoming_duration = (
            incoming.shape[1] / sample_rate
        )

        # -----------------------------------------------------
        # TRANSITION DURATION
        # -----------------------------------------------------

        duration = _clamp(
            float(transition.duration),
            0.1,
            min(
                outgoing_duration,
                incoming_duration,
            ),
        )

        # -----------------------------------------------------
        # SOURCE / TARGET POSITIONS
        # -----------------------------------------------------

        source_end = _clamp(
            float(transition.source_end),
            duration,
            outgoing_duration,
        )

        target_start = _clamp(
            float(transition.target_start),
            0.0,
            max(
                0.0,
                incoming_duration - duration,
            ),
        )

        # -----------------------------------------------------
        # OUTGOING PREFIX
        # -----------------------------------------------------

        outgoing_prefix = _slice_audio(
            outgoing,
            current_start_seconds,
            source_end - duration,
            sample_rate,
        )

        if outgoing_prefix.shape[1] > 0:
            parts.append(
                outgoing_prefix
            )

        # -----------------------------------------------------
        # CROSSFADE SECTIONS
        # -----------------------------------------------------

        outgoing_tail = _slice_audio(
            outgoing,
            source_end - duration,
            source_end,
            sample_rate,
        )

        incoming_head = _slice_audio(
            incoming,
            target_start,
            target_start + duration,
            sample_rate,
        )

        crossfade_samples = min(
            outgoing_tail.shape[1],
            incoming_head.shape[1],
        )

        if crossfade_samples <= 0:
            raise ValueError(
                f"Transition {i + 1} produced "
                f"an empty crossfade."
            )

        outgoing_tail = outgoing_tail[
            :,
            :crossfade_samples,
        ]

        incoming_head = incoming_head[
            :,
            :crossfade_samples,
        ]

        crossfade = equal_power_crossfade(
            outgoing_tail,
            incoming_head,
        )

        parts.append(
            crossfade
        )

        # -----------------------------------------------------
        # CONTINUE INSIDE INCOMING TRACK
        # -----------------------------------------------------

        current_start_seconds = (
            target_start + duration
        )

    # ---------------------------------------------------------
    # FINAL TRACK
    # ---------------------------------------------------------

    final_track = prepared[-1]

    final_duration = (
        final_track.shape[1] / sample_rate
    )

    final_remainder = _slice_audio(
        final_track,
        current_start_seconds,
        final_duration,
        sample_rate,
    )

    if final_remainder.shape[1] > 0:
        parts.append(
            final_remainder
        )

    if not parts:
        raise RuntimeError(
            "Renderer produced no audio."
        )

    # ---------------------------------------------------------
    # CONCATENATE
    # ---------------------------------------------------------

    mix = np.concatenate(
        parts,
        axis=1,
    )

    mix = ensure_finite(
        mix
    )

    # ---------------------------------------------------------
    # MASTER PEAK PROTECTION
    # ---------------------------------------------------------

    current_peak = peak_dbfs(
        mix
    )

    master_gain_db = 0.0

    if current_peak > -1.0:
        master_gain_db = (
            -1.0 - current_peak
        )

        mix = apply_gain(
            mix,
            master_gain_db,
        )

    mix = ensure_finite(
        mix
    )

    # ---------------------------------------------------------
    # WRITE LOSSLESS 24-BIT FLAC
    # ---------------------------------------------------------

    sf.write(
        output_path,
        mix.T,
        sample_rate,
        subtype="PCM_24",
        format="FLAC",
    )

    duration_seconds = (
        mix.shape[1] / sample_rate
    )

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    return {
        "output": str(output_path),
        "sample_rate": sample_rate,
        "channels": 2,
        "duration": duration_seconds,
        "peak_dbfs": peak_dbfs(mix),
        "rms_dbfs": rms_dbfs(mix),
        "master_gain_db": master_gain_db,
        "track_gains": [
            {
                "path": str(
                    plan.tracks[i].path
                ),
                "gain_db": gains[i],
            }
            for i in range(
                len(plan.tracks)
            )
        ],
        "transitions": [
            {
                **asdict(
                    track.transition_to_next
                ),
            }
            for track in plan.tracks[:-1]
            if track.transition_to_next
            is not None
        ],
    }


def print_render_report(
    result: dict,
) -> None:
    print()
    print("=" * 60)
    print("RENDER REPORT")
    print("=" * 60)

    print(
        f"Output:       "
        f"{result['output']}"
    )

    print(
        f"Duration:     "
        f"{result['duration']:.2f} sec"
    )

    print(
        f"Sample rate:  "
        f"{result['sample_rate']} Hz"
    )

    print(
        f"Channels:     "
        f"{result['channels']}"
    )

    print(
        f"Peak:         "
        f"{result['peak_dbfs']:.2f} dBFS"
    )

    print(
        f"RMS:          "
        f"{result['rms_dbfs']:.2f} dBFS"
    )

    print(
        f"Master gain:  "
        f"{result['master_gain_db']:.2f} dB"
    )

    print()
    print("TRACK GAINS")

    for item in result[
        "track_gains"
    ]:
        print(
            f"  {Path(item['path']).name}: "
            f"{item['gain_db']:+.2f} dB"
        )

    print()
    print("TRANSITIONS")

    for index, transition in enumerate(
        result["transitions"],
        start=1,
    ):
        print(
            f"  {index}: "
            f"{Path(transition['source_path']).name} "
            f"-> "
            f"{Path(transition['target_path']).name}"
        )

        print(
            f"      duration: "
            f"{transition['duration']:.2f}s | "
            f"source_end: "
            f"{transition['source_end']:.2f}s | "
            f"target_start: "
            f"{transition['target_start']:.2f}s | "
            f"score: "
            f"{transition['score']:.3f}"
        )

    print("=" * 60)

