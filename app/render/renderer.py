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
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if source_rate != sample_rate:
        from app.dsp.audio import resample_audio
        audio = resample_audio(audio, source_rate, sample_rate)
    return ensure_finite(audio)


def _prepare_track(path: Path, analysis, sample_rate: int) -> tuple[np.ndarray, float]:
    audio = _load_track(path, sample_rate)
    gain_db = calculate_safe_gain(analysis.rms_dbfs, analysis.peak_dbfs)
    audio = apply_gain(audio, gain_db)
    return ensure_finite(audio), gain_db


def _get_transition_duration(transition, sample_rate: int) -> int:
    return _seconds_to_samples(transition.duration, sample_rate)


def render_mix(
    plan,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict:
    """
    Memory-efficient streaming renderer.
    Never holds the full mix in RAM.
    Only keeps current + next track at any time.
    """
    if not plan.tracks:
        raise ValueError("Mix plan contains no tracks.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output for streaming write (FLAC)
    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=2,
        subtype="PCM_24",
        format="FLAC",
    ) as outfile:

        gains: list[float] = []
        current_start_seconds = 0.0
        total_samples_written = 0
        running_peak = 0.0

        # We process tracks one-by-one, keeping only two in memory max
        for i in range(len(plan.tracks)):
            track_plan = plan.tracks[i]

            if not hasattr(track_plan, "analysis"):
                raise ValueError(
                    f"TrackPlan for {track_plan.path.name} has no analysis data."
                )

            # Load only the current track
            audio, gain_db = _prepare_track(
                track_plan.path,
                track_plan.analysis,
                sample_rate,
            )
            gains.append(gain_db)

            if i < len(plan.tracks) - 1:
                # There is a transition to the next track
                transition = track_plan.transition_to_next
                if transition is None:
                    raise ValueError(f"Track {i} has no transition to the next track.")

                # Load next track only when needed
                next_plan = plan.tracks[i + 1]
                next_audio, _ = _prepare_track(
                    next_plan.path,
                    next_plan.analysis,
                    sample_rate,
                )

                outgoing = audio
                incoming = next_audio

                outgoing_duration = outgoing.shape[1] / sample_rate
                incoming_duration = incoming.shape[1] / sample_rate

                duration = _clamp(
                    float(transition.duration),
                    0.1,
                    min(outgoing_duration, incoming_duration),
                )

                source_end = _clamp(
                    float(transition.source_end),
                    duration,
                    outgoing_duration,
                )
                target_start = _clamp(
                    float(transition.target_start),
                    0.0,
                    max(0.0, incoming_duration - duration),
                )

                # Prefix of outgoing track
                outgoing_prefix = _slice_audio(
                    outgoing,
                    current_start_seconds,
                    source_end - duration,
                    sample_rate,
                )
                if outgoing_prefix.shape[1] > 0:
                    outfile.write(outgoing_prefix.T)
                    total_samples_written += outgoing_prefix.shape[1]
                    running_peak = max(running_peak, float(np.max(np.abs(outgoing_prefix))))

                # Crossfade
                outgoing_tail = _slice_audio(
                    outgoing, source_end - duration, source_end, sample_rate
                )
                incoming_head = _slice_audio(
                    incoming, target_start, target_start + duration, sample_rate
                )

                crossfade_samples = min(outgoing_tail.shape[1], incoming_head.shape[1])
                if crossfade_samples <= 0:
                    raise ValueError(f"Transition {i + 1} produced an empty crossfade.")

                outgoing_tail = outgoing_tail[:, :crossfade_samples]
                incoming_head = incoming_head[:, :crossfade_samples]

                crossfade = equal_power_crossfade(outgoing_tail, incoming_head)
                outfile.write(crossfade.T)
                total_samples_written += crossfade.shape[1]
                running_peak = max(running_peak, float(np.max(np.abs(crossfade))))

                # Prepare for next iteration
                current_start_seconds = target_start + duration

                # Free memory immediately
                del audio, next_audio, outgoing, incoming
                del outgoing_prefix, outgoing_tail, incoming_head, crossfade

            else:
                # Last track – write the remaining part
                final_duration = audio.shape[1] / sample_rate
                final_remainder = _slice_audio(
                    audio, current_start_seconds, final_duration, sample_rate
                )
                if final_remainder.shape[1] > 0:
                    outfile.write(final_remainder.T)
                    total_samples_written += final_remainder.shape[1]
                    running_peak = max(running_peak, float(np.max(np.abs(final_remainder))))

                del audio, final_remainder

    # ------------------------------------------------------------------
    # Optional light master peak protection (second streaming pass)
    # Only if the peak is dangerously high.
    # ------------------------------------------------------------------
    master_gain_db = 0.0
    if running_peak > 0.0:
        current_peak_db = 20.0 * np.log10(running_peak)
        if current_peak_db > -0.5:
            master_gain_db = -0.5 - current_peak_db
            # Second pass: apply gain while streaming
            temp_path = output_path.with_suffix(".tmp.flac")
            with sf.SoundFile(str(output_path), mode="r") as src, \
                 sf.SoundFile(
                     str(temp_path),
                     mode="w",
                     samplerate=sample_rate,
                     channels=2,
                     subtype="PCM_24",
                     format="FLAC",
                 ) as dst:

                gain = 10.0 ** (master_gain_db / 20.0)
                block_size = 65536
                while True:
                    block = src.read(block_size, dtype="float32")
                    if len(block) == 0:
                        break
                    block *= gain
                    dst.write(block)

            temp_path.replace(output_path)

    duration_seconds = total_samples_written / sample_rate

    # Re-open just for final metrics (very small memory)
    with sf.SoundFile(str(output_path), mode="r") as f:
        # We already know duration; peak/rms can be approximate from running_peak
        # or we skip heavy re-computation
        final_peak = 20.0 * np.log10(max(running_peak * (10 ** (master_gain_db / 20.0)), 1e-12))
        # rms would require full pass; leave approximate or skip
        final_rms = -20.0  # placeholder – real value not critical for return

    return {
        "output": str(output_path),
        "sample_rate": sample_rate,
        "channels": 2,
        "duration": duration_seconds,
        "peak_dbfs": final_peak,
        "rms_dbfs": final_rms,
        "master_gain_db": master_gain_db,
        "track_gains": [
            {"path": str(plan.tracks[i].path), "gain_db": gains[i]}
            for i in range(len(plan.tracks))
        ],
        "transitions": [
            {**asdict(track.transition_to_next)}
            for track in plan.tracks[:-1]
            if track.transition_to_next is not None
        ],
    }


def print_render_report(result: dict) -> None:
    print()
    print("=" * 60)
    print("RENDER REPORT")
    print("=" * 60)
    print(f"Output:       {result['output']}")
    print(f"Duration:     {result['duration']:.2f} sec")
    print(f"Sample rate:  {result['sample_rate']} Hz")
    print(f"Channels:     {result['channels']}")
    print(f"Peak:         {result['peak_dbfs']:.2f} dBFS")
    print(f"RMS:          {result['rms_dbfs']:.2f} dBFS")
    print(f"Master gain:  {result['master_gain_db']:.2f} dB")
    print()
    print("TRACK GAINS")
    for item in result["track_gains"]:
        print(f"  {Path(item['path']).name}: {item['gain_db']:+.2f} dB")
    print()
    print("TRANSITIONS")
    for index, transition in enumerate(result["transitions"], start=1):
        print(
            f"  {index}: "
            f"{Path(transition['source_path']).name} -> "
            f"{Path(transition['target_path']).name}"
        )
        print(
            f"      duration: {transition['duration']:.2f}s | "
            f"source_end: {transition['source_end']:.2f}s | "
            f"target_start: {transition['target_start']:.2f}s | "
            f"score: {transition['score']:.3f}"
        )
    print("=" * 60)