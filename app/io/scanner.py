from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".flac", ".wav"}


@dataclass
class AudioInfo:
    path: Path
    codec: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int | None
    bitrate: int | None


@dataclass
class ScanResult:
    valid: list[AudioInfo]
    skipped: list[tuple[Path, str]]


def probe_file(path: Path) -> AudioInfo:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_raw_sample,bits_per_sample,bit_rate",
        "-show_entries",
        "format=duration,bit_rate",
        "-of", "json",
        str(path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFprobe failed")

    data = json.loads(result.stdout)

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No audio stream found")

    stream = streams[0]
    fmt = data.get("format", {})

    duration = float(fmt.get("duration", 0))

    if duration <= 0:
        raise RuntimeError("Invalid or zero duration")

    sample_rate = int(stream.get("sample_rate", 0))
    channels = int(stream.get("channels", 0))

    if sample_rate <= 0:
        raise RuntimeError("Invalid sample rate")

    if channels <= 0:
        raise RuntimeError("Invalid channel count")

    bit_depth = (
        stream.get("bits_per_raw_sample")
        or stream.get("bits_per_sample")
    )

    bitrate = stream.get("bit_rate") or fmt.get("bit_rate")

    return AudioInfo(
        path=path,
        codec=stream.get("codec_name", "unknown"),
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=int(bit_depth) if bit_depth else None,
        bitrate=int(bitrate) if bitrate else None,
    )


def scan_folder(folder: str | Path) -> ScanResult:
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    valid: list[AudioInfo] = []
    skipped: list[tuple[Path, str]] = []

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for path in files:
        try:
            info = probe_file(path)
            valid.append(info)

        except Exception as exc:
            skipped.append((path, str(exc)))

    return ScanResult(valid=valid, skipped=skipped)


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes}:{secs:02d}"


def print_scan_report(result: ScanResult) -> None:
    print()
    print("=" * 60)
    print("                    AUDIO SCAN")
    print("=" * 60)

    for index, info in enumerate(result.valid, start=1):
        print(f"[OK] {index:02d}  {info.path.name}")
        print(
            f"     {format_duration(info.duration)} | "
            f"{info.sample_rate} Hz | "
            f"{info.channels} ch | "
            f"{info.bit_depth or '?'} bit | "
            f"{info.codec}"
        )

    for path, reason in result.skipped:
        print(f"[WARN] {path.name}")
        print(f"       {reason}")

    print()
    print(f"Valid tracks : {len(result.valid)}")
    print(f"Skipped      : {len(result.skipped)}")
    print("=" * 60)