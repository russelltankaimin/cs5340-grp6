#!/usr/bin/env python3
"""
Single-file frequency-based audio corruption tool.

Features
- Bit Crushing (bit-depth reduction + sample-and-hold decimation)
- FFT Frequency Corruption (random bin masking, magnitude noise, phase noise)

Region control
- Corrupt the full audio
- Corrupt explicit time ranges
- Corrupt random segments

Examples
--------
# Full-file bit crushing to 4 bits
python audio_corruptor_freq.py input.wav output.wav --mode bit_crush --bits 4
                                    OR
uv run audio_corruptor_freq.py input.wav output.wav --mode bit_crush --bits 4

# Full-file FFT masking (zero out 20% of bins)
python audio_corruptor_freq.py input.wav output.wav --mode fft_corruption --mask_ratio 0.2
                                    OR
uv run audio_corruptor_freq.py input.wav output.wav --mode fft_corruption --mask_ratio 0.2

# Corrupt 4 random regions with bit crushing (8 bits, decimation factor 4)
python audio_corruptor_freq.py input.wav output.wav --mode bit_crush \
    --region_mode random_segments --num_segments 4 --segment_duration 0.75 \
    --bits 8 --decimation 4

Dependencies
------------
pip install numpy scipy soundfile
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal


# ============================================================
# I/O
# ============================================================

def load_audio(path: str, target_sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load audio as float32 waveform in roughly [-1, 1]."""
    audio, sr = sf.read(path, always_2d=False)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        max_abs = max(abs(info.min), abs(info.max))
        audio = audio.astype(np.float32) / float(max_abs)
    else:
        audio = audio.astype(np.float32)

    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_sr is not None and target_sr != sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = signal.resample_poly(audio, up, down).astype(np.float32)
        sr = target_sr

    return audio, sr


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr, subtype="FLOAT")


def peak_normalize(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(audio)))
    if m < 1e-12:
        return audio.astype(np.float32)
    return (audio / m * peak).astype(np.float32)


# ============================================================
# Utils
# ============================================================

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))) + 1e-12))


# ============================================================
# Corruption modes
# ============================================================

def bit_crush_corruption(
    clean: np.ndarray,
    bits: int,
    decimation: int,
) -> Tuple[np.ndarray, dict]:
    """
    Applies bit-depth reduction and sample-and-hold decimation.
    """
    y = clean.copy()

    # 1. Bit-depth reduction (quantization)
    if bits < 32:
        q = 2 ** (bits - 1)
        y = np.round(y * q) / q

    # 2. Decimation (sample-and-hold)
    if decimation > 1:
        indices = np.arange(len(y))
        downsampled_indices = (indices // decimation) * decimation
        # Clip to ensure indices don't overflow (though they shouldn't)
        downsampled_indices = np.clip(downsampled_indices, 0, len(y) - 1)
        y = y[downsampled_indices]

    return y.astype(np.float32), {
        "mode": "bit_crush",
        "bits": bits,
        "decimation": decimation,
    }


def fft_frequency_corruption(
    clean: np.ndarray,
    mask_ratio: float,
    noise_std: float,
    phase_noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, dict]:
    """
    Manipulates the audio in the frequency domain.
    """
    n = len(clean)
    X = np.fft.rfft(clean)
    magnitudes = np.abs(X)
    phases = np.angle(X)

    # 1. Bin Masking
    if mask_ratio > 0:
        mask = rng.uniform(0, 1, size=len(magnitudes)) > mask_ratio
        magnitudes *= mask

    # 2. Magnitude Noise
    if noise_std > 0:
        # Scale noise relative to the mean magnitude to keep it reasonable
        avg_mag = np.mean(magnitudes)
        mag_noise = rng.standard_normal(len(magnitudes)).astype(np.float32) * noise_std * avg_mag
        magnitudes = np.maximum(0, magnitudes + mag_noise)

    # 3. Phase Noise
    if phase_noise_std > 0:
        phase_noise = rng.standard_normal(len(phases)).astype(np.float32) * phase_noise_std
        phases += phase_noise

    # Reconstruction
    X_new = magnitudes * np.exp(1j * phases)
    y = np.fft.irfft(X_new, n=n)

    # Maintain energy level
    clean_rms = rms(clean)
    y_rms = rms(y)
    if y_rms > 1e-12:
        y = y * (clean_rms / y_rms)

    return y.astype(np.float32), {
        "mode": "fft_corruption",
        "mask_ratio": mask_ratio,
        "noise_std": noise_std,
        "phase_noise_std": phase_noise_std,
    }


def corrupt_segment(
    clean: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    bits: int,
    decimation: int,
    mask_ratio: float,
    noise_std: float,
    phase_noise_std: float,
) -> Tuple[np.ndarray, dict]:
    mode = mode.lower()
    if mode == "bit_crush":
        return bit_crush_corruption(
            clean=clean,
            bits=bits,
            decimation=decimation,
        )
    if mode == "fft_corruption":
        return fft_frequency_corruption(
            clean=clean,
            mask_ratio=mask_ratio,
            noise_std=noise_std,
            phase_noise_std=phase_noise_std,
            rng=rng,
        )
    raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Region selection
# ============================================================

def parse_ranges(ranges_text: str, sr: int, total_samples: int) -> List[Tuple[int, int]]:
    """
    Parse string like '0.5:1.2,3.0:4.0' into sample ranges.
    """
    result: List[Tuple[int, int]] = []
    if not ranges_text.strip():
        return result

    for chunk in ranges_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid range '{chunk}'. Expected start:end in seconds.")
        start_s, end_s = chunk.split(":", 1)
        start = max(0, int(round(float(start_s) * sr)))
        end = min(total_samples, int(round(float(end_s) * sr)))
        if end <= start:
            continue
        result.append((start, end))
    return merge_intervals(result)


def merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def sample_random_segments(
    total_samples: int,
    sr: int,
    rng: np.random.Generator,
    num_segments: int,
    segment_duration: Optional[float],
    min_segment_duration: float,
    max_segment_duration: float,
    allow_overlap: bool,
) -> List[Tuple[int, int]]:
    if num_segments <= 0:
        return []

    intervals: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = max(100, num_segments * 40)

    while len(intervals) < num_segments and attempts < max_attempts:
        attempts += 1
        dur_s = segment_duration if segment_duration is not None else float(rng.uniform(min_segment_duration, max_segment_duration))
        seg_len = max(1, int(round(dur_s * sr)))
        if seg_len >= total_samples:
            candidate = (0, total_samples)
        else:
            start = int(rng.integers(0, total_samples - seg_len + 1))
            candidate = (start, start + seg_len)

        if allow_overlap:
            intervals.append(candidate)
            continue

        overlap = False
        for a, b in intervals:
            if not (candidate[1] <= a or candidate[0] >= b):
                overlap = True
                break
        if not overlap:
            intervals.append(candidate)

    return merge_intervals(intervals)


def build_regions(
    region_mode: str,
    total_samples: int,
    sr: int,
    rng: np.random.Generator,
    ranges_text: str,
    num_segments: int,
    segment_duration: Optional[float],
    min_segment_duration: float,
    max_segment_duration: float,
    allow_overlap: bool,
) -> List[Tuple[int, int]]:
    region_mode = region_mode.lower()
    if region_mode == "full":
        return [(0, total_samples)]
    if region_mode == "time_ranges":
        return parse_ranges(ranges_text, sr, total_samples)
    if region_mode == "random_segments":
        return sample_random_segments(
            total_samples=total_samples,
            sr=sr,
            rng=rng,
            num_segments=num_segments,
            segment_duration=segment_duration,
            min_segment_duration=min_segment_duration,
            max_segment_duration=max_segment_duration,
            allow_overlap=allow_overlap,
        )
    raise ValueError(f"Unsupported region_mode: {region_mode}")


# ============================================================
# Region application with fades
# ============================================================

def apply_corruption_to_regions(
    clean: np.ndarray,
    sr: int,
    regions: Sequence[Tuple[int, int]],
    corrupt_fn: Callable[[np.ndarray], Tuple[np.ndarray, dict]],
    fade_ms: float = 10.0,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Applies corruption only to selected regions. Each corrupted region is blended
    back into the original signal with a short fade to reduce clicks.
    """
    out = clean.astype(np.float32).copy()
    region_metadata: List[dict] = []
    fade_samples = max(0, int(round(fade_ms * sr / 1000.0)))

    for idx, (start, end) in enumerate(regions):
        if end <= start:
            continue
        segment = clean[start:end]
        corrupted_segment, meta = corrupt_fn(segment)

        blend = np.ones(len(segment), dtype=np.float32)
        f = min(fade_samples, len(segment) // 2)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            blend[:f] = ramp
            blend[-f:] = ramp[::-1]

        out[start:end] = (1.0 - blend) * clean[start:end] + blend * corrupted_segment
        region_metadata.append(
            {
                "region_index": idx,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_time_sec": float(start / sr),
                "end_time_sec": float(end / sr),
                "duration_sec": float((end - start) / sr),
                **meta,
            }
        )

    return out.astype(np.float32), region_metadata


# ============================================================
# Metadata writing
# ============================================================

def write_metadata_txt(path: str, metadata: dict) -> None:
    lines: List[str] = []
    for key, value in metadata.items():
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                lines.append(f"{key}=")
                for item in value:
                    lines.append("  - " + ", ".join(f"{k}={v}" for k, v in item.items()))
            else:
                lines.append(f"{key}=" + ",".join(map(str, value)))
        else:
            lines.append(f"{key}={value}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regions_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-file frequency-based audio corruption tool.")
    p.add_argument("input", type=str, help="Input clean audio file")
    p.add_argument("output", type=str, help="Output corrupted audio file")

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["bit_crush", "fft_corruption"],
        help="Corruption mode",
    )
    p.add_argument("--sr", type=int, default=None, help="Optional target sample rate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--normalize", action="store_true", help="Peak normalize after corruption")

    # Bit Crush controls
    p.add_argument("--bits", type=int, default=8, help="Target bit depth for bit_crush mode")
    p.add_argument("--decimation", type=int, default=1, help="Decimation factor for bit_crush mode")

    # FFT Corruption controls
    p.add_argument("--mask_ratio", type=float, default=0.1, help="Ratio of FFT bins to zero out")
    p.add_argument("--noise_std", type=float, default=0.0, help="Magnitude noise std relative to mean magnitude")
    p.add_argument("--phase_noise_std", type=float, default=0.0, help="Phase noise std (radians)")

    # Region controls
    p.add_argument(
        "--region_mode",
        type=str,
        default="full",
        choices=["full", "time_ranges", "random_segments"],
        help="Where to apply corruption",
    )
    p.add_argument(
        "--ranges",
        type=str,
        default="",
        help="For time_ranges mode: comma-separated start:end in seconds, e.g. 0.5:1.2,3.0:4.0",
    )
    p.add_argument("--num_segments", type=int, default=3, help="For random_segments mode")
    p.add_argument("--segment_duration", type=float, default=None, help="Fixed duration in seconds for random segments")
    p.add_argument("--min_segment_duration", type=float, default=0.25, help="Minimum random segment duration in seconds")
    p.add_argument("--max_segment_duration", type=float, default=1.0, help="Maximum random segment duration in seconds")
    p.add_argument("--allow_overlap", action="store_true", help="Allow overlapping random segments")
    p.add_argument("--fade_ms", type=float, default=10.0, help="Crossfade at region boundaries in milliseconds")

    return p


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    clean, sr = load_audio(args.input, target_sr=args.sr, mono=True)
    total_samples = len(clean)

    regions = build_regions(
        region_mode=args.region_mode,
        total_samples=total_samples,
        sr=sr,
        rng=rng,
        ranges_text=args.ranges,
        num_segments=args.num_segments,
        segment_duration=args.segment_duration,
        min_segment_duration=args.min_segment_duration,
        max_segment_duration=args.max_segment_duration,
        allow_overlap=args.allow_overlap,
    )

    if not regions:
        raise ValueError("No valid regions were selected for corruption.")

    def _corrupt_fn(segment: np.ndarray) -> Tuple[np.ndarray, dict]:
        return corrupt_segment(
            clean=segment,
            mode=args.mode,
            rng=rng,
            bits=args.bits,
            decimation=args.decimation,
            mask_ratio=args.mask_ratio,
            noise_std=args.noise_std,
            phase_noise_std=args.phase_noise_std,
        )

    corrupted, region_metadata = apply_corruption_to_regions(
        clean=clean,
        sr=sr,
        regions=regions,
        corrupt_fn=_corrupt_fn,
        fade_ms=args.fade_ms,
    )

    if args.normalize:
        corrupted = peak_normalize(corrupted)

    save_audio(args.output, corrupted, sr)

    base = Path(args.output)
    txt_path = str(base.with_suffix(".txt"))
    csv_path = str(base.with_name(base.stem + "_regions.csv"))

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "sample_rate": sr,
        "num_samples": total_samples,
        "duration_sec": total_samples / sr,
        "mode": args.mode,
        "region_mode": args.region_mode,
        "num_regions": len(region_metadata),
        "fade_ms": args.fade_ms,
        "seed": args.seed,
        "regions": region_metadata,
    }

    write_metadata_txt(txt_path, metadata)
    write_regions_csv(csv_path, region_metadata)

    print(f"Saved corrupted audio: {args.output}")
    print(f"Saved metadata: {txt_path}")
    print(f"Saved region CSV: {csv_path}")


if __name__ == "__main__":
    main()
