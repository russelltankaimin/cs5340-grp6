#!/usr/bin/env python3
"""
Single-file audio corruption tool.

Features
- Additive white Gaussian noise (AWGN / gaussian / white)
- Pink noise
- Brown noise
- Hum noise (50/60 Hz + harmonics)
- Band-limited noise
- Random smooth frequency-response distortion (random EQ)
- Combo mode: random EQ + Gaussian noise

Region control
- Corrupt the full audio
- Corrupt explicit time ranges
- Corrupt random segments

Examples
--------
# Full-file Gaussian noise
python audio_corruptor_single.py input.wav sample_bitcrushing.wav --mode gaussian --snr_db 10

# Only corrupt 0.5-1.2s and 3.0-4.0s
python audio_corruptor_single.py input.wav sample_bitcrushing.wav --mode pink \
    --region_mode time_ranges --ranges 0.5:1.2,3.0:4.0

# Corrupt 4 random regions of 0.75s each
python audio_corruptor_single.py input.wav sample_bitcrushing.wav --mode combo \
    --region_mode random_segments --num_segments 4 --segment_duration 0.75 \
    --snr_db 8 --bands 12 --max_gain_db 6

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


def scale_noise_for_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + 1e-12)
    return (noise * scale).astype(np.float32)


def db_to_linear(gain_db: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (np.asarray(gain_db) / 20.0)


# ============================================================
# Noise generation
# ============================================================

def gaussian_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    # In this script, white noise is Gaussian white noise.
    return gaussian_noise(n, rng)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    shaping = np.ones_like(freqs)
    shaping[1:] = 1.0 / np.sqrt(freqs[1:])
    y = np.fft.irfft(X * shaping, n=n)
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-12)
    return y.astype(np.float32)


def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    y = np.cumsum(rng.standard_normal(n))
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-12)
    return y.astype(np.float32)


def hum_noise(n: int, sr: int, hum_freq: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sr)
    y = (
        np.sin(2 * np.pi * hum_freq * t)
        + 0.5 * np.sin(2 * np.pi * 2 * hum_freq * t)
        + 0.25 * np.sin(2 * np.pi * 3 * hum_freq * t)
        + 0.01 * rng.standard_normal(n)
    )
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-12)
    return y.astype(np.float32)


def band_limited_noise(n: int, sr: int, low_hz: float, high_hz: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(n).astype(np.float32)
    nyq = sr / 2.0
    low = max(1.0, low_hz) / nyq
    high = min(high_hz, nyq * 0.999) / nyq
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid band: low={low_hz}, high={high_hz}, sr={sr}")
    b, a = signal.butter(4, [low, high], btype="bandpass")
    y = signal.filtfilt(b, a, noise).astype(np.float32)
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-12)
    return y


# ============================================================
# Random EQ / frequency distortion
# ============================================================

@dataclass
class RandomEQConfig:
    bands: int = 12
    max_gain_db: float = 8.0
    smoothing_sigma: float = 1.5


def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def smooth_random_gains(num_bands: int, max_gain_db: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    gains = rng.uniform(-max_gain_db, max_gain_db, size=num_bands).astype(np.float32)
    kernel = gaussian_kernel1d(sigma)
    gains = np.convolve(gains, kernel, mode="same")
    return gains.astype(np.float32)


def apply_random_eq(
    audio: np.ndarray,
    sr: int,
    cfg: RandomEQConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(audio)
    X = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    fmin = 20.0
    fmax = sr / 2.0
    if fmax <= fmin:
        raise ValueError("Sample rate too low for random EQ.")

    edges = np.geomspace(fmin, fmax, cfg.bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    gains_db = smooth_random_gains(cfg.bands, cfg.max_gain_db, cfg.smoothing_sigma, rng)
    gains_lin = db_to_linear(gains_db)

    interp_gain = np.ones_like(freqs, dtype=np.float32)
    valid = freqs >= fmin
    interp_gain[valid] = np.interp(
        freqs[valid],
        centers,
        gains_lin,
        left=float(gains_lin[0]),
        right=float(gains_lin[-1]),
    ).astype(np.float32)

    y = np.fft.irfft(X * interp_gain, n=n).astype(np.float32)
    y = y / (rms(y) + 1e-12) * rms(audio)
    return y.astype(np.float32), centers.astype(np.float32), gains_db.astype(np.float32)


# ============================================================
# Corruption modes
# ============================================================

def additive_noise_corruption(
    clean: np.ndarray,
    sr: int,
    mode: str,
    snr_db: float,
    rng: np.random.Generator,
    hum_freq: float = 50.0,
    band_low: float = 300.0,
    band_high: float = 3000.0,
) -> Tuple[np.ndarray, dict]:
    n = len(clean)
    normalized_mode = mode.lower()

    if normalized_mode in {"gaussian", "awgn", "white"}:
        noise = gaussian_noise(n, rng)
        canonical_mode = "gaussian"
    elif normalized_mode == "pink":
        noise = pink_noise(n, rng)
        canonical_mode = "pink"
    elif normalized_mode == "brown":
        noise = brown_noise(n, rng)
        canonical_mode = "brown"
    elif normalized_mode == "hum":
        noise = hum_noise(n, sr, hum_freq, rng)
        canonical_mode = "hum"
    elif normalized_mode == "band":
        noise = band_limited_noise(n, sr, band_low, band_high, rng)
        canonical_mode = "band"
    else:
        raise ValueError(f"Unsupported additive noise mode: {mode}")

    noise = scale_noise_for_snr(clean, noise, snr_db)
    return (clean + noise).astype(np.float32), {
        "mode": canonical_mode,
        "snr_db": snr_db,
        "hum_freq": hum_freq,
        "band_low": band_low,
        "band_high": band_high,
    }


def random_eq_corruption(
    clean: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    bands: int,
    max_gain_db: float,
    smoothing_sigma: float,
) -> Tuple[np.ndarray, dict]:
    cfg = RandomEQConfig(bands=bands, max_gain_db=max_gain_db, smoothing_sigma=smoothing_sigma)
    y, centers, gains_db = apply_random_eq(clean, sr, cfg, rng)
    return y, {
        "mode": "random_eq",
        "bands": int(bands),
        "max_gain_db": float(max_gain_db),
        "smoothing_sigma": float(smoothing_sigma),
        "band_centers_hz": [float(x) for x in centers],
        "gains_db": [float(x) for x in gains_db],
    }


def combo_corruption(
    clean: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    snr_db: float,
    bands: int,
    max_gain_db: float,
    smoothing_sigma: float,
) -> Tuple[np.ndarray, dict]:
    y_eq, meta_eq = random_eq_corruption(
        clean=clean,
        sr=sr,
        rng=rng,
        bands=bands,
        max_gain_db=max_gain_db,
        smoothing_sigma=smoothing_sigma,
    )
    noise = gaussian_noise(len(clean), rng)
    noise = scale_noise_for_snr(y_eq, noise, snr_db)
    y = (y_eq + noise).astype(np.float32)
    return y, {
        "mode": "combo",
        "snr_db": snr_db,
        **meta_eq,
    }


def corrupt_segment(
    clean: np.ndarray,
    sr: int,
    mode: str,
    rng: np.random.Generator,
    snr_db: float,
    hum_freq: float,
    band_low: float,
    band_high: float,
    bands: int,
    max_gain_db: float,
    smoothing_sigma: float,
) -> Tuple[np.ndarray, dict]:
    mode = mode.lower()
    if mode in {"gaussian", "awgn", "white", "pink", "brown", "hum", "band"}:
        return additive_noise_corruption(
            clean=clean,
            sr=sr,
            mode=mode,
            snr_db=snr_db,
            rng=rng,
            hum_freq=hum_freq,
            band_low=band_low,
            band_high=band_high,
        )
    if mode == "random_eq":
        return random_eq_corruption(
            clean=clean,
            sr=sr,
            rng=rng,
            bands=bands,
            max_gain_db=max_gain_db,
            smoothing_sigma=smoothing_sigma,
        )
    if mode == "combo":
        return combo_corruption(
            clean=clean,
            sr=sr,
            rng=rng,
            snr_db=snr_db,
            bands=bands,
            max_gain_db=max_gain_db,
            smoothing_sigma=smoothing_sigma,
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
    p = argparse.ArgumentParser(description="Single-file audio corruption tool with region selection.")
    p.add_argument("input", type=str, help="Input clean audio file")
    p.add_argument("output", type=str, help="Output corrupted audio file")

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gaussian", "awgn", "white", "pink", "brown", "hum", "band", "random_eq", "combo"],
        help="Corruption mode",
    )
    p.add_argument("--sr", type=int, default=None, help="Optional target sample rate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--normalize", action="store_true", help="Peak normalize after corruption")

    # Additive noise controls
    p.add_argument("--snr_db", type=float, default=10.0, help="Target SNR for additive-noise modes")
    p.add_argument("--hum_freq", type=float, default=50.0, help="Base hum frequency in Hz")
    p.add_argument("--band_low", type=float, default=300.0, help="Band-limited noise low cutoff in Hz")
    p.add_argument("--band_high", type=float, default=3000.0, help="Band-limited noise high cutoff in Hz")

    # Random EQ controls
    p.add_argument("--bands", type=int, default=12, help="Number of EQ bands")
    p.add_argument("--max_gain_db", type=float, default=8.0, help="Max absolute gain per EQ band")
    p.add_argument("--smoothing_sigma", type=float, default=1.5, help="Smoothing sigma over band gains")

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
            sr=sr,
            mode=args.mode,
            rng=rng,
            snr_db=args.snr_db,
            hum_freq=args.hum_freq,
            band_low=args.band_low,
            band_high=args.band_high,
            bands=args.bands,
            max_gain_db=args.max_gain_db,
            smoothing_sigma=args.smoothing_sigma,
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
