"""
File: utils/regions.py
Description: Logic for applying differentiable corruptions to specific time segments 
of an audio file with crossfading, region selection, and metadata logging.
"""

import csv
import torch
from pathlib import Path
from typing import Callable, Sequence, Tuple, List, Optional, Dict, Any

def parse_ranges(ranges_text: str, sr: int, total_samples: int) -> List[Tuple[int, int]]:
    result = []
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
    merged = [intervals[0]]
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
    num_segments: int,
    segment_duration: Optional[float],
    min_segment_duration: float,
    max_segment_duration: float,
    allow_overlap: bool,
) -> List[Tuple[int, int]]:
    if num_segments <= 0:
        return []

    intervals = []
    attempts = 0
    max_attempts = max(100, num_segments * 40)

    while len(intervals) < num_segments and attempts < max_attempts:
        attempts += 1
        
        if segment_duration is not None:
            dur_s = segment_duration
        else:
            # Random uniform duration
            dur_s = float(torch.rand(1).item() * (max_segment_duration - min_segment_duration) + min_segment_duration)
            
        seg_len = max(1, int(round(dur_s * sr)))
        
        if seg_len >= total_samples:
            candidate = (0, total_samples)
        else:
            start = int(torch.randint(0, total_samples - seg_len + 1, (1,)).item())
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
    ranges_text: str = "",
    num_segments: int = 3,
    segment_duration: Optional[float] = None,
    min_segment_duration: float = 0.25,
    max_segment_duration: float = 1.0,
    allow_overlap: bool = False,
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
            num_segments=num_segments,
            segment_duration=segment_duration,
            min_segment_duration=min_segment_duration,
            max_segment_duration=max_segment_duration,
            allow_overlap=allow_overlap,
        )
        
    raise ValueError(f"Unsupported region_mode: {region_mode}")

def apply_corruption_to_regions(
    clean: torch.Tensor,
    sr: int,
    regions: Sequence[Tuple[int, int]],
    corrupt_fn: Callable[[torch.Tensor], torch.Tensor],
    fade_ms: float = 10.0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Applies a corruption function to targeted regions and blends them using a linear crossfade.
    Returns the corrupted tensor alongside standard metadata.
    """
    out = clean.clone()
    fade_samples = max(0, int(round(fade_ms * sr / 1000.0)))
    region_metadata = []
    
    for idx, (start, end) in enumerate(regions):
        if end <= start:
            continue
            
        segment = clean[..., start:end]
        corrupted_segment = corrupt_fn(segment)
        
        # Build the crossfade blend mask
        blend = torch.ones(segment.shape[-1], device=clean.device, dtype=clean.dtype)
        f = min(fade_samples, segment.shape[-1] // 2)
        
        if f > 0:
            ramp = torch.linspace(0.0, 1.0, steps=f, device=clean.device, dtype=clean.dtype)
            blend[:f] = ramp
            blend[-f:] = torch.flip(ramp, dims=[0])
            
        # Ensure blend shape matches segment for broadcasting
        blend = blend.expand_as(segment)
        out[..., start:end] = (1.0 - blend) * clean[..., start:end] + blend * corrupted_segment
        
        region_metadata.append({
            "region_index": idx,
            "start_sample": int(start),
            "end_sample": int(end),
            "start_time_sec": float(start / sr),
            "end_time_sec": float(end / sr),
            "duration_sec": float((end - start) / sr),
        })
        
    return out, region_metadata

def write_metadata_txt(path: str, metadata: dict) -> None:
    lines = []
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