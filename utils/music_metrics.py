"""
Audio quality analysis across corruption types.

Per-clip metrics: Chroma cosine similarity, MFCC distance, PEAQ proxy (NMR).
Distributional metric: Fréchet Audio Distance (VGGish) with bootstrap std/sem.

For each corruption type (sinusoidal, tape_wow_flutter, soft_clip_rms) and variant
(corrupted, reconstructed), results are aggregated over all clips and written to JSON.

Requires: librosa, numpy, scipy, tqdm, frechet_audio_distance
"""

import json
import os
import tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force GPU choice for FAD

import librosa
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
from frechet_audio_distance import FrechetAudioDistance

# -------- Config --------
BASE = "/home/dltp_yang/disk4/joseph/cs5340-grp6/results/20260418_071932_savfk_jtts"
CORRUPTION_TYPES = ["sinusoidal", "tape_wow_flutter", "soft_clip_rms"]
VARIANTS = ["corrupted", "reconstructed"]
NUM_CLIPS = 61  # clip_00 .. clip_60
OUTPUT_JSON = "metrics_report.json"

# FAD bootstrap config
NUM_BOOTSTRAP = 50          # K subsets
BOOTSTRAP_SUBSET_SIZE = 10  # clips per subset (sampled with replacement)
BOOTSTRAP_SEED = 0


# -------- Metric helpers --------
def _load_pair(ref_path, est_path, sr):
    ref, _ = librosa.load(ref_path, sr=sr, mono=True)
    est, _ = librosa.load(est_path, sr=sr, mono=True)
    n = min(len(ref), len(est))
    return ref[:n], est[:n]


def chroma_cos_sim(ref_path, est_path, sr=22050):
    ref, est = _load_pair(ref_path, est_path, sr)
    c_ref = librosa.feature.chroma_cqt(y=ref, sr=sr).mean(axis=1)
    c_est = librosa.feature.chroma_cqt(y=est, sr=sr).mean(axis=1)
    return float(1 - cosine(c_ref, c_est))


def mfcc_dist(ref_path, est_path, sr=22050, n_mfcc=13):
    ref, est = _load_pair(ref_path, est_path, sr)
    m_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    m_est = librosa.feature.mfcc(y=est, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    return float(euclidean(m_ref, m_est))


def peaq_proxy(ref_path, est_path, sr=48000):
    ref, est = _load_pair(ref_path, est_path, sr)
    noise = est - ref
    ref_p = np.mean(ref ** 2) + 1e-10
    noise_p = np.mean(noise ** 2) + 1e-10
    nmr_db = 10 * np.log10(noise_p / ref_p)
    return float(nmr_db)


# -------- Aggregation --------
def aggregate(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sem": float(np.std(arr) / np.sqrt(n)) if n > 0 else None,
        "n": int(n),
    }


# -------- Path builders --------
def clip_paths(corruption, idx):
    s = f"{idx:02d}"
    clean = f"{BASE}/clip_{s}/clean_{s}.wav"
    corrupted = f"{BASE}/clip_{s}/{corruption}/corrupted_{corruption}_{s}.wav"
    reconstructed = f"{BASE}/clip_{s}/{corruption}/reconstructed_v1_1_{corruption}_{s}.wav"
    return clean, {"corrupted": corrupted, "reconstructed": reconstructed}


def combined_dirs(corruption):
    return {
        "clean": f"{BASE}/combined/clean",
        "corrupted": f"{BASE}/combined/{corruption}/corrupted",
        "reconstructed": f"{BASE}/combined/{corruption}/recon",
    }


# -------- Runners --------
def run_per_clip(corruption):
    buckets = {
        v: {"chroma_cos_sim": [], "mfcc_dist": [],
            "peaq_nmr_db": []}
        for v in VARIANTS
    }

    for i in tqdm(range(NUM_CLIPS), desc=f"Per-clip [{corruption}]"):
        clean, targets = clip_paths(corruption, i)
        if not os.path.exists(clean):
            continue
        for variant, path in targets.items():
            if not os.path.exists(path):
                continue
            buckets[variant]["chroma_cos_sim"].append(chroma_cos_sim(clean, path))
            buckets[variant]["mfcc_dist"].append(mfcc_dist(clean, path))
            nmr = peaq_proxy(clean, path)
            buckets[variant]["peaq_nmr_db"].append(nmr)

    return {
        variant: {metric: aggregate(vals) for metric, vals in metrics.items() if vals}
        for variant, metrics in buckets.items()
    }


def _collect_valid_clip_files(corruption):
    """Collect per-clip (clean, variant_paths) for clips where all files exist."""
    clean_files = []
    variant_files = {v: [] for v in VARIANTS}
    for i in range(NUM_CLIPS):
        clean, targets = clip_paths(corruption, i)
        if not os.path.exists(clean):
            continue
        if not all(os.path.exists(p) for p in targets.values()):
            continue
        clean_files.append(clean)
        for v in VARIANTS:
            variant_files[v].append(targets[v])
    return clean_files, variant_files


def _symlink_subset(src_files, indices, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for j, i in enumerate(indices):
        src = src_files[i]
        # prefix with j so duplicates (sampling with replacement) don't collide
        dst = os.path.join(dst_dir, f"s{j:03d}_{os.path.basename(src)}")
        os.symlink(src, dst)


def run_fad(corruption, frechet,
            k=NUM_BOOTSTRAP,
            subset_size=BOOTSTRAP_SUBSET_SIZE,
            seed=BOOTSTRAP_SEED):
    """Bootstrap FAD: sample K subsets of `subset_size` clips (with replacement)
    and compute FAD per subset. Returns aggregated mean/std/sem/n per variant."""
    clean_files, variant_files = _collect_valid_clip_files(corruption)
    n_clips = len(clean_files)
    if n_clips == 0 or n_clips < 1:
        return {v: None for v in VARIANTS}

    rng = np.random.default_rng(seed)
    scores = {v: [] for v in VARIANTS}

    for _ in tqdm(range(k), desc=f"FAD bootstrap [{corruption}]"):
        idx = rng.choice(n_clips, size=subset_size, replace=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_dir = os.path.join(tmpdir, "clean")
            _symlink_subset(clean_files, idx, clean_dir)
            for variant in VARIANTS:
                var_dir = os.path.join(tmpdir, variant)
                _symlink_subset(variant_files[variant], idx, var_dir)
                scores[variant].append(float(frechet.score(clean_dir, var_dir)))

    return {v: aggregate(s) for v, s in scores.items() if s}


# -------- Main --------
def main():
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    report = {"corruption_types": {}}
    for corruption in CORRUPTION_TYPES:
        per_clip = run_per_clip(corruption)
        fad_scores = run_fad(corruption, frechet)
        for variant, score in fad_scores.items():
            per_clip.setdefault(variant, {})["fad"] = score
        report["corruption_types"][corruption] = per_clip

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {OUTPUT_JSON}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()