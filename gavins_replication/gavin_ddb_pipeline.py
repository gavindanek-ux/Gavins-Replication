import hashlib
import json

import numpy as np
import pandas as pd

from pathlib import Path


## Missing-level sentinel handling
# What deeplob_recorder.py writes when a level isn't in the book.
ASK_PRICE_SENTINEL = 2 ** 32  # DeepLOBRecorder.max_price
BID_PRICE_SENTINEL = 0
N_LEVELS = 10

def mask_missing_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Turn missing-level sentinels into NaN.

    2**32 is huge next to a real price, so leaving it in would wreck the
    5-day mean/std and the mid-price. pandas skips NaN in both.
    """
    df = df.copy()
    for l in range(1, N_LEVELS + 1):
        ask_missing = df[f'ASKp{l}'] == ASK_PRICE_SENTINEL
        bid_missing = df[f'BIDp{l}'] == BID_PRICE_SENTINEL
        df.loc[ask_missing, [f'ASKp{l}', f'ASKs{l}']] = np.nan
        df.loc[bid_missing, [f'BIDp{l}', f'BIDs{l}']] = np.nan
    return df


## 5-day rolling z-score normalization
def compute_prev_5day_stats(daily_dfs: list[pd.DataFrame], current_day_idx: int):
    """Mean/std of the 40 features over days d-5..d-1.

    Takes the index instead of a pre-sliced list so the current day can't
    end up in its own normalization stats.
    """
    if current_day_idx < 5:
        raise ValueError(
            f"Need 5 prior days of history to normalize day index "
            f"{current_day_idx}; only {current_day_idx} available.")

    history = daily_dfs[current_day_idx - 5:current_day_idx]
    combined = pd.concat(history, axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    return mean, std

def normalize_day(df_day: pd.DataFrame, mean: pd.Series, std: pd.Series, winsorize_sigma: float = 3.0):
    """5-day z-score, clipped at +/- winsorize_sigma.

    Expects sentinels already masked to NaN. Those stay NaN through the
    z-score, then become 0.0 (the mean) so the array stays dense for torch.
    """
    z_scored = (df_day - mean) / std
    if winsorize_sigma is not None:
        z_scored = z_scored.clip(lower=-winsorize_sigma, upper=winsorize_sigma)
    z_scored = z_scored.fillna(0.0)
    return z_scored


## Equation (4) Smoothed Label Generation

# Marks ticks with no label: the first k-1 and last k of a day, where the
# rolling means run off the edge. Without this they come out "Stationary",
# since NaN is False against both threshold checks.
INVALID_LABEL = -1

def generate_smoothed_labels(df: pd.DataFrame, horizons=[10, 20, 50, 100], alpha=0.0002):
    mid_prices = (df['ASKp1'] + df['BIDp1']) / 2.0
    targets = pd.DataFrame(index=df.index)

    for k in horizons:
        # m_minus: past k ticks including current t
        m_minus = mid_prices.rolling(window=k).mean()

        # m_plus: next k ticks starting from t+1
        m_plus = mid_prices.iloc[::-1].rolling(window=k).mean().iloc[::-1].shift(-1)

        pct_diff = (m_plus - m_minus) / m_minus

        # Label encoding: 0 (Up), 1 (Stationary), 2 (Down), INVALID_LABEL (undefined)
        labels = np.full(len(df), INVALID_LABEL, dtype=np.int64)
        valid = pct_diff.notna().to_numpy()
        pct_diff_np = pct_diff.to_numpy()
        labels[valid & (np.abs(pct_diff_np) <= alpha)] = 1  # Stationary
        labels[valid & (pct_diff_np > alpha)] = 0            # Up
        labels[valid & (pct_diff_np < -alpha)] = 2           # Down

        targets[f'label_{k}'] = labels

    return targets

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

class DeepLOBDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int = 100,
                 invalid_label: int = INVALID_LABEL):
        """
        features: (N_ticks, 40) float32 numpy array
        labels: (N_ticks, num_horizons) int64 numpy array

        Windows whose target tick has no label on some horizon are skipped.
        We filter the sample list, not the rows -- dropping rows would break
        the 100 ticks of contiguity a window needs.
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.window_size = window_size

        n_windows = len(self.features) - window_size + 1
        target_labels = self.labels[window_size - 1: window_size - 1 + n_windows]
        valid_mask = (target_labels != invalid_label).all(dim=1)
        self.valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx].item()
        # Window: (100, 40) -> Unsqueeze to (1, 100, 40) for PyTorch Conv2D
        x = self.features[start : start + self.window_size].unsqueeze(0)
        # Target corresponding to the last tick in the window
        y = self.labels[start + self.window_size - 1]
        return x, y


## End-to-end wiring

# 9:40-15:50, same as data_loader.py. Outside these hours the book is still
# filling out, which is where nearly all the sentinels live: ~0% inside the
# window vs up to ~6% at deep levels outside, over 24 days and 6 tickers.
# This trim, not the masking, is what makes sentinels a non-issue.
TRADING_WINDOW_START = 34_800_000_000_000  # 9:40:00, ns since midnight
TRADING_WINDOW_END = 57_000_000_000_000    # 15:50:00, ns since midnight

def load_daily_lob(path: Path) -> pd.DataFrame:
    """Read one day of recorded LOB, timestamp as the index.

    Takes .parquet or .csv. Parquet is much faster on files this size, so
    convert once with to_parquet_cache and point the pipeline at those.
    """
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col='timestamp')


def to_parquet_cache(csv_paths: list[Path], out_dir: Path) -> list[Path]:
    """Convert recorded CSVs to parquet once, return the new paths.

    Same rows, same index, just a faster format to re-read. Skips files
    already converted so it's safe to re-run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        out_path = out_dir / (csv_path.stem + '.parquet')
        if not out_path.exists():
            pd.read_csv(csv_path, index_col='timestamp').to_parquet(out_path)
        out_paths.append(out_path)
    return out_paths


def restrict_trading_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only rows timestamped within [TRADING_WINDOW_START, TRADING_WINDOW_END]."""
    return df[(df.index >= TRADING_WINDOW_START) & (df.index <= TRADING_WINDOW_END)]


def build_dataset(day_csv_paths: list[Path], window_size: int = 100,
                   horizons=[10, 20, 50, 100], alpha=0.0002,
                   winsorize_sigma=3.0) -> ConcatDataset:
    """Date-ordered LOB CSVs -> training windows.

    Per day: trim hours, mask sentinels, z-score off the previous 5 days,
    label, window. One Dataset per day so windows don't straddle days. The
    first 5 days are dropped, nothing to normalize them against.
    """
    daily_dfs = [
        mask_missing_levels(restrict_trading_window(load_daily_lob(p)))
        for p in day_csv_paths
    ]

    day_datasets = []
    for i in range(5, len(daily_dfs)):
        mean, std = compute_prev_5day_stats(daily_dfs, i)

        day_df = daily_dfs[i]
        normalized = normalize_day(day_df, mean, std, winsorize_sigma)
        labels = generate_smoothed_labels(day_df, horizons=horizons, alpha=alpha)

        day_datasets.append(DeepLOBDataset(
            features=normalized.to_numpy(dtype=np.float32),
            labels=labels.to_numpy(dtype=np.int64),
            window_size=window_size,
        ))

    return ConcatDataset(day_datasets)


## Golden record validation
# Same idea as the toolkit repo's validation/: hash every input file once,
# then re-check before a run. Catches a file that got truncated, re-recorded
# with different settings, or partially synced, which would otherwise show
# up as a weird training result instead of an error.

def compute_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so big files don't blow up memory."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def generate_manifest(paths: list[Path], manifest_path: Path) -> dict:
    """Hash every file in paths and write the manifest as JSON."""
    manifest = {Path(p).name: compute_sha256(p) for p in paths}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def validate_manifest(paths: list[Path], manifest_path: Path) -> list[str]:
    """Check paths against the manifest. Returns problems, empty means clean.

    Returns instead of exiting so it can be called from a run script or a
    notebook without killing the process.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    problems = []
    for p in paths:
        p = Path(p)
        if p.name not in manifest:
            problems.append(f'{p.name}: not in manifest')
        elif not p.exists():
            problems.append(f'{p.name}: missing')
        elif compute_sha256(p) != manifest[p.name]:
            problems.append(f'{p.name}: hash mismatch')

    for name in manifest:
        if name not in {Path(p).name for p in paths}:
            problems.append(f'{name}: in manifest but not passed in')

    return problems