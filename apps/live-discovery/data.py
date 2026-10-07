"""Reproducible presets and bounded numeric CSV uploads."""

from __future__ import annotations

import csv
import hashlib
import secrets
from pathlib import Path

import numpy as np

from andrey.data import SCM, benchmarks, functional, graphs, noise
from andrey.data.latent import marginal
from andrey.metrics import to_cpdag

MAX_BYTES = 5_000_000
MAX_COLUMNS = 50
MAX_ROWS = 20_000
FAMILIES = {
    "PC": "linear_gauss_er",
    "GES": "linear_gauss_er",
    "FCI": "latent_confounded_er",
    "DirectLiNGAM": "lingam_sf",
}
# Explore shows one clear recovery per method: graphs sparse enough, with enough observations, that
# the estimate beats the empty graph by a wide margin on most seeds.
EXPLORE = {
    "PC": (
        "Linear-Gaussian data from a sparse random graph",
        SCM(
            graph=graphs.erdos_renyi(10, 2.0),
            functional=functional.linear(),
            noise=noise.gaussian(),
        ),
        2_000,
    ),
    "FCI": (
        "Linear-Gaussian data from a sparse random graph with one hidden variable",
        SCM(
            graph=graphs.erdos_renyi(11, 1.5),
            functional=functional.linear(),
            noise=noise.gaussian(),
            latents=1,
        ),
        5_000,
    ),
    "DirectLiNGAM": (
        "Linear non-Gaussian data from a scale-free graph",
        benchmarks.scm("lingam_sf", 10),
        1_000,
    ),
}
EXPLORE["GES"] = EXPLORE["PC"]


def integer(value, name, low, high):
    """Validate integer API inputs without silently truncating them."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or int(value) != value
    ):
        raise ValueError(f"{name} must be a whole number.")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return value


def new_seed():
    """Draw a fresh displayed seed without global RNG state."""
    return secrets.randbelow(2**32)


def preset(method, size, seed, cap):
    """Return the same observations and truth for a family, size, and seed."""
    if method not in FAMILIES:
        raise ValueError("Choose a listed method.")
    size = integer(size, "Variables", 50 if method == "FCI" else 5, cap)
    seed = integer(seed, "Seed", 0, 2**32 - 1)
    # The latent family counts hidden nodes in d; the slider counts observed variables.
    total = size
    if method == "FCI":
        while total - max(1, total // 10) < size:
            total += 1
    sample = benchmarks.scm(FAMILIES[method], total).sample(
        n=10 * size, seed=seed, scale="standardize"
    )
    truth = marginal(sample.graph, "pag", max_nodes=250) if method == "FCI" else sample.graph
    return sample.data, truth, seed


def explore_preset(method, seed):
    """Return an Explore preset's observations, truth, description, and validated seed."""
    if method not in EXPLORE:
        raise ValueError("Choose a listed method.")
    seed = integer(seed, "Seed", 0, 2**32 - 1)
    source, scm, n = EXPLORE[method]
    sample = scm.sample(n=n, seed=seed, scale="standardize")
    description = f"{source}, {n:,} observations."
    truth = sample.graph
    if method == "FCI":
        truth = marginal(truth, "pag", max_nodes=250)
    elif method in ("PC", "GES"):
        # SHD scores the equivalence class, the most observational data can identify.
        truth = to_cpdag(truth)
        description += (
            " The truth is drawn as its CPDAG; the data cannot orient its undirected edges."
        )
    return sample.data, truth, description, seed


def fingerprint(data):
    """Identify the exact observations shared by all race participants."""
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()[:12]


def read_csv(path):
    """Read a headered CSV only after checking byte, column, and row limits."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Upload a .csv file with a header row.")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("CSV files must be at most 5 MB.")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            labels = tuple(next(reader))
            if not 3 <= len(labels) <= MAX_COLUMNS:
                raise ValueError("CSV files need 3 to 50 columns.")
            if any(not label.strip() or len(label) > 80 for label in labels):
                raise ValueError("Use nonempty column names of at most 80 characters.")
            if len(set(labels)) != len(labels):
                raise ValueError("Column names must be unique.")
            rows = []
            for row in reader:
                if len(rows) >= MAX_ROWS:
                    raise ValueError("CSV files must have at most 20,000 data rows.")
                if len(row) != len(labels):
                    raise ValueError("Every row must have the same number of columns.")
                try:
                    rows.append([float(value) for value in row])
                except ValueError as error:
                    raise ValueError("Every data cell must be numeric.") from error
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise ValueError("Upload a UTF-8 numeric CSV with a header row.") from error
    data = np.asarray(rows, dtype=np.float64)
    if len(rows) < len(labels) + 3:
        raise ValueError("Use at least three more data rows than columns.")
    if not np.isfinite(data).all():
        raise ValueError("Remove missing values, NaN, and infinity.")
    if np.any(np.ptp(data, axis=0) == 0):
        raise ValueError("Remove constant columns.")
    correlations = np.abs(np.corrcoef(data, rowvar=False))
    np.fill_diagonal(correlations, 0)
    if np.any(correlations >= 1 - 1e-12):
        raise ValueError("Remove duplicate or perfectly correlated columns.")
    return data, labels
