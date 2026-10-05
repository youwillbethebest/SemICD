"""Hierarchical k-means (HKM): the default SID quantizer."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits



def _node_seed(seed: int, depth: int, prefix: tuple[int, ...]) -> int:
    payload = f"{seed}|{depth}|{','.join(map(str, prefix))}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _canonicalize_labels(labels: np.ndarray, codes: np.ndarray) -> np.ndarray:
    ordering = sorted(
        np.unique(labels).tolist(),
        key=lambda label: min(codes[labels == label].tolist()),
    )
    remap = {old: new for new, old in enumerate(ordering)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int16)


@dataclass(frozen=True)
class HierarchicalKMeansResult:
    """HKM assignments plus diagnostics for every active hierarchy node."""

    tokens: np.ndarray
    node_diagnostics: pd.DataFrame


def _assigned_inertia(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Compute SSE for already assigned labels, including degenerate nodes."""
    total = 0.0
    for label in np.unique(labels):
        selected = vectors[labels == label].astype(np.float64, copy=False)
        center = selected.mean(axis=0)
        total += float(np.square(selected - center).sum())
    return total


def hierarchical_kmeans(
    embeddings: np.ndarray,
    codes: np.ndarray,
    *,
    depth: int,
    branching_factor: int,
    seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    fit_threads: int = 1,
    strict_cluster_count: bool = True,
) -> np.ndarray:
    """Return HKM tokens using the historical public API."""
    return hierarchical_kmeans_with_diagnostics(
        embeddings,
        codes,
        depth=depth,
        branching_factor=branching_factor,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        fit_threads=fit_threads,
        strict_cluster_count=strict_cluster_count,
    ).tokens


def hierarchical_kmeans_with_diagnostics(
    embeddings: np.ndarray,
    codes: np.ndarray,
    *,
    depth: int,
    branching_factor: int,
    seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    fit_threads: int = 1,
    strict_cluster_count: bool = True,
) -> HierarchicalKMeansResult:
    """Run HKM and retain selected-estimator diagnostics at every active node."""
    if len(embeddings) != len(codes):
        raise ValueError("Embedding/code length mismatch")
    if depth <= 0 or branching_factor <= 0:
        raise ValueError("depth and branching_factor must be positive")
    tokens = np.zeros((len(codes), depth), dtype=np.int16)
    groups: dict[tuple[int, ...], np.ndarray] = {(): np.arange(len(codes))}
    diagnostic_rows: list[dict[str, Any]] = []
    for level in range(depth):
        next_groups: dict[tuple[int, ...], np.ndarray] = {}
        for prefix in sorted(groups):
            indices = groups[prefix]
            node_vectors = embeddings[indices]
            distinct_count = len(np.unique(node_vectors, axis=0))
            clusters = min(branching_factor, len(indices), distinct_count)
            started = time.monotonic()
            estimator_fitted = False
            estimator_inertia: float | None = None
            estimator_n_iter: int | None = None
            if len(indices) == 1:
                labels = np.zeros(1, dtype=np.int16)
            else:
                if clusters <= 1:
                    labels = np.zeros(len(indices), dtype=np.int16)
                else:
                    estimator = KMeans(
                        n_clusters=clusters,
                        init="k-means++",
                        n_init=n_init,
                        max_iter=max_iter,
                        tol=tol,
                        algorithm=algorithm,
                        random_state=_node_seed(seed, level, prefix),
                    )
                    with threadpool_limits(limits=fit_threads):
                        labels = estimator.fit_predict(node_vectors).astype(np.int16)
                    estimator_fitted = True
                    try:
                        raw_inertia = estimator.inertia_
                    except AttributeError:
                        raw_inertia = None
                    try:
                        raw_n_iter = estimator.n_iter_
                    except AttributeError:
                        raw_n_iter = None
                    estimator_inertia = (
                        float(raw_inertia) if raw_inertia is not None else None
                    )
                    estimator_n_iter = (
                        int(raw_n_iter) if raw_n_iter is not None else None
                    )
                    labels = _canonicalize_labels(labels, codes[indices])
                    if strict_cluster_count and len(np.unique(labels)) != clusters:
                        raise RuntimeError(
                            f"K-means returned an empty cluster at level {level + 1}, "
                            f"prefix={prefix}: expected {clusters}, observed "
                            f"{len(np.unique(labels))}"
                        )
            occupied_clusters = int(len(np.unique(labels)))
            inertia = (
                estimator_inertia
                if estimator_inertia is not None
                else _assigned_inertia(node_vectors, labels)
            )
            diagnostic_rows.append(
                {
                    "level": level + 1,
                    "parent_prefix": ".".join(map(str, prefix)),
                    "parent_prefix_tokens": list(prefix),
                    "n_samples": int(len(indices)),
                    "distinct_vector_count": int(distinct_count),
                    "requested_clusters": int(branching_factor),
                    "effective_clusters": int(clusters),
                    "occupied_clusters": occupied_clusters,
                    "inertia": float(inertia),
                    "normalized_inertia": float(inertia / len(indices)),
                    "n_iter": estimator_n_iter,
                    "cap_active": bool(
                        estimator_n_iter is not None and estimator_n_iter == max_iter
                    ),
                    "estimator_fitted": estimator_fitted,
                    "runtime_seconds": float(time.monotonic() - started),
                }
            )
            tokens[indices, level] = labels
            for label in np.unique(labels):
                child = indices[labels == label]
                next_groups[prefix + (int(label),)] = child
        groups = next_groups
    return HierarchicalKMeansResult(
        tokens=tokens,
        node_diagnostics=pd.DataFrame(diagnostic_rows),
    )
