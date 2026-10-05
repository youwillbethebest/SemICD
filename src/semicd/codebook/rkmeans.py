"""Residual k-means (RK-Means): an alternative SID quantizer."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits



def _layer_seed(seed: int, layer: int) -> int:
    payload = f"residual-kmeans|{seed}|{layer}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _canonicalize_codebook(
    labels: np.ndarray,
    codes: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ordering = sorted(
        np.unique(labels).tolist(),
        key=lambda label: min(codes[labels == label].tolist()),
    )
    remap = {int(old): new for new, old in enumerate(ordering)}
    canonical_labels = np.asarray(
        [remap[int(label)] for label in labels], dtype=np.int16
    )
    canonical_centroids = centroids[np.asarray(ordering, dtype=np.int64)]
    return canonical_labels, canonical_centroids.astype(np.float32, copy=False)


def residual_kmeans(
    embeddings: np.ndarray,
    codes: np.ndarray,
    *,
    depth: int,
    codebook_size: int,
    seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    fit_threads: int = 1,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[dict[str, float | int]]]:
    """Quantize embeddings by repeatedly clustering and subtracting residuals."""
    if len(embeddings) != len(codes):
        raise ValueError("Embedding/code length mismatch")
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be a non-empty matrix")
    residual = embeddings.astype(np.float32, copy=True)
    tokens = np.zeros((len(codes), depth), dtype=np.int16)
    codebooks: list[np.ndarray] = []
    stats: list[dict[str, float | int]] = []
    original_energy = float(np.square(embeddings.astype(np.float64)).sum())

    for layer in range(depth):
        energy_before = float(np.square(residual.astype(np.float64)).sum())
        distinct_count = len(np.unique(residual, axis=0))
        clusters = min(codebook_size, len(residual), distinct_count)
        if clusters <= 1:
            labels = np.zeros(len(residual), dtype=np.int16)
            centroids = residual.mean(axis=0, keepdims=True).astype(np.float32)
        else:
            estimator = KMeans(
                n_clusters=clusters,
                init="k-means++",
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                algorithm=algorithm,
                random_state=_layer_seed(seed, layer),
            )
            with threadpool_limits(limits=fit_threads):
                raw_labels = estimator.fit_predict(residual).astype(np.int16)
            if len(np.unique(raw_labels)) != clusters:
                raise RuntimeError(
                    f"Residual K-means returned an empty cluster at layer {layer + 1}: "
                    f"expected {clusters}, observed {len(np.unique(raw_labels))}"
                )
            labels, centroids = _canonicalize_codebook(
                raw_labels, codes, estimator.cluster_centers_
            )

        tokens[:, layer] = labels
        assigned = centroids[labels]
        residual = (residual - assigned).astype(np.float32, copy=False)
        energy_after = float(np.square(residual.astype(np.float64)).sum())
        if energy_after > energy_before + max(1e-10, energy_before * 1e-8):
            raise RuntimeError(
                f"Residual energy increased at layer {layer + 1}: "
                f"{energy_before} -> {energy_after}"
            )
        codebooks.append(centroids)
        stats.append(
            {
                "layer": layer + 1,
                "clusters": clusters,
                "mean_residual_l2_before": float(
                    np.linalg.norm(residual + assigned, axis=1).mean()
                ),
                "mean_residual_l2_after": float(np.linalg.norm(residual, axis=1).mean()),
                "residual_energy_fraction": float(
                    energy_after / original_energy if original_energy else 0.0
                ),
                "quantization_mse": float(energy_after / embeddings.size),
            }
        )
    return tokens, codebooks, residual, stats
