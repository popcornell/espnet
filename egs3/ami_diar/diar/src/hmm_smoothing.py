"""Inference-only HMM forward-backward for smoothing powerset outputs.

Transition matrix is fitted on the training set (e.g. count bigrams, normalize);
load from file and run forward-backward at inference. No gradients.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def hmm_forward_backward_log(
    log_B: np.ndarray,
    log_A: np.ndarray,
    log_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Log-domain forward-backward; returns state posteriors (T, K).

    Args:
        log_B: (T, K) log emission probabilities.
        log_A: (K, K) log transition matrix, row-stochastic.
        log_init: (K,) log initial state distribution; if None, uniform.

    Returns:
        posteriors: (T, K) state posteriors (sum to 1 per frame).
    """
    T, K = log_B.shape
    if log_init is None:
        log_init = np.log(np.ones(K, dtype=np.float64) / K)

    # Forward
    log_alpha = np.empty((T, K), dtype=np.float64)
    log_alpha[0] = log_init + log_B[0]
    for t in range(1, T):
        # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_A[i, j]) + log_B[t, j]
        log_alpha[t] = (
            _logsumexp(log_alpha[t - 1 : t] + log_A.T, axis=0).ravel() + log_B[t]
        )

    # Backward
    log_beta = np.empty((T, K), dtype=np.float64)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j(log_A[i, j] + log_B[t+1, j] + log_beta[t+1, j])
        log_beta[t] = _logsumexp(
            log_A + log_B[t + 1] + log_beta[t + 1], axis=1
        )

    # Posteriors: gamma[t, j] propto alpha[t, j] * beta[t, j]
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma).astype(np.float32)


def hmm_viterbi_log(
    log_B: np.ndarray,
    log_A: np.ndarray,
    log_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Log-domain Viterbi: most likely state sequence (T,).

    Args:
        log_B: (T, K) log emission probabilities.
        log_A: (K, K) log transition matrix, row-stochastic.
        log_init: (K,) log initial state distribution; if None, uniform.

    Returns:
        path: (T,) integer state indices (best sequence).
    """
    T, K = log_B.shape
    if log_init is None:
        log_init = np.log(np.ones(K, dtype=np.float64) / K)
    log_delta = np.empty((T, K), dtype=np.float64)
    backptr = np.empty((T, K), dtype=np.int64)
    log_delta[0] = log_init + log_B[0]
    for t in range(1, T):
        # log_delta[t, j] = max_i(log_delta[t-1, i] + log_A[i, j]) + log_B[t, j]
        trans = log_delta[t - 1 : t] + log_A.T  # (1, K) + (K,) -> (K, K) after broadcast
        trans = trans.reshape(K, K)
        backptr[t] = np.argmax(trans, axis=0)
        log_delta[t] = np.max(trans, axis=0) + log_B[t]
    path = np.empty(T, dtype=np.int64)
    path[T - 1] = np.argmax(log_delta[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def _logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """Stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    if keepdims:
        out = out + a_max
    else:
        out = out + np.squeeze(a_max, axis=axis)
    return out


def fit_transition_matrix_from_labels(
    labels: np.ndarray,
    lengths: Optional[np.ndarray],
    num_states: int,
    pseudo_count: float = 1.0,
) -> np.ndarray:
    """Compute (K, K) row-stochastic transition matrix from frame-level state indices.

    labels: (batch, T) or (N,) integer state indices (e.g. powerset class).
    lengths: (batch,) valid length per sequence; if None, use full length.
    num_states: K.
    pseudo_count: added to counts before normalizing.
    Returns: (K, K) row-stochastic matrix.
    """
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    B, T = labels.shape
    if lengths is None:
        lengths = np.full(B, T, dtype=np.int64)
    counts = np.zeros((num_states, num_states), dtype=np.float64)
    for b in range(B):
        L = int(lengths[b])
        for t in range(L - 1):
            i, j = int(labels[b, t]), int(labels[b, t + 1])
            if 0 <= i < num_states and 0 <= j < num_states:
                counts[i, j] += 1
    A = (counts + pseudo_count) / (counts.sum(axis=1, keepdims=True) + num_states * pseudo_count)
    return A.astype(np.float32)


def load_hmm_transition_matrix(path: str) -> np.ndarray:
    """Load (K, K) row-stochastic transition matrix from .npy; return log_A."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HMM transition matrix not found: {path}")
    A = np.load(path).astype(np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix (K, K), got shape {A.shape}")
    # Row-stochastic
    A = A / A.sum(axis=1, keepdims=True)
    return np.log(np.clip(A, 1e-9, 1.0))
