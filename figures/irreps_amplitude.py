import numpy as np
import pandas as pd


def required_loss(gamma: np.ndarray) -> float:
    """
    Cross-entropy loss from margins gamma(Δ)
    """
    return float(np.log1p(np.sum(np.exp(-gamma))))


def precompute_basis(p: int) -> np.ndarray:
    """
    B[k-1, Δ-1] = 1 - cos(2π k Δ / p)
    """
    ks = np.arange(1, (p - 1) // 2 + 1)[:, None]
    deltas = np.arange(1, p)[None, :]
    return 1.0 - np.cos(2.0 * np.pi * ks * deltas / p)


def minimal_irreps_for_amplitude(p: int, A: float, delta: float, max_m: int = 20):
    """
    For fixed amplitude A, find minimal m such that loss <= delta
    using greedy optimal irreps
    """
    B = precompute_basis(p)
    n_irreps = B.shape[0]

    available = np.ones(n_irreps, dtype=bool)
    selected = []
    running = np.zeros(p - 1)

    for m in range(1, min(max_m, n_irreps) + 1):

        candidates = np.where(available)[0]

        # Try adding each candidate irrep
        best_idx = None
        best_loss = float("inf")

        for idx in candidates:
            trial = running + B[idx]
            gamma = (A / m) * trial
            loss = required_loss(gamma)

            if loss < best_loss:
                best_loss = loss
                best_idx = idx

        # Update
        selected.append(best_idx)
        available[best_idx] = False
        running += B[best_idx]

        # Check condition
        gamma = (A / m) * running
        loss = required_loss(gamma)

        if loss <= delta:
            return m, loss

    return None, None


def sweep_amplitudes(p: int, delta: float, A_max: int = 20):
    """
    Compute minimal m for A = 1,...,A_max
    """
    rows = []

    for A in range(1, A_max + 1):
        m, loss = minimal_irreps_for_amplitude(p, A, delta)

        rows.append({
            "p": p,
            "A": A,
            "m_star": m,
            "loss": loss
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    primes = [97, 113, 197]
    delta = 0.05
    A_max = 20

    for p in primes:
        df = sweep_amplitudes(p, delta, A_max)
        print(f"\np = {p}")
        print(df.to_string(index=False))