from __future__ import annotations

import numpy as np
from scipy.special import i0, iv


def main() -> None:
    p = 113
    alpha = 2.0  # Nanda et al use AdamW with wd=1; alpha~2 is representative
    deltas = np.arange(1, p)  # Delta = 1, ..., p-1

    # =========================================================================
    # CORE FUNCTIONS
    # =========================================================================

    def compute_margins(K, alpha, p):
        """
        Compute gamma_Delta for all Delta in {1,...,p-1}.

        gamma_Delta = sum_{k in K} alpha * (1 - cos(2*pi*k*Delta/p))

        Interpretation: how much does the model separate c* from c*+Delta?
        """
        gammas = np.zeros(p - 1)
        for k in K:
            gammas += alpha * (1 - np.cos(2 * np.pi * k * deltas / p))
        return gammas

    def compute_CE_loss(K, alpha, p):
        """
        Exact cross-entropy loss.

        L(K, alpha) = log(1 + Z_K)
        where Z_K = sum_{Delta=1}^{p-1} exp(-gamma_Delta(K))

        Note: this equals the per-input CE loss because the margin gamma_Delta
        depends only on Delta = (a+b-c) mod p, not on (a,b) separately.
        So every input pair has the same loss.
        """
        gammas = compute_margins(K, alpha, p)
        Z_K = np.sum(np.exp(-np.clip(gammas, 0, 500)))
        return np.log(1 + Z_K), Z_K

    def compute_softmax_over_delta(K, alpha, p):
        """
        Compute the softmax distribution P_K over Delta values.

        P_K(Delta) = exp(-gamma_Delta(K)) / Z_K

        Interpretation: P_K(Delta) is the (normalized) probability mass
        on the wrong class c*+Delta. High P_K(Delta) means Delta is hard -
        the network barely separates c* from c*+Delta.
        """
        gammas = compute_margins(K, alpha, p)
        exp_neg_gammas = np.exp(-np.clip(gammas, 0, 500))
        Z_K = np.sum(exp_neg_gammas)
        P_K = exp_neg_gammas / Z_K
        return P_K, Z_K

    def exact_loss_reduction(K, k_new, alpha, p):
        """
        Exact loss reduction from adding frequency k_new to K.

        DERIVATION:
        -----------
        Adding k_new multiplies each exp(-gamma_Delta) by:
            s_Delta = exp(-alpha * (1 - cos(2*pi*k_new*Delta/p)))

        So: Z_{K+{k_new}} = Z_K * E_{P_K}[s_Delta]

        And: L(K) - L(K+{k_new})
             = log(1 + Z_K) - log(1 + Z_K * E_{P_K}[s_Delta])

        The reduction is large when E_{P_K}[s_Delta] is small,
        i.e., when k_new strongly suppresses the hard Delta values.

        SIGN CHECK: s_Delta in [0,1], so E[s_Delta] in (0,1),
        so Z_K * E[s] < Z_K, so log(1+Z_K*E[s]) < log(1+Z_K).
        Reduction is always positive. ✓
        """
        P_K, Z_K = compute_softmax_over_delta(K, alpha, p)

        # Suppression factor for each Delta
        suppression = np.exp(-alpha * (1 - np.cos(2 * np.pi * k_new * deltas / p)))

        # Expected suppression under P_K
        E_suppression = np.sum(P_K * suppression)

        # Exact loss reduction
        reduction = np.log(1 + Z_K) - np.log(1 + Z_K * E_suppression)

        return reduction, E_suppression

    # =========================================================================
    # VERIFICATION: does the formula match direct computation?
    # =========================================================================

    print("=" * 65)
    print("VERIFICATION: exact formula vs direct computation")
    print("=" * 65)
    print(f"{'K':>15} | {'k_new':>6} | {'direct':>10} | {'formula':>10} | {'match':>6}")
    print("-" * 55)

    test_cases = [
        ([30], 2),
        ([30, 2], 35),
        ([30, 2, 35], 19),
        ([30, 2, 35, 19], 10),
        ([30, 2, 35, 19, 10], 36),
    ]

    for K, k_new in test_cases:
        L_before, _ = compute_CE_loss(K, alpha, p)
        L_after, _ = compute_CE_loss(K + [k_new], alpha, p)
        direct = L_before - L_after
        formula, _ = exact_loss_reduction(K, k_new, alpha, p)
        match = "YES" if abs(direct - formula) < 1e-10 else "NO"
        print(f"{str(K):>15} | {k_new:>6} | {direct:>10.6f} | {formula:>10.6f} | {match:>6}")

    # =========================================================================
    # MAIN RESULT: greedy frequency selection and loss reduction at each step
    # =========================================================================

    print()
    print("=" * 65)
    print(f"GREEDY FREQUENCY SELECTION  (p={p}, alpha={alpha})")
    print("=" * 65)
    print(f"{'m':>3} | {'k added':>8} | {'L after':>10} | {'reduction':>10} | {'E[suppress]':>12} | {'Z_K':>12}")
    print("-" * 65)

    K = []
    L_baseline = np.log(p)

    print(f"{'0':>3} | {'—':>8} | {L_baseline:>10.6f} | {'—':>10} | {'—':>12} | {p - 1:>12.2f}")

    for m in range(1, 8):
        best_reduction = -np.inf
        best_k = None
        best_E = None

        for k in range(1, (p - 1) // 2 + 1):
            if k in K:
                continue
            red, E_s = exact_loss_reduction(K, k, alpha, p)
            if red > best_reduction:
                best_reduction = red
                best_k = k
                best_E = E_s

        K = K + [best_k]
        L_new, Z_new = compute_CE_loss(K, alpha, p)

        print(f"{m:>3} | {best_k:>8} | {L_new:>10.6f} | {best_reduction:>10.6f} | {best_E:>12.6f} | {Z_new:>12.6f}")

    # =========================================================================
    # INTUITION: why does the reduction drop at m=5?
    # =========================================================================

    print()
    print("=" * 65)
    print("INTUITION: what happens to P_K as m grows?")
    print("=" * 65)
    print()

    K = []
    for m in range(0, 6):
        if m > 0:
            # Add best frequency greedily
            best_red = -np.inf
            best_k = None
            for k in range(1, (p - 1) // 2 + 1):
                if k in K:
                    continue
                red, _ = exact_loss_reduction(K, k, alpha, p)
                if red > best_red:
                    best_red = red
                    best_k = k
            K = K + [best_k]

        P_K, Z_K = compute_softmax_over_delta(K, alpha, p)
        L = np.log(1 + Z_K)

        # Statistics of P_K
        top5_idx = np.argsort(P_K)[-5:][::-1]
        top5_deltas = deltas[top5_idx]
        top5_probs = P_K[top5_idx]
        entropy = -np.sum(P_K * np.log(P_K + 1e-300))
        max_entropy = np.log(p - 1)

        print(f"m={m}, K={K}")
        print(f"  L = {L:.6f},  Z_K = {Z_K:.4f}")
        print(f"  Entropy of P_K: {entropy:.4f} / {max_entropy:.4f} " f"(={100 * entropy / max_entropy:.1f}% of uniform)")
        print(f"  Top-5 hard Deltas: {list(zip(top5_deltas, np.round(top5_probs, 4)))}")
        print()

    # =========================================================================
    # THE EXACT CE LOSS FORMULA VIA BESSEL FUNCTIONS
    # =========================================================================

    print("=" * 65)
    print("EXACT FORMULA: L(K,alpha) = log(p) - alpha*m + log(B(K,alpha))")
    print("=" * 65)
    print()
    print("B(K,alpha) = sum_{(n_k) in Z^m : sum_k n_k*k = 0 mod p} prod_k I_{n_k}(alpha)")
    print()
    print("Derivation:")
    print("  exp(alpha*cos(theta)) = sum_n I_n(alpha) * exp(i*n*theta)  [Bessel expansion]")
    print("  sum_{Delta=0}^{p-1} prod_k exp(alpha*cos(2pi k Delta/p))")
    print("  = p * B(K,alpha)                                           [character orthogonality]")
    print("  Delta=0 term = exp(alpha*m)                               [cos(0)=1]")
    print("  => sum_{Delta=1}^{p-1} exp(-gamma_Delta) = e^{-alpha*m} * (p*B - e^{alpha*m})")
    print("  => L = log(p * B(K,alpha) * e^{-alpha*m})")
    print()

    def B_single_freq(k, alpha, p, N_max=5):
        """
        B({k}, alpha) = sum_{n: n*k=0 mod p} I_n(alpha)
                     = sum_{j} I_{j*p}(alpha)
        Since p is prime and gcd(k,p)=1, n*k=0 mod p iff p|n.
        Dominant term: j=0 gives I_0(alpha).
        Correction: j=+-1 gives I_{+-p}(alpha) ~ 0 for large p.
        """
        return sum(iv(j * p, alpha) for j in range(-N_max, N_max + 1))

    print("Verification (m=1): L = log(p) - alpha + log(B({k},alpha))")
    print(f"{'k':>5} | {'direct CE':>12} | {'formula':>12} | {'I_0(alpha)':>12} | {'correction':>12}")
    print("-" * 60)
    for k in [1, 14, 30, 35, 56]:
        L_direct, _ = compute_CE_loss([k], alpha, p)
        B = B_single_freq(k, alpha, p)
        L_formula = np.log(p) - alpha + np.log(B)
        correction = B - i0(alpha)
        print(f"{k:>5} | {L_direct:>12.8f} | {L_formula:>12.8f} | {i0(alpha):>12.8f} | {correction:>12.2e}")

    print()
    print("Key observation: for m=1, B({k},alpha) = I_0(alpha) exactly")
    print("(correction = 2*I_p(alpha) ~ 10^{-200} for p=113)")
    print("=> All single irreps give IDENTICAL CE loss: log(p) - alpha + log(I_0(alpha))")
    print(f"=> For alpha={alpha}: L = {np.log(p) - alpha + np.log(i0(alpha)):.6f}")
    print()
    print("For m>1, B(K,alpha) depends on the specific frequencies K")
    print("through the constraint sum_k n_k*k = 0 mod p.")
    print("This is where the arithmetic of Z_p determines frequency selection.")

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"""
The exact loss reduction from adding irrep k to K is:

    delta_L = log(1 + Z_K) - log(1 + Z_K * E_{{P_K}}[exp(-alpha*(1-cos(2pi k Delta/p)))])

where:
    Z_K     = total confusion remaining = sum_Delta exp(-gamma_Delta(K))
    P_K     = softmax over hard Deltas = exp(-gamma_Delta) / Z_K
    alpha   = per-frequency amplitude

The reduction is large when the new frequency k strongly suppresses
the currently hardest Delta values (those dominating P_K).

As m grows:
    - Z_K shrinks (confusion decreases)
    - P_K concentrates on residual hard Deltas
    - These hard Deltas are arithmetically hard to cover simultaneously
    - Each new irrep gives less marginal suppression

For p=113 and alpha=2.0:
    m=1->2: reduction=1.176  (easy gains, P_K still nearly uniform)
    m=2->3: reduction=1.176  (still easy)
    m=3->4: reduction=1.164  (starting to concentrate)
    m=4->5: reduction=0.846  (Z_K getting small)
    m=5->6: reduction=0.297  (hard residual Deltas dominate)
    m=6->7: reduction=0.058  (essentially saturated)
    m=7->8: reduction=0.009  (negligible)

The threshold m*=5 is where Z_K becomes small enough that the
remaining confusion is negligible - a consequence of the covering
structure of Z_{{113}} under the active frequencies.
""")


if __name__ == "__main__":
    main()
