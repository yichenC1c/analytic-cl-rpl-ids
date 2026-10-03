"""Random Fourier features and the recursive least-squares analytic head.

Everything here runs in float64 on the CPU. The MPS backend does not support float64,
and the equivalence check in check_equivalence.py is meaningless at single precision:
48 successive Woodbury updates accumulate enough rounding error to swamp the signal.
"""
import numpy as np


class RFF:
    """z(x) = sqrt(2/D) cos(x W + b), so that z(x).z(y) approximates exp(-gamma ||x-y||^2).

    Two properties distinguish this from the buffer layers used elsewhere in the analytic
    continual-learning literature. It carries a uniform approximation bound (Rahimi and
    Recht, 2007), and W and b are generated from a seed, so the deployed map stores no
    data. Gaussian-kernel embeddings whose centres are drawn from the training set must
    retain those centres permanently.
    """

    def __init__(self, d_in, D, gamma, seed=0):
        rng = np.random.default_rng(seed)
        self.D, self.gamma = D, gamma
        self.W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d_in, D))
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=D)
        self.scale = np.sqrt(2.0 / D)

    def __call__(self, X):
        return self.scale * np.cos(X.astype(np.float64) @ self.W + self.b)

    @property
    def n_bytes(self):
        return self.W.nbytes + self.b.nbytes


def median_gamma(X, n=2000, seed=0):
    """Median heuristic: gamma = 1 / (2 * median pairwise distance squared)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    S = X[idx].astype(np.float64)
    d2 = np.maximum((S ** 2).sum(1)[:, None] + (S ** 2).sum(1)[None] - 2 * S @ S.T, 0.0)
    med = np.median(np.sqrt(d2[np.triu_indices(len(S), 1)]))
    return 1.0 / (2.0 * med ** 2) if med > 0 else 1.0


class RLSHead:
    """Maintains R = (sum Z^T Z + ridge I)^-1 and W, updated once per domain.

    Under domain-incremental learning the label space is fixed, so W keeps a constant
    shape. The column concatenation and residual-cleansing steps that class-incremental
    analytic methods require have no counterpart here, and the recursion reduces to plain
    block RLS.
    """

    def __init__(self, D, n_out=2, ridge=1e-2, block=64):
        # The block size must not exceed D. Woodbury inverts (I + Z R Z^T), which is
        # block x block; once block > D that matrix is the identity plus a correction of
        # rank at most D, and a general inverse of it is both slower and ill-conditioned.
        # Measured relative error against the joint solution at D=256 (see
        # test_analytic.py): 2.6e-12 at block 16, 7.3e-11 at 64, 7.6e-09 at 256,
        # 2.4e-07 at 512. Small blocks are strictly better.
        self.D, self.n_out, self.ridge = D, n_out, ridge
        self.block = max(1, min(block, D))
        self.R = np.eye(D, dtype=np.float64) / ridge
        self.W = np.zeros((D, n_out), dtype=np.float64)

    def partial_fit(self, Z, Y):
        """Absorb one domain. Blocking affects peak memory only; RLS is exact for any
        partition of the incoming rows."""
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        for s in range(0, len(Z), self.block):
            Zb, Yb = Z[s:s + self.block], Y[s:s + self.block]
            RZt = self.R @ Zb.T
            S = np.eye(len(Zb)) + Zb @ RZt
            K = RZt @ np.linalg.inv(S)              # equals R_new Zb^T
            self.W += K @ (Yb - Zb @ self.W)
            self.R -= K @ (Zb @ self.R)
            self.R = 0.5 * (self.R + self.R.T)      # suppress drift from symmetry
        return self

    def predict_scores(self, Z):
        return np.asarray(Z, dtype=np.float64) @ self.W

    @property
    def state_bytes(self):
        """Resident state required to keep adapting: R and W."""
        return self.R.nbytes + self.W.nbytes

    @property
    def inference_bytes(self):
        """Resident state required to classify only. R is needed for further updates,
        not for prediction, so the two figures differ by roughly D/2."""
        return self.W.nbytes


class JointHead:
    """Joint-batch reference solution. Accumulates G and A, then solves once.

    Used by probe_upper_bound.py as a joint-fitting diagnostic and by
    check_equivalence.py as the quantity the recursion is compared against.
    """

    def __init__(self, D, n_out=2, ridge=1e-2):
        self.D, self.ridge = D, ridge
        self.G = np.zeros((D, D), dtype=np.float64)
        self.A = np.zeros((D, n_out), dtype=np.float64)

    def accumulate(self, Z, Y):
        Z = np.asarray(Z, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.G += Z.T @ Z
        self.A += Z.T @ Y
        return self

    def solve(self):
        return np.linalg.solve(self.G + self.ridge * np.eye(self.D), self.A)


class ExactKELM:
    """Exact kernel head refitted from scratch on all accumulated data at every domain.

    Memory is O(N^2) in the number of accumulated samples, which is what makes this
    infeasible past a handful of domains on a fixed budget.
    """

    def __init__(self, gamma, ridge=1e-2, max_n=35000):
        self.gamma, self.ridge, self.max_n = gamma, ridge, max_n
        self.Xs, self.Ys = [], []
        self.Xtr = None
        self.alpha = None

    def _k(self, A, B):
        d2 = np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0.0)
        return np.exp(-self.gamma * d2)

    def add_and_refit(self, X, Y):
        """Return (succeeded, cumulative_n). On failure the state is left untouched so the
        previous solution stays usable; the budget overrun is reported rather than worked
        around by subsampling."""
        N = sum(len(x) for x in self.Xs) + len(X)
        if N > self.max_n:
            return False, N
        self.Xs.append(np.asarray(X, np.float64))
        self.Ys.append(np.asarray(Y, np.float64))
        Xa = np.concatenate(self.Xs)
        Ya = np.concatenate(self.Ys)
        K = self._k(Xa, Xa)
        K[np.diag_indices_from(K)] += self.ridge
        self.alpha = np.linalg.solve(K, Ya)
        self.Xtr = Xa
        return True, N

    def predict_scores(self, X, chunk=512):
        """Chunked so the n_test x N kernel matrix is never materialised in full."""
        if self.Xtr is None:
            raise RuntimeError("no successful fit yet")
        X = np.asarray(X, np.float64)
        out = np.empty((len(X), self.alpha.shape[1]))
        for s in range(0, len(X), chunk):
            out[s:s + chunk] = self._k(X[s:s + chunk], self.Xtr) @ self.alpha
        return out

    @property
    def state_bytes(self):
        """All support vectors plus coefficients must stay resident, so this grows
        linearly in N."""
        return self.Xtr.nbytes + self.alpha.nbytes

    @property
    def kernel_evals_per_inference(self):
        """N kernel evaluations per prediction, against D cosines for the analytic head."""
        return len(self.Xtr)
