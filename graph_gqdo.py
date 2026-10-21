"""Disease Relationship Graph (DRG) and the Graph-Guided Quantum Dragonfly Optimizer (GQDO).

Variants used in the ablation:
  'none' : all features kept
  'da'   : classical binary Dragonfly Algorithm (sigmoid transfer, no quantum encoding, no graph term)
  'qda'  : quantum amplitude encoding + Levy flight, no graph term
  'gqdo' : quantum encoding + Levy flight + graph-topology preservation penalty (proposed)
"""
import numpy as np
from scipy.special import gamma as Gamma
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score


# ----------------------------------------------------------------------------- Disease Relationship Graph
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def envelope_modulation(seq_mel):
    """seq_mel: (N, T, 128) log-mel frames -> (N, 16) envelope-modulation magnitude spectrum."""
    env = seq_mel.mean(2)                          # (N, T) frame energy envelope
    env = env - env.mean(1, keepdims=True)
    M = np.abs(np.fft.rfft(env, axis=1))[:, 1:17]  # modulation bins 0.5-8 Hz at 50 frames/s
    return M / (M.sum(1, keepdims=True) + 1e-12)


def disease_graph(X, y, C, mel_profile=None, mod_spec=None, w=(0.5, 0.3, 0.2), topk=3):
    """X: standardized fused features (N, D); y: int labels; returns row-normalized adjacency A (C, C)
    plus the raw weighted similarity S."""
    S = np.zeros((C, C))
    cent = np.stack([X[y == c].mean(0) for c in range(C)])
    spec = None if mel_profile is None else np.stack([mel_profile[y == c].mean(0) for c in range(C)])
    temp = None if mod_spec is None else np.stack([mod_spec[y == c].mean(0) for c in range(C)])
    for i in range(C):
        for j in range(C):
            if i == j: continue
            s = w[0] * _cos(cent[i], cent[j])
            if spec is not None: s += w[1] * _cos(spec[i] - spec.mean(0), spec[j] - spec.mean(0))
            if temp is not None: s += w[2] * _cos(temp[i], temp[j])
            S[i, j] = s
    S = (S + S.T) / 2; np.fill_diagonal(S, 0)
    A = np.zeros_like(S)
    for i in range(C):
        idx = np.argsort(-S[i])[:topk]
        A[i, idx] = np.maximum(S[i, idx], 1e-6)
    A = A / (A.sum(1, keepdims=True) + 1e-12)
    return A, S


def graph_penalty(X, y, C, mask, A_ref):
    if mask.sum() < 2: return 1.0
    A_b, _ = disease_graph(X[:, mask], y, C)
    return float(np.linalg.norm(A_b - A_ref) / (np.linalg.norm(A_ref) + 1e-12))


# ----------------------------------------------------------------------------- GQDO
def levy(D, beta=1.5, rng=None):
    rng = rng or np.random
    sigma = (Gamma(1 + beta) * np.sin(np.pi * beta / 2) / (Gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, D); v = rng.normal(0, 1, D)
    return 0.01 * u / (np.abs(v) ** (1 / beta))


class GQDO:
    def __init__(self, variant="gqdo", pop=30, iters=50, lam=(0.7, 0.1, 0.2), beta=1.5, seed=0,
                 surrogate_max=3000, verbose=True):
        self.variant, self.P, self.T, self.lam, self.beta = variant, pop, iters, lam, beta
        self.rng = np.random.RandomState(seed); self.surrogate_max = surrogate_max; self.verbose = verbose
        self.history = []

    def _surrogate(self, Xtr, ytr, Xva, yva, mask):
        clf = RidgeClassifier(alpha=10.0, class_weight="balanced")   # closed-form linear surrogate (fast wrapper evaluation)
        clf.fit(Xtr[:, mask], ytr)
        return balanced_accuracy_score(yva, clf.predict(Xva[:, mask]))

    def fitness(self, mask):
        if mask.sum() == 0: return 10.0
        acc = self._surrogate(self.Xtr, self.ytr, self.Xva, self.yva, mask)
        f = self.lam[0] * (1 - acc) + self.lam[1] * mask.mean()
        if self.variant == "gqdo":
            f += self.lam[2] * graph_penalty(self.Xtr, self.ytr, self.C, mask, self.A_ref)
        return f

    def run(self, Xtr, ytr, Xva, yva, C, A_ref=None):
        rng = self.rng
        if len(Xtr) > self.surrogate_max:
            idx = rng.choice(len(Xtr), self.surrogate_max, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
        if len(Xva) > self.surrogate_max // 2:
            idx = rng.choice(len(Xva), self.surrogate_max // 2, replace=False); Xva, yva = Xva[idx], yva[idx]
        self.Xtr, self.ytr, self.Xva, self.yva, self.C, self.A_ref = Xtr, ytr, Xva, yva, C, A_ref
        D = Xtr.shape[1]; P, T = self.P, self.T
        quantum = self.variant in ("qda", "gqdo")
        if quantum:
            q = np.full((P, D), 1 / np.sqrt(2))        # equal superposition
        else:
            q = rng.uniform(-1, 1, (P, D))              # continuous positions (classical binary DA)
        dq = np.zeros((P, D))
        best_f, best_b, best_q = np.inf, None, None
        worst_q = q[0].copy()
        for t in range(T):
            w = 0.9 - t * (0.9 - 0.4) / T
            s, a, c = 0.1 * (1 - t / T), 0.1 * (1 - t / T), 0.7 * (1 - t / T) + 0.1
            f_, e_ = 2 * rng.rand() * (t / T + 0.1), 0.1 * (1 - t / T)
            fits = np.zeros(P); masks = []
            for i in range(P):
                if quantum: b = rng.rand(D) < q[i] ** 2
                else: b = rng.rand(D) < 1 / (1 + np.exp(-q[i]))
                if b.sum() == 0: b[rng.randint(D)] = True
                fits[i] = self.fitness(b); masks.append(b)
                if fits[i] < best_f: best_f, best_b, best_q = fits[i], b.copy(), q[i].copy()
            worst_q = q[np.argmax(fits)].copy()
            r = (t / T) * 2 + 0.5                      # neighbourhood radius grows with iterations
            for i in range(P):
                d = np.linalg.norm(q - q[i], axis=1); nb = np.where((d < r * np.sqrt(D) / 10) & (d > 0))[0]
                if len(nb) > 0:
                    S_i = -(q[i] - q[nb]).sum(0); A_i = dq[nb].mean(0); C_i = q[nb].mean(0) - q[i]
                    F_i = best_q - q[i]; E_i = worst_q + q[i]
                    dq[i] = s * S_i + a * A_i + c * C_i + f_ * F_i + e_ * E_i + w * dq[i]
                else:
                    dq[i] = levy(D, self.beta, rng) * (q[i] if quantum else 1.0)
                if quantum:
                    theta = np.pi * np.tanh(dq[i]) * 0.1    # rotation gate, step scaled to the unit interval
                    q[i] = np.clip(q[i] * np.cos(theta) + np.sqrt(np.clip(1 - q[i] ** 2, 0, 1)) * np.sin(theta), 0.02, 0.98)
                else:
                    q[i] = np.clip(q[i] + dq[i], -6, 6)
            self.history.append((t, best_f, int(best_b.sum())))
            if self.verbose and (t % 10 == 0 or t == T - 1):
                print(f"  [{self.variant}] iter {t:3d} best fitness {best_f:.4f} |b|={best_b.sum()}", flush=True)
            if t >= 15 and all(abs(h[1] - best_f) < 1e-5 for h in self.history[-10:]):
                break
        return best_b, best_f
