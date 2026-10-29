"""Core pieces for testing replay-ratio prediction inside and outside the lazy regime.

Model: one-hidden-layer tanh MLP in NTK parameterisation (no biases), output scaled as
    f(x; theta) = alpha * (g(x; theta) - g(x; theta_0)),   loss scaled by 1/alpha^2
(Chizat, Oyallon & Bach 2019). In function space one GD step is
    F <- F - eta * K(theta) W (F - Y),
so alpha leaves the linearised dynamics unchanged and only controls how far the real kernel
K(theta) drifts: large alpha = lazy (linearisation accurate), small alpha = rich.

Tasks: sklearn digits. Task A = original images; task B = images transformed by `transform`
(rot90 or a fixed pixel permutation), same 10 labels. Loss: per-sample 0.5*||f - y||^2 with
centred one-hot targets, averaged within each task. Mixed objective with replay ratio r:
(1 - r) L_B + r L_A, full-batch GD (no SGD noise, so differences come from the kernel only).
"""
import numpy as np
from sklearn.datasets import load_digits


def make_data(seed, nA=150, nB=150, transform='rot90'):
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    iA, iB = idx[:nA], idx[nA:nA + nB]
    XA = X[iA].astype(np.float64) / 16.0
    XB = X[iB].astype(np.float64) / 16.0
    if transform == 'rot90':
        XB = np.rot90(XB.reshape(-1, 8, 8), k=1, axes=(1, 2)).reshape(-1, 64)
    elif transform == 'perm':
        XB = XB[:, np.random.default_rng(777).permutation(64)]
    else:
        raise ValueError(transform)
    Xall = np.vstack([XA, XB])
    Xall = Xall - Xall.mean(0)
    Xall = Xall / Xall.std()
    Y = np.eye(10)[np.concatenate([y[iA], y[iB]])] - 0.1
    isA = np.zeros(nA + nB, bool); isA[:nA] = True
    return Xall, Y, isA


class Net:
    def __init__(self, d, h, C, seed):
        rng = np.random.default_rng(10_000 + seed)
        self.d, self.h, self.C = d, h, C
        self.theta0 = dict(W1=rng.standard_normal((h, d)), W2=rng.standard_normal((C, h)))

    def raw(self, th, X):
        P = np.tanh(X @ th['W1'].T / np.sqrt(self.d))
        return P, P @ th['W2'].T / np.sqrt(self.h)

    def vjp(self, th, X, P, V):
        """J^T V for the raw network g (V: n x C cotangent)."""
        gW2 = V.T @ P / np.sqrt(self.h)
        dZ = (V @ th['W2']) / np.sqrt(self.h) * (1.0 - P * P)
        return dict(W1=dZ.T @ X / np.sqrt(self.d), W2=gW2)

    def ntk(self, th, X):
        """Empirical NTK of the raw network on X, rows ordered (sample, output)."""
        P, _ = self.raw(th, X)
        n, C, h = X.shape[0], self.C, self.h
        D = 1.0 - P * P
        G = (D[:, None, :] * th['W2'][None, :, :]).reshape(n * C, h) / np.sqrt(h)
        K = (G @ G.T) * np.kron(X @ X.T / self.d, np.ones((C, C)))
        K += np.kron(P @ P.T / h, np.eye(C))
        return K


class Problem:
    def __init__(self, seed, alpha, nA=150, nB=150, transform='rot90', h=256, tolA=0.03, tolB=0.1):
        self.X, self.Y, self.isA = make_data(seed, nA, nB, transform)
        self.nA, self.nB = nA, nB
        self.net = Net(self.X.shape[1], h, 10, seed)
        self.alpha = alpha
        _, self.g0 = self.net.raw(self.net.theta0, self.X)
        self.rowA = np.repeat(self.isA, 10)
        self.tolA, self.tolB = tolA, tolB

    def weights(self, r):
        return np.where(self.isA, r / self.nA, (1.0 - r) / self.nB)

    def losses_from_F(self, F):
        E = F - self.Y
        return 0.5 * (E[self.isA] ** 2).sum() / self.nA, 0.5 * (E[~self.isA] ** 2).sum() / self.nB

    def F(self, th):
        return self.alpha * (self.net.raw(th, self.X)[1] - self.g0)

    def train_A(self, eta, steps=20000, target_rel=2e-3):
        th = {k: v.copy() for k, v in self.net.theta0.items()}
        w = self.weights(1.0)
        LB_init = None
        for t in range(steps):
            P, out = self.net.raw(th, self.X)
            F = self.alpha * (out - self.g0)
            LA, LB = self.losses_from_F(F)
            if LB_init is None: LB_init = LB
            if LA <= target_rel * LB_init: break
            g = self.net.vjp(th, self.X, P, w[:, None] * (F - self.Y))
            th = {k: th[k] - eta * g[k] / self.alpha for k in th}
        self.thetaA = th
        self.FA = self.F(th)
        self.LA0, self.LB0 = self.losses_from_F(self.FA)
        self.tauA = self.LA0 + self.tolA * self.LB0
        self.tauB = self.tolB * self.LB0
        return t

    def kernel(self, th, idx=None):
        return self.net.ntk(th, self.X if idx is None else self.X[idx])

    def run_real(self, eta, H, r0, check_every=None, on_check=None, start=None):
        """Real full-batch GD on (1-r) L_B + r L_A starting at theta_A (or `start`).
        on_check(t, th, F) -> new r or None, called every check_every steps.
        Returns T (first step inside the tolerance band, or None), peak forgetting, r trace."""
        th = {k: v.copy() for k, v in (self.thetaA if start is None else start).items()}
        r = float(r0); r_trace = [(0, r)]; peak = 0.0
        for t in range(H + 1):
            P, out = self.net.raw(th, self.X)
            F = self.alpha * (out - self.g0)
            LA, LB = self.losses_from_F(F)
            peak = max(peak, (LA - self.LA0) / self.LB0)
            if LA <= self.tauA and LB <= self.tauB:
                return dict(T=t, peak=peak, r_trace=r_trace, th=th)
            if t == H: break
            if on_check is not None and check_every and t > 0 and t % check_every == 0:
                new_r = on_check(t, th, F)
                if new_r is not None and new_r != r:
                    r = new_r; r_trace.append((t, r))
            g = self.net.vjp(th, self.X, P, self.weights(r)[:, None] * (F - self.Y))
            th = {k: th[k] - eta * g[k] / self.alpha for k in th}
        return dict(T=None, peak=peak, r_trace=r_trace, th=th)

    def simulate(self, segments, E0, r_list, eta, steps, stop_first=False):
        """Linearised function-space GD  E <- E - eta K W_r E  for all r at once.
        segments: list of (K, n_steps); the last kernel is used for all remaining steps.
        Returns hitting times (-1 = not reached) and the normalised violation at hit / end."""
        R = len(r_list)
        E = np.repeat(E0.reshape(-1, 1), R, axis=1).astype(np.float32)
        rowA = self.rowA
        W = np.stack([np.repeat(self.weights(r), 10) for r in r_list], 1).astype(np.float32)
        hit = np.full(R, -1); viol = np.full(R, np.inf)
        bounds = list(np.cumsum([ns for _, ns in segments[:-1]]))
        kernels = [k.astype(np.float32) for k, _ in segments]
        seg = 0
        for t in range(steps + 1):
            LA = 0.5 * (E[rowA] ** 2).sum(0) / self.nA
            LB = 0.5 * (E[~rowA] ** 2).sum(0) / self.nB
            v = np.maximum(LA / self.tauA, LB / self.tauB)
            newly = (LA <= self.tauA) & (LB <= self.tauB) & (hit < 0)
            hit[newly] = t; viol[newly] = v[newly]
            if (stop_first and newly.any()) or (hit >= 0).all(): break
            while seg < len(bounds) and t >= bounds[seg]: seg += 1
            E -= eta * (kernels[seg] @ (W * E))
        miss = hit < 0
        viol[miss] = v[miss]
        return dict(hit=hit, viol=viol)

    def plan(self, segments, E0, r_list, eta, remaining):
        """r with the earliest predicted entry into the band (ties: smaller violation);
        if none enters within `remaining`, the r with the smallest final violation."""
        res = self.simulate(segments, E0, r_list, eta, remaining, stop_first=True)
        hit = res['hit']
        if (hit >= 0).any():
            tmin = hit[hit >= 0].min()
            cands = np.where(hit == tmin)[0]
            i = cands[np.argmin(res['viol'][cands])]
            return float(r_list[i]), int(tmin)
        return float(r_list[int(np.argmin(res['viol']))]), None
