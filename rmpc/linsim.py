"""Linearised function-space dynamics and the replay-ratio planner.

Around a linearisation point the network is replaced by its tangent model, so full-batch GD on
    sum_s w_s 0.5 ||f(x_s) - y_s||^2,   w_s = cA / nA (task A samples), cB / nB (task B samples)
becomes, in function space on the training set,
    E <- E - eta K W E,      E = F - Y  (n x C residual),  K = empirical NTK,  W = diag(w).
For a fixed replay ratio r, (cA, cB) = (r, 1 - r). The simulator runs all candidate mixings at once
(columns of an (n, C, R) array), so one matrix product per step serves the whole r grid.

T(r) = first step at which L_A <= tau_A and L_B <= tau_B (both tasks inside the tolerance band).
"""
import math
import torch


def _masks(isA, dev):
    mA = isA.to(dev).to(torch.float32)
    return mA, 1.0 - mA


@torch.no_grad()
def simulate(K, E0, coefs, isA, nA, nB, eta, steps, tauA, tauB, stop_first=False, stride=16,
             record=None, coefs2=None, ts=None):
    """Return dict(hit (R,) int [-1 = not reached], viol (R,) max(L_A/tau_A, L_B/tau_B) at hit or at the end,
    LA, LB (R,) at the end, E (n, C, R) at the end, rec {step: E} for steps in `record`).
    Two-phase columns: with coefs2 and ts given, column c uses coefs[c] for t < ts[c] and coefs2[c] after."""
    dev = K.mat.device
    Km = K.mat if K.mat.dtype == torch.float32 else K.mat.float()
    Kf = type(K)(Km, K.kind, K.n, K.C)
    coefs = torch.as_tensor(coefs, dtype=torch.float32, device=dev).reshape(-1, 2)
    R = coefs.shape[0]
    E = E0.to(dev, torch.float32).unsqueeze(-1).repeat(1, 1, R)
    mA, mB = _masks(isA, dev)
    W3 = torch.where(isA.to(dev)[:, None], coefs[None, :, 0] / nA, coefs[None, :, 1] / nB)[:, None, :]
    W3b = tsw = None
    if coefs2 is not None:
        c2 = torch.as_tensor(coefs2, dtype=torch.float32, device=dev).reshape(-1, 2)
        W3b = torch.where(isA.to(dev)[:, None], c2[None, :, 0] / nA, c2[None, :, 1] / nB)[:, None, :]
        tsw = torch.as_tensor(ts, dtype=torch.float32, device=dev).reshape(1, 1, -1)
    hit = torch.full((R,), -1.0, device=dev)
    viol = torch.full((R,), float('inf'), device=dev)
    rec = {}
    record = set(record or [])
    t = 0
    for t in range(steps + 1):
        if t in record:
            rec[t] = E.clone()
        E2 = (E * E).sum(1)                                   # (n, R)
        LA = (mA @ E2) * (0.5 / nA)
        LB = (mB @ E2) * (0.5 / nB)
        v = torch.maximum(LA / tauA, LB / tauB)
        newly = (LA <= tauA) & (LB <= tauB) & (hit < 0)
        hit = torch.where(newly, torch.full_like(hit, float(t)), hit)
        viol = torch.where(newly, v, viol)
        if t == steps:
            break
        if t % stride == 0 or t == steps - 1:
            reached = hit >= 0
            if (stop_first and bool(reached.any())) or bool(reached.all()):
                break
        W = W3 if W3b is None else torch.where(tsw > t, W3, W3b)
        E -= eta * Kf.mv(W * E)
    miss = hit < 0
    viol = torch.where(miss, v, viol)
    return dict(hit=hit.long().cpu(), viol=viol.cpu(), LA=LA.cpu(), LB=LB.cpu(), E=E, rec=rec, steps=t)


def r_coefs(r_list):
    return [[float(r), 1.0 - float(r)] for r in r_list]


def choose(hit, viol, r_list):
    """Earliest predicted entry (ties: smaller violation); none reached -> smallest final violation."""
    hit = hit.tolist()
    viol = viol.tolist()
    ok = [i for i, h in enumerate(hit) if h >= 0]
    if ok:
        tmin = min(hit[i] for i in ok)
        cands = [i for i in ok if hit[i] == tmin]
        i = min(cands, key=lambda j: viol[j])
        return float(r_list[i]), int(tmin)
    i = min(range(len(hit)), key=lambda j: viol[j])
    return float(r_list[i]), None


def plan(K, E0, r_list, isA, nA, nB, eta, remaining, tauA, tauB):
    res = simulate(K, E0, r_coefs(r_list), isA, nA, nB, eta, remaining, tauA, tauB, stop_first=True)
    r, T = choose(res['hit'], res['viol'], r_list)
    return dict(r=r, T_hat=T)


def plan2(K, E0, r_list, isA, nA, nB, eta, remaining, tauA, tauB, M, js=(1, 2, 4, 8), r_list2=None):
    """Two-phase receding-horizon plan: candidates 'r1 for the next j*M steps, then r2' plus every constant r.
    Returns the r1 of the candidate with the earliest predicted entry (ties: smaller violation)."""
    r2s = list(r_list2 or r_list)
    c1, c2, ts, lab = [], [], [], []
    for a in r_list:
        for b in r2s:
            if a == b:
                continue
            for j in js:
                c1.append([a, 1 - a]); c2.append([b, 1 - b]); ts.append(j * M); lab.append((a, b, j * M))
    for a in r_list:
        c1.append([a, 1 - a]); c2.append([a, 1 - a]); ts.append(0); lab.append((a, a, 0))
    res = simulate(K, E0, c1, isA, nA, nB, eta, remaining, tauA, tauB, stop_first=True, coefs2=c2, ts=ts)
    hit, viol = res['hit'].tolist(), res['viol'].tolist()
    ok = [i for i, h in enumerate(hit) if h >= 0]
    if ok:
        tmin = min(hit[i] for i in ok)
        i = min((i for i in ok if hit[i] == tmin), key=lambda k: viol[k])
        T = int(tmin)
    else:
        i = min(range(len(hit)), key=lambda k: viol[k])
        T = None
    a, b, t_s = lab[i]
    return dict(r=float(a), r2=float(b), t_switch=int(t_s), T_hat=T, n_cols=len(lab))


def predict_curve(K, E0, r_list, isA, nA, nB, eta, steps, tauA, tauB):
    """Predicted T(r) for every r (no early stop) -> (list of hits, list of violations)."""
    res = simulate(K, E0, r_coefs(r_list), isA, nA, nB, eta, steps, tauA, tauB, stop_first=False)
    return res['hit'].tolist(), res['viol'].tolist()


def predict_window(K, E0, coef, isA, nA, nB, eta, M):
    """Residual after M linearised steps with one mixing coef = (cA, cB). Returns (n, C) float32."""
    res = simulate(K, E0, [list(coef)], isA, nA, nB, eta, M, -1.0, -1.0, stop_first=False)
    return res['E'][..., 0]


@torch.no_grad()
def lam_max_weighted(K, w, iters=200, seed=0):
    """lambda_max(K W), W = diag(w) with one weight per sample (shared by its C outputs).
    Computed as lambda_max(W^1/2 K W^1/2): exactly for small kernels, by power iteration otherwise."""
    sw = w.to(K.mat.device, K.mat.dtype).clamp(min=0).sqrt()
    N = K.mat.shape[0]
    if N <= 4000:
        s = sw.repeat_interleave(K.C) if K.kind == 'full' else sw
        M = s[:, None] * K.mat * s[None, :]
        return float(torch.linalg.eigvalsh(M.double())[-1])
    g = torch.Generator(device='cpu').manual_seed(seed)
    v = torch.randn(K.n, K.C, 1, generator=g, dtype=torch.float64).to(K.mat.device, K.mat.dtype)
    v /= torch.linalg.norm(v)
    s3 = sw[:, None, None]
    for _ in range(iters):
        u = s3 * K.mv(s3 * v)
        v = u / torch.linalg.norm(u)
    u = s3 * K.mv(s3 * v)
    return float((v * u).sum())


def sample_weights(isA, nA, nB, cA, cB):
    return torch.where(isA, torch.full(isA.shape, cA / nA, dtype=torch.float64, device=isA.device),
                       torch.full(isA.shape, cB / nB, dtype=torch.float64, device=isA.device))
