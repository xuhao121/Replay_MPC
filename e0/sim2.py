"""Fair comparison (learning rate tuned per method) of replay-weighting rules in an
overparameterized two-task linear regression where the joint solution set exists.
Reports steps to reach the joint tolerance band and the peak old-task loss -> results/sim2.json.
    python e0/sim2.py
"""
import numpy as np, json, os, sys

def make(seed, d=400, nA=100, nB=100, alpha=0.5, scaleB=1.0):
    rng = np.random.default_rng(seed)
    eig = 1.0 / np.arange(1, d + 1) ** alpha
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S = (Q * np.sqrt(eig)) @ Q.T
    XA = rng.standard_normal((nA, d)) @ S
    XB = scaleB * rng.standard_normal((nB, d)) @ S
    w = rng.standard_normal(d)
    return XA, XA @ w, XB, XB @ w

def lmax(M): return np.linalg.eigvalsh(M)[-1]

def run(XA, yA, XB, yB, thetaA, method, eta, par, T=20000, tol=1e-3):
    nA, nB = len(yA), len(yB)
    w = thetaA.copy()
    rB = XB @ w - yB; LB0 = 0.5 * np.mean(rB * rB)
    hit = None; maxLA = 0.0; last_bad = -1; t_end = T
    for t in range(T):
        rA = XA @ w - yA; rB = XB @ w - yB
        LA = 0.5 * np.mean(rA * rA) / LB0; LB = 0.5 * np.mean(rB * rB) / LB0
        maxLA = max(maxLA, LA)
        ok = (LA <= tol) and (LB <= tol)
        if not ok: last_bad = t
        if ok and hit is None: hit = t
        if hit is not None and t > hit + 300: t_end = t; break
        if not np.isfinite(LA) or LA > 1e6: return dict(T_hit=None, T_stable=None, maxLA=float('inf'))
        gA = XA.T @ rA / nA; gB = XB.T @ rB / nB
        if method == 'fixed_r':
            r = par; G = (1 - r) * gB + r * gA
        elif method == 'normbal':                      # ratio-only equal-norm rule (user's rule)
            na, nb = np.linalg.norm(gA), np.linalg.norm(gB)
            r = np.clip(nb / (na + nb + 1e-300), 0.01, 0.99); G = (1 - r) * gB + r * gA
        elif method == 'normbal_lr':                   # equal norms, new-task weight fixed at 1
            na, nb = np.linalg.norm(gA), np.linalg.norm(gB)
            lam = min(nb / (na + 1e-300), 1e4); G = gB + lam * gA
        elif method == 'agem':                         # A-GEM projection with the replay gradient
            dot = gB @ gA
            G = gB - (dot / (gA @ gA + 1e-300)) * gA if dot < 0 else gB
        elif method == 'lossbal':                      # MEGA-I style loss-ratio weight, no switch
            G = gB + (LA / (LB + 1e-300)) * gA
        elif method == 'megaI':                        # MEGA-I with epsilon switch
            G = gB + (LA / LB) * gA if LB > par else gA
        else:
            raise ValueError(method)
        w = w - eta * G
    T_stable = (last_bad + 1) if hit is not None else None
    return dict(T_hit=hit, T_stable=T_stable, maxLA=float(maxLA))

def scenario(seed, alpha, scaleB, T):
    XA, yA, XB, yB = make(seed, alpha=alpha, scaleB=scaleB)
    nA, nB = len(yA), len(yB)
    HA, HB = XA.T @ XA / nA, XB.T @ XB / nB
    thetaA = np.linalg.pinv(XA) @ yA
    mults = [0.5, 1.0, 1.5, 1.9]
    res = {}
    def best(name, method, base, par=None):
        cands = []
        for m in mults:
            out = run(XA, yA, XB, yB, thetaA, method, m / base, par, T=T)
            out['eta_mult'] = m; cands.append(out)
        ok = [c for c in cands if c['T_stable'] is not None]
        res[name] = min(ok, key=lambda c: (c['T_stable'], c['maxLA'])) if ok else \
            dict(T_hit=None, T_stable=None, maxLA=min(c['maxLA'] for c in cands))
    for r in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        best(f'fixed_r={r}', 'fixed_r', lmax((1 - r) * HB + r * HA), r)
    best('normbal(ratio only)', 'normbal', max(lmax(HA), lmax(HB)))
    best('normbal_lr', 'normbal_lr', lmax(HB))
    best('A-GEM', 'agem', lmax(HB))
    best('lossbal(MEGA-I no switch)', 'lossbal', lmax(HB))
    best('MEGA-I(eps=1e-3)', 'megaI', lmax(HB), 1e-3)
    return res

if __name__ == '__main__':
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    allres = {}
    for alpha, scaleB in [(0.5, 1.0), (0.5, 3.0), (0.5, 1 / 3), (1.0, 1.0)]:
        for seed in [0, 1, 2]:
            allres[f'alpha={alpha},scaleB={scaleB:.2f},seed={seed}'] = scenario(seed, alpha, scaleB, T)
            print('done', alpha, scaleB, seed, flush=True, file=sys.stderr)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sim2.json')
    json.dump(allres, open(out, 'w'), indent=1)
    print('written', out)
