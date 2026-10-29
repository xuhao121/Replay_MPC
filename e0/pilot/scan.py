"""Find scenarios where the lazy prediction (P_lin2, linearised at theta_A) picks a different
replay ratio than the true grid optimum."""
import numpy as np, json, sys, time
from core import Problem
RS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
SCEN = {
    'rot_sym':      dict(transform='rot90', nA=150, nB=150, tolA=0.03, tolB=0.1),
    'perm_sym':     dict(transform='perm',  nA=150, nB=150, tolA=0.03, tolB=0.1),
    'rot_smallA':   dict(transform='rot90', nA=75,  nB=225, tolA=0.03, tolB=0.1),
    'rot_tightA':   dict(transform='rot90', nA=150, nB=150, tolA=0.01, tolB=0.15),
    'perm_smallA':  dict(transform='perm',  nA=75,  nB=225, tolA=0.03, tolB=0.1),
    'rot_h64':      dict(transform='rot90', nA=150, nB=150, tolA=0.03, tolB=0.1, h=64),
    'perm_h64':     dict(transform='perm',  nA=150, nB=150, tolA=0.03, tolB=0.1, h=64),
}
def one(name, alpha, seed=0, H=5000):
    P = Problem(seed=seed, alpha=alpha, **SCEN[name])
    K0 = P.kernel(P.net.theta0); etaA = 0.9 / (np.linalg.eigvalsh(K0)[-1] / min(P.nA, P.nB))
    P.train_A(etaA)
    KA = P.kernel(P.thetaA); etaB = 0.9 * min(P.nA, P.nB) / np.linalg.eigvalsh(KA)[-1]
    lin = P.simulate([(KA, 0)], (P.FA - P.Y).reshape(-1), RS, etaB, H)['hit']
    true = [P.run_real(etaB, H, r)['T'] for r in RS]
    tl = [t if t is not None else -1 for t in true]
    def argbest(ts):
        ok = [(t, r) for t, r in zip(ts, RS) if t >= 0]
        return min(ok)[1] if ok else None
    rl, rt = argbest(lin), argbest(tl)
    T_at_rl = tl[RS.index(rl)] if rl is not None else None
    Tbest = min(t for t in tl if t >= 0) if any(t >= 0 for t in tl) else None
    return dict(scen=name, alpha=alpha, seed=seed, r_lin=rl, r_true=rt, T_true_at_rlin=T_at_rl, T_best=Tbest,
                regret=(T_at_rl / Tbest if (T_at_rl not in (None, -1) and Tbest) else None),
                lin=dict(zip(RS, lin.tolist())), true=dict(zip(RS, tl)))
if __name__ == '__main__':
    out = []
    for name in ['rot_sym', 'perm_sym', 'rot_smallA', 'rot_tightA', 'perm_smallA']:
        for alpha in [1.0, 0.05]:
            t = time.time(); res = one(name, alpha); res['sec'] = round(time.time() - t)
            print(json.dumps({k: res[k] for k in ['scen', 'alpha', 'r_lin', 'r_true', 'T_true_at_rlin', 'T_best', 'regret', 'sec']}), flush=True)
            out.append(res)
    json.dump(out, open('scan.json', 'w'), indent=1)
