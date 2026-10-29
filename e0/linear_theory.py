"""E0: exact checks of the theory in a two-task over-parameterised linear regression (CPU, minutes).

The model is linear, so the NTK is constant and every linearised prediction is exact. What is tested:
  1. quasi-static forgetting   L_A - L_A(theta_A) = 0.5 ((1-r)/r)^2 g_B^T H_A^+ g_B        (leaky integrator)
  2. U-shaped T(r) = max(T_A(r), T_B(r)): forgetting constraint decreasing in r, learning constraint increasing
  3. schedule invariance       every converging schedule ends at the same joint solution (closest to theta_A)
  4. receding-horizon re-planning beats the best fixed r even with an exact model (no kernel drift),
     and the best two-phase schedule bounds how much a time-varying r can gain
  5. gradient-statistics controllers (sim2.py): ratio rule stalls; LR-compensated rule ~ best tuned fixed r
  6. quasi-static norm ratio  r|g_A| / ((1-r)|g_B|) = rho = |P_N g_B| / |g_B| < 1, independent of r
Writes e0/results/linear_theory.json and e0/results/fig_linear.png.
    python e0/linear_theory.py            (about 5 min on 2 cores; --quick for about 1 min)
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2 import make, lmax, scenario as sim2_scenario

GRID = [round(0.05 * i, 2) for i in range(1, 20)]


class Lin:
    """Function-space view of the linear model: F = X w, K = X X^T, loss 0.5 mean (F - y)^2 per task."""

    def __init__(self, seed=0, alpha=0.5, scaleB=1.0, tolA=0.03, tolB=0.1):
        # alpha: power-law exponent of the shared input covariance spectrum (1.0 gives T* ~ 200-800 steps,
        # comparable to the networks); scaleB rescales task-B inputs (curvature x scaleB^2)
        self.XA, self.yA, self.XB, self.yB = make(seed, alpha=alpha, scaleB=scaleB)
        self.nA, self.nB = len(self.yA), len(self.yB)
        self.X = np.vstack([self.XA, self.XB])
        self.y = np.concatenate([self.yA, self.yB])
        self.isA = np.r_[np.ones(self.nA, bool), np.zeros(self.nB, bool)]
        self.K = self.X @ self.X.T
        self.thA = np.linalg.pinv(self.XA) @ self.yA
        self.EA = self.X @ self.thA - self.y
        self.LA0, self.LB0 = self.losses(self.EA)
        self.tauA, self.tauB = self.LA0 + tolA * self.LB0, tolB * self.LB0
        m = min(self.nA, self.nB)
        self.eta = 0.9 * m / np.linalg.eigvalsh(self.K)[-1]
        self.HA = self.XA.T @ self.XA / self.nA
        self.HA_pinv = np.linalg.pinv(self.HA, rcond=1e-10)

    def losses(self, E):
        return 0.5 * np.mean(E[self.isA] ** 2), 0.5 * np.mean(E[~self.isA] ** 2)

    def w(self, r):
        return np.where(self.isA, r / self.nA, (1 - r) / self.nB)

    def run(self, schedule, H=20000, E0=None, t0=0):
        """schedule(t, E) -> r. Returns T (first step inside the band), T_A, T_B and the loss curves."""
        E = (self.EA if E0 is None else E0).copy()
        LA_c, LB_c = [], []
        T = None
        for t in range(H + 1):
            LA, LB = self.losses(E)
            LA_c.append(LA); LB_c.append(LB)
            if T is None and LA <= self.tauA and LB <= self.tauB:
                T = t
                break
            r = schedule(t, E)
            E = E - self.eta * self.K @ (self.w(r) * E)
        LA_c, LB_c = np.array(LA_c), np.array(LB_c)
        above = np.where(LA_c > self.tauA)[0]
        T_A = int(above[-1] + 1) if len(above) else 0
        T_B = int(np.argmax(LB_c <= self.tauB)) if (LB_c <= self.tauB).any() else None
        return dict(T=T, T_A=T_A, T_B=T_B, LA=LA_c, LB=LB_c)

    def predict_T(self, r, E0, H):
        return self.run(lambda t, E: r, H=H, E0=E0)['T']

    def plan(self, E0, H, grid=GRID):
        Ts = {r: self.predict_T(r, E0, H) for r in grid}
        ok = {r: T for r, T in Ts.items() if T is not None}
        return (min(ok, key=ok.get), ok[min(ok, key=ok.get)]) if ok else (0.5, None)


# ----------------------------------------------------------------------------- 1. quasi-static forgetting
def check_quasistatic(seed=0):
    out = []
    P = Lin(seed, scaleB=0.2)                    # new task 25x flatter: old task relaxes much faster
    for r in (0.2, 0.5, 0.8):
        w = P.thA.copy()
        eta = 1.0 / lmax((1 - r) * P.XB.T @ P.XB / P.nB + r * P.HA)
        row = dict(r=r)
        for t in range(3001):
            gB = P.XB.T @ (P.XB @ w - P.yB) / P.nB
            gA = P.XA.T @ (P.XA @ w - P.yA) / P.nA
            if t in (100, 1000, 3000):
                dLA = 0.5 * np.mean((P.XA @ w - P.yA) ** 2)
                pred = 0.5 * ((1 - r) / r) ** 2 * gB @ P.HA_pinv @ gB
                row[f't{t}'] = dict(measured=dLA, predicted=pred, ratio=dLA / pred)
            w = w - eta * ((1 - r) * gB + r * gA)
        out.append(row)
    return out


# ----------------------------------------------------------------------------- 2. U shape
def check_ushape(seed=0, H=20000):
    P = Lin(seed, alpha=1.0)
    rows = []
    for r in GRID:
        o = P.run(lambda t, E: r, H=H)
        rows.append(dict(r=r, T=o['T'], T_A=o['T_A'], T_B=o['T_B'],
                         max_rule=(max(o['T_A'], o['T_B']) if o['T_B'] is not None else None)))
    return rows


# ----------------------------------------------------------------------------- 3. schedule invariance
def check_invariance(seed=0, T=20000):
    P = Lin(seed)
    th_joint = P.thA + np.linalg.pinv(P.X) @ (P.y - P.X @ P.thA)
    scale = np.linalg.norm(th_joint - P.thA)
    HB = P.XB.T @ P.XB / P.nB
    res = {}
    HA = P.HA
    mix = lambda r: (1 - r) * HB + r * HA
    rules = {'fixed r=0.2': (lambda t, a, b: 0.8 * b + 0.2 * a, 1.9 / lmax(mix(0.2))),
             'fixed r=0.7': (lambda t, a, b: 0.3 * b + 0.7 * a, 1.9 / lmax(mix(0.7))),
             'switching 0.2 -> 0.8': (lambda t, a, b: (0.8 * b + 0.2 * a) if t < 500 else (0.2 * b + 0.8 * a),
                                      1.9 / max(lmax(mix(0.2)), lmax(mix(0.8)))),
             'normbal_lr': (lambda t, a, b: b + min(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-300), 1e4) * a,
                            1.0 / lmax(HB))}
    for name, (rule, eta) in rules.items():
        w = P.thA.copy()
        for t in range(T):
            gA = P.XA.T @ (P.XA @ w - P.yA) / P.nA
            gB = P.XB.T @ (P.XB @ w - P.yB) / P.nB
            w = w - eta * rule(t, gA, gB)
        res[name] = float(np.linalg.norm(w - th_joint) / scale)
    return res


# ----------------------------------------------------------------------------- 4. re-planning vs fixed r
def check_replanning(seeds=(0, 1, 2), scales=(1.0, 3.0, 1 / 3), n_checks=10, H=20000):
    out = []
    for seed in seeds:
        for sc in scales:
            P = Lin(seed, alpha=1.0, scaleB=sc)
            T = {r: P.run(lambda t, E, r=r: r, H=H)['T'] for r in GRID}
            ok = {r: v for r, v in T.items() if v is not None}
            r_best = min(ok, key=ok.get)
            T_best = ok[r_best]
            r_lin, T_hat = P.plan(P.EA, H)
            M = max(20, int(T_hat / n_checks))
            state = dict(r=r_lin)

            def sched(t, E, state=state):
                if t > 0 and t % M == 0:
                    state['r'] = P.plan(E, H - t)[0]
                return state['r']
            T_mpc = P.run(sched, H=H)['T']
            # best two-phase schedule r1 -> r2 at t_s (exhaustive over a coarse grid)
            best2 = (T_best, r_best, r_best, 0)
            for ts_frac in (0.2, 0.4, 0.6, 0.8):
                ts = int(ts_frac * T_best)
                for r1 in GRID[::2]:
                    o1 = P.run(lambda t, E, r1=r1: r1, H=ts)
                    if o1['T'] is not None:
                        continue
                    E1 = P.EA.copy()
                    for t in range(ts):
                        E1 = E1 - P.eta * P.K @ (P.w(r1) * E1)
                    for r2 in GRID:
                        T2 = P.run(lambda t, E, r2=r2: r2, H=T_best, E0=E1)['T']
                        if T2 is not None and ts + T2 < best2[0]:
                            best2 = (ts + T2, r1, r2, ts)
            out.append(dict(seed=seed, scaleB=sc, r_best=r_best, T_best=T_best, r_lin=r_lin, plin2_exact=(T[r_lin] == T_best),
                            T_mpc=T_mpc, regret_mpc=T_mpc / T_best, T_two_phase=best2[0], two_phase=best2[1:],
                            regret_two_phase=best2[0] / T_best))
            print(f"  seed={seed} scaleB={sc:.2f}: r*={r_best} T*={T_best}  P_lin2 r={r_lin}  MPC regret={T_mpc / T_best:.3f}"
                  f"  best two-phase {best2[1]}->{best2[2]} at t={best2[3]}: regret={best2[0] / T_best:.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------- 6. norm ratio
def check_ratio(scaleB=0.2, seed=0):
    XA, yA, XB, yB = make(seed, alpha=0.5, scaleB=scaleB)
    nA, nB = len(yA), len(yB)
    HA, HB = XA.T @ XA / nA, XB.T @ XB / nB
    PN = np.linalg.pinv(XA) @ XA
    thA = np.linalg.pinv(XA) @ yA
    rows = []
    for r in (0.3, 0.5, 0.8):
        eta = 1.9 / lmax((1 - r) * HB + r * HA)
        w = thA.copy()
        for t in range(1001):
            gA = XA.T @ (XA @ w - yA) / nA
            gB = XB.T @ (XB @ w - yB) / nB
            if t in (10, 100, 1000):
                rows.append(dict(r=r, t=t, ratio=float(r * np.linalg.norm(gA) / ((1 - r) * np.linalg.norm(gB))),
                                 rho=float(np.linalg.norm(PN @ gB) / np.linalg.norm(gB))))
            w = w - eta * ((1 - r) * gB + r * gA)
    return rows


def figure(ush, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = [x['r'] for x in ush]
    f = lambda k: [x[k] if x[k] is not None else np.nan for x in ush]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(r, f('T'), 'o-', color='#222222', label='T(r): both tasks in band')
    ax.plot(r, f('T_A'), '--', color='#eb6834', label='T_A(r): old task back in band')
    ax.plot(r, f('T_B'), '--', color='#2a78d6', label='T_B(r): new task in band')
    ax.set_yscale('log'); ax.set_xlabel('replay ratio r'); ax.set_ylabel('steps')
    ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    out = {}
    print('1. quasi-static forgetting (measured / predicted):')
    out['quasistatic'] = check_quasistatic()
    for row in out['quasistatic']:
        print(f"  r={row['r']}: " + '  '.join(f"{k}: {v['ratio']:.3f}" for k, v in row.items() if k != 'r'))
    print('2. U shape, T = max(T_A, T_B):')
    out['ushape'] = check_ushape()
    for x in out['ushape']:
        print(f"  r={x['r']:.2f}  T={x['T']}  T_A={x['T_A']}  T_B={x['T_B']}  max={x['max_rule']}")
    figure(out['ushape'], os.path.join(HERE, 'results', 'fig_linear.png'))
    print('3. schedule invariance (relative distance to the closest joint solution):')
    out['invariance'] = check_invariance(T=5000 if a.quick else 20000)
    for k, v in out['invariance'].items():
        print(f'  {k:22s} {v:.2e}')
    print('4. receding-horizon re-planning with an exact model:')
    out['replanning'] = check_replanning(seeds=(0,) if a.quick else (0, 1, 2))
    print('6. norm ratio vs rho:')
    out['ratio'] = check_ratio()
    for x in out['ratio']:
        print(f"  r={x['r']} t={x['t']}: ratio {x['ratio']:.3f}  rho {x['rho']:.3f}")
    if not a.quick:
        print('5. gradient-statistics controllers (per-method tuned step size; sim2.py):')
        out['gradstat'] = {}
        for alpha, sc in [(0.5, 1.0), (0.5, 3.0), (0.5, 1 / 3), (1.0, 1.0)]:
            res = sim2_scenario(0, alpha, sc, 20000)
            out['gradstat'][f'alpha={alpha},scaleB={sc:.2f}'] = res
            best = min((v['T_stable'], k) for k, v in res.items() if k.startswith('fixed') and v['T_stable'])
            print(f"  alpha={alpha} scaleB={sc:.2f}: best fixed {best[1]} T={best[0]}; " +
                  ', '.join(f"{k}: {v['T_stable']}" for k, v in res.items() if not k.startswith('fixed')))
    with open(os.path.join(HERE, 'results', 'linear_theory.json'), 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('written e0/results/linear_theory.json and fig_linear.png')


if __name__ == '__main__':
    main()
