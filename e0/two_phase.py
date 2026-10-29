"""E0 follow-up: how much can a time-varying replay ratio gain over the best fixed r, and does a two-phase
receding-horizon planner (plan "r1 for the next j windows, then r2", apply r1, re-plan) capture it?
Exact linear model (no kernel drift), vectorised over candidate schedules.
    python e0/two_phase.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from linear_theory import Lin, GRID


def sim_cols(P, E0, r1, r2, ts, H):
    """Columns c: r1[c] for t < ts[c], r2[c] afterwards. Returns first hit time per column (-1 = none)."""
    R = len(r1)
    E = np.repeat(E0[:, None], R, 1)
    W1 = np.where(P.isA[:, None], np.asarray(r1)[None] / P.nA, (1 - np.asarray(r1))[None] / P.nB)
    W2 = np.where(P.isA[:, None], np.asarray(r2)[None] / P.nA, (1 - np.asarray(r2))[None] / P.nB)
    ts = np.asarray(ts)
    hit = np.full(R, -1)
    mA = P.isA.astype(float)
    for t in range(H + 1):
        E2 = E * E
        LA = 0.5 * (mA @ E2) / P.nA
        LB = 0.5 * ((1 - mA) @ E2) / P.nB
        newly = (LA <= P.tauA) & (LB <= P.tauB) & (hit < 0)
        hit[newly] = t
        if (hit >= 0).all():
            break
        W = np.where((t < ts)[None, :], W1, W2)
        E = E - P.eta * P.K @ (W * E)
    return hit


def best_of(hit, *labels):
    ok = np.where(hit >= 0)[0]
    if not len(ok):
        return None
    i = ok[np.argmin(hit[ok])]
    return (int(hit[i]),) + tuple(l[i] for l in labels)


def plan_two_phase(P, E0, H, M, js=(1, 2, 4, 8)):
    r1, r2, ts = [], [], []
    for a in GRID:
        for b in GRID:
            for j in js:
                r1.append(a); r2.append(b); ts.append(j * M)
    for a in GRID:                                     # constant schedules
        r1.append(a); r2.append(a); ts.append(0)
    hit = sim_cols(P, E0, r1, r2, ts, H)
    return best_of(hit, r1, r2, ts)


def run_policy(P, H, M, planner):
    """Receding horizon: every M steps re-plan from the current state, apply the first phase."""
    E = P.EA.copy()
    r = planner(E, H)[1]
    trace = [(0, r)]
    for t in range(H + 1):
        LA, LB = P.losses(E)
        if LA <= P.tauA and LB <= P.tauB:
            return t, trace
        if t > 0 and t % M == 0:
            res = planner(E, H - t)
            if res is not None and res[1] != r:
                r = res[1]
                trace.append((t, r))
        E = E - P.eta * P.K @ (P.w(r) * E)
    return None, trace


def main():
    out = []
    for seed in (0, 1, 2):
        for sc in (1.0, 1 / 3):
            P = Lin(seed, alpha=1.0, scaleB=sc)
            H = 6000
            const = sim_cols(P, P.EA, GRID, GRID, [0] * len(GRID), H)
            T_best, r_best = best_of(const, GRID)
            M = max(5, T_best // 10)
            # offline optimum over two-phase schedules (switch time on a fine grid)
            r1, r2, ts = [], [], []
            for a in GRID:
                for b in GRID:
                    for f in np.linspace(0.05, 0.95, 19):
                        r1.append(a); r2.append(b); ts.append(int(f * T_best))
            two = best_of(sim_cols(P, P.EA, r1, r2, ts, H), r1, r2, ts)
            # receding horizon, constant-r plans (the MPC of the main experiments, exact model)
            const_planner = lambda E, Hr: best_of(sim_cols(P, E, GRID, GRID, [0] * len(GRID), Hr), GRID)
            T_mpc, tr1 = run_policy(P, H, M, const_planner)
            # receding horizon, two-phase plans
            T_mpc2, tr2 = run_policy(P, H, M, lambda E, Hr: plan_two_phase(P, E, Hr, M))
            row = dict(seed=seed, scaleB=sc, r_best=r_best, T_best=T_best, M=M,
                       two_phase_opt=dict(T=two[0], r1=two[1], r2=two[2], ts=two[3], regret=two[0] / T_best),
                       mpc_const=dict(T=T_mpc, regret=T_mpc / T_best, trace=tr1),
                       mpc_two_phase=dict(T=T_mpc2, regret=T_mpc2 / T_best, trace=tr2))
            out.append(row)
            print(f"seed={seed} scaleB={sc:.2f}: r*={r_best} T*={T_best} | offline two-phase {two[1]}->{two[2]} at {two[3]}: "
                  f"{two[0] / T_best:.3f} | MPC const {T_mpc / T_best:.3f} | MPC two-phase {T_mpc2 / T_best:.3f} trace {tr2[:5]}",
                  flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'two_phase.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
