"""Day-1 laziness calibration: kernel drift along a P_lin2-chosen run as a function of alpha.

    python scripts/calibrate.py --exp E1 --alphas 0.003 0.01 0.03 0.1 0.3 1 3
    python scripts/calibrate.py --exp E2 --alphas 0.1 0.2 0.3 0.5 1 2 3 5 10 20 50 100
    python scripts/calibrate.py --exp E1W --alphas 3 10 30 --width 8      # width sweep must stay lazy
    python scripts/calibrate.py --exp E1N --alphas 1 0.03 --batch 32      # is the band reachable with SGD?
For each alpha: phase A, K_A, P_lin2 -> r_lin, one real run at r_lin, drift of the subset / full kernel.
Writes calib.json; 'chosen' holds one distinct alpha per drift target (closest on the log scale, assigned in
order of drift), which make_jobs.py --calib calib.json then uses. E2 needs 6 + 2 distinct alphas (P0 and the
two P3 settings), hence the denser alpha list. Inspect the table before accepting the choice.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from rmpc import grids
from rmpc.config import JobConfig
from rmpc.problem import TwoTaskProblem
from rmpc.trainer import train_A, run_lanes
from rmpc.linsim import predict_curve, choose
from rmpc.controllers import Fixed
from rmpc.ntk import rel_change
from rmpc.utils import pick_device, configure_torch, set_seed, write_json_atomic

TARGETS = {'E1': [0.02, 0.07, 0.2, 0.5, 1.0], 'E2': [0.02, 0.05, 0.1, 0.3, 0.5, 0.8]}
SUBSET_P1 = {'E2': [0.05, 0.3, 0.8]}            # E2 alphas reused by the P1 extensions (drift targets)
EXTRA_P3 = {'E2': [0.15, 0.4]}                  # two more E2 alphas for P3, distinct from the P0 ones
POOR_MATCH = math.log(1.5)                      # warn when the chosen drift is off by more than 1.5x


class Ctx:
    pass


def drift_of(r):
    return r['drift_full'] if r.get('drift_full') is not None else r['drift_sub']


def assign(rows, targets, exclude=()):
    """One distinct calibration run per drift target, closest on the log scale. Runs are sorted by drift and
    matched to the sorted targets in the same order (dynamic programming), so two targets never share an alpha.
    Returns {target: row}; with fewer runs than targets the worst-matched targets stay unassigned."""
    cand = sorted((r for r in rows if r['alpha'] not in exclude), key=lambda r: (drift_of(r), -r['alpha']))
    tg = sorted(targets)
    n, m = len(cand), len(tg)
    cost = lambda i, j: abs(math.log(max(drift_of(cand[j]), 1e-6)) - math.log(tg[i]))
    skip = 1e3                                   # an unassigned target costs more than any match
    f = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        f[i][0] = i * skip
        for j in range(1, n + 1):
            f[i][j] = min(f[i][j - 1], f[i - 1][j - 1] + cost(i - 1, j - 1), f[i - 1][j] + skip)
    out, i, j = {}, m, n
    while i > 0 and j > 0:
        if f[i][j] == f[i - 1][j - 1] + cost(i - 1, j - 1):
            out[tg[i - 1]] = cand[j - 1]
            i, j = i - 1, j - 1
        elif f[i][j] == f[i][j - 1]:
            j -= 1
        else:
            i -= 1
    return out


def suggest(rows, target):
    """Alpha with drift ~ target by log-log interpolation between the two calibration runs around it (a hint)."""
    pts = sorted((math.log(r['alpha']), math.log(max(drift_of(r), 1e-6))) for r in rows)
    lt = math.log(target)
    for (a0, d0), (a1, d1) in zip(pts, pts[1:]):
        if min(d0, d1) <= lt <= max(d0, d1) and d0 != d1:
            return round(math.exp(a0 + (lt - d0) * (a1 - a0) / (d1 - d0)), 3)
    return None


def warn_rows(rows, exp):
    """Flag runs that cannot be trusted and a drift that does not fall with alpha (the assignment assumes it does)."""
    cap = (grids.e2(1.0) if exp == 'E2' else grids.e1('rot_sym', 1.0)).stepsA_max
    for r in rows:
        if r['stepsA'] >= cap - 1:
            print(f'WARNING alpha={r["alpha"]:g}: phase A hit stepsA_max={cap} (L_A={r["LA_end"]:.3g}); theta_A is not fitted')
        if r['T_real'] is None:
            print(f'WARNING alpha={r["alpha"]:g}: the run at r_lin={r["r_lin"]} never reached the band; drift measured at H')
    if len(rows) >= 4:
        from scipy.stats import spearmanr
        rho = spearmanr([r['alpha'] for r in rows], [drift_of(r) for r in rows]).correlation
        if not rho < -0.8:
            print(f'WARNING drift does not fall monotonically with alpha (Spearman {rho:.2f}); the chosen alphas are '
                  'not ordered and should not be used as they are')


def calib_one(cfg, dev):
    t0 = time.time()
    set_seed(cfg.seed)
    prob = TwoTaskProblem(cfg, dev)
    m = min(prob.nA, prob.nB)
    K0 = prob.kernel(prob.theta0)
    etaA = cfg.lr_safety * m / K0.lam_max()
    del K0
    LBt0 = float(0.5 * (prob.Y[~prob.isA] ** 2).sum() / prob.nB)
    thA, infoA = train_A(prob, etaA, cfg.stepsA_max, cfg.targetA_rel * LBt0)
    EA = prob.F(thA) - prob.Y
    LA0, LB0 = [float(x) for x in prob.losses(EA)]
    tauA, tauB = LA0 + cfg.tolA * LB0, cfg.tolB * LB0
    KA = prob.kernel(thA)
    eta = cfg.lr_safety * m / KA.lam_max()
    grid = list(cfg.r_grid)
    hits, viols = predict_curve(KA, EA.float(), grid, prob.isA, prob.nA, prob.nB, eta, cfg.H_max, tauA, tauB)
    r_lin, T_hat = choose(torch.tensor(hits), torch.tensor(viols), grid)
    H = cfg.H_max if T_hat is None else int(min(cfg.H_max, max(cfg.H_min, cfg.H_mult * T_hat)))
    ctx = Ctx()
    ctx.eta, ctx.tauA, ctx.tauB, ctx.LA0, ctx.LB0 = eta, tauA, tauB, LA0, LB0
    res, fin = run_lanes(prob, thA, [Fixed('cal', r_lin)], ctx, H, keep_final=True, batch_size=cfg.batch_size,
                         eval_every=cfg.eval_every)
    rng = np.random.default_rng(cfg.seed)
    sub = torch.as_tensor(np.concatenate([rng.choice(prob.idxA.cpu().numpy(), 30, replace=False),
                                          rng.choice(prob.idxB.cpu().numpy(), 30, replace=False)]), device=dev)
    d_sub = rel_change(prob.kernel(fin[0], sub), prob.kernel(thA, sub))
    d_full = rel_change(prob.kernel(fin[0]), KA) if prob.n * prob.C <= 40000 else None
    return dict(alpha=cfg.alpha, width=cfg.width, stepsA=infoA['steps'], LA_end=infoA['LA_end'], r_lin=r_lin,
                T_hat=T_hat, T_real=res[0]['T'], drift_sub=d_sub, drift_full=d_full,
                peak=res[0]['peak'], sec=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=['E1', 'E2', 'E1W', 'E1N'], required=True)
    ap.add_argument('--alphas', nargs='+', type=float, required=True)
    ap.add_argument('--width', type=int, default=None)
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default='calib.json')
    a = ap.parse_args()
    dev = pick_device(a.device)
    configure_torch(False)
    rows = []
    for al in a.alphas:
        if a.exp in ('E1', 'E1W', 'E1N'):
            cfg = grids.e1('rot_sym', al, width=a.width or 1024, seed=a.seed)
            cfg.batch_size, cfg.eval_every = a.batch, (5 if a.batch else 1)
        else:
            cfg = grids.e2(al, k=a.width or 16, seed=a.seed)
        row = calib_one(cfg, dev)
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(f'\n{"alpha":>8s} {"drift_sub":>9s} {"drift_full":>10s} {"r_lin":>6s} {"T_hat":>6s} {"T_real":>7s} {"stepsA":>7s} {"sec":>6s}')
    for r in rows:
        df = f'{r["drift_full"]:.3f}' if r['drift_full'] is not None else '-'
        print(f'{r["alpha"]:8.3g} {r["drift_sub"]:9.3f} {df:>10s} {r["r_lin"]:6.2f} {str(r["T_hat"]):>6s} '
              f'{str(r["T_real"]):>7s} {r["stepsA"]:7d} {r["sec"]:6.0f}')
    warn_rows(rows, a.exp)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out.setdefault('runs', {})[a.exp] = rows
    if a.exp in TARGETS:
        chosen = out.setdefault('chosen', {})
        by_target = out.setdefault('targets', {})
        main = assign(rows, TARGETS[a.exp])
        chosen[a.exp] = sorted(r['alpha'] for r in main.values())
        by_target[a.exp] = {str(t): r['alpha'] for t, r in sorted(main.items())}
        report = [(a.exp, t, main.get(t)) for t in TARGETS[a.exp]]
        if a.exp in SUBSET_P1:
            chosen[f'{a.exp}_P1'] = sorted(main[t]['alpha'] for t in SUBSET_P1[a.exp] if t in main)
        if a.exp in EXTRA_P3:
            extra = assign(rows, EXTRA_P3[a.exp], exclude=set(chosen[a.exp]))
            chosen[f'{a.exp}_P3'] = sorted(r['alpha'] for r in extra.values())
            by_target[f'{a.exp}_P3'] = {str(t): r['alpha'] for t, r in sorted(extra.items())}
            report += [(f'{a.exp}_P3', t, extra.get(t)) for t in EXTRA_P3[a.exp]]
        print(f'\n{"list":>7s} {"target":>7s} {"alpha":>8s} {"drift":>7s}')
        for lst, t, r in report:
            if r is None:
                hint = suggest(rows, t)
                print(f'{lst:>7s} {t:7.3g} {"-":>8s} {"-":>7s}  UNASSIGNED: too few alphas'
                      + (f'; add one near {hint}' if hint else ''))
                continue
            d = drift_of(r)
            off = abs(math.log(max(d, 1e-6)) - math.log(t)) > POOR_MATCH
            hint = suggest(rows, t) if off else None
            print(f'{lst:>7s} {t:7.3g} {r["alpha"]:8.3g} {d:7.3f}' + ('  POOR MATCH' if off else '')
                  + (f'; an alpha near {hint} would fit better' if hint else ''))
        for k in sorted(chosen):
            if k.split('_')[0] == a.exp:
                print(f'chosen {k}: {chosen[k]}')
    write_json_atomic(a.out, out)


if __name__ == '__main__':
    main()
