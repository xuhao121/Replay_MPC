"""One job: a (setting, seed) pair.

Pipeline
  1. phase A: GD on task A alone from theta0                         -> theta_A, band (tau_A, tau_B)
  2. K_A = NTK at theta_A (both tasks), eta = 0.9 min(nA, nB) / lambda_max(K_A)
  3. P_lin2: linearised prediction of T(r) for the whole grid       -> r_lin, T_hat, horizon H, check interval M
  4. oracle: real training for every r on the grid (one stacked run) -> T(r), r_best, T_best
  5. methods: probe_swap, MPC variants, gradient-statistics controllers (real training, same eta and H)
  6. mechanism: kernel drift along the best fixed-r run, test metrics at T for every lane
  7. cost: wall time of one training step of a single lane, of K_A, of the P_lin2 decision (grid simulated up to
     the first predicted arrival) and of each online controller's own work (kernels, plans), for the wall-clock
     regret (timing only: no result depends on it)
Writes <out_dir>/<job_id>.json (summary) and <job_id>.npz (curves, test outputs at T).
"""
import json, math, os, sys, time, traceback
import numpy as np
import torch

from .config import JobConfig, parse_method
from .problem import TwoTaskProblem
from .trainer import train_A, run_lanes
from .linsim import predict_curve, choose, plan, lam_max_weighted, sample_weights
from .ntk import rel_change, Kernel
from .controllers import Fixed, build_controllers
from .utils import (Timer, set_seed, pick_device, configure_torch, write_json_atomic, savez_atomic,
                    host_tag, gpu_name, sync)

TIMING_STEPS = 100          # steps of the single-lane timing run (after a 5-step warm-up)


class Ctx:
    pass


def _clocked(ctrl, device):
    """Accumulate the wall time a controller spends in its own hooks (start, on_check: kernels, plans,
    one-window predictions) in ctrl.ctrl_sec. Wraps the bound methods of this instance only."""
    ctrl.ctrl_sec = 0.0
    for name in ('start', 'on_check'):
        f = getattr(ctrl, name)

        def g(*args, _f=f, **kw):
            sync(device)
            t0 = time.time()
            try:
                return _f(*args, **kw)
            finally:
                sync(device)
                ctrl.ctrl_sec += time.time() - t0
        setattr(ctrl, name, g)
    return ctrl


def _lam_block(mat):
    if mat.shape[0] <= 4000:
        return float(torch.linalg.eigvalsh(mat.double())[-1])
    return Kernel(mat, 'full', 0, 0).lam_max()


def edge_mult(KA, prob, eta, r, target):
    """mu with eta * mu * lambda_max(K_A W_r) = target, i.e. the largest stable step for fixed r (x 0.9)."""
    w = sample_weights(prob.isA, prob.nA, prob.nB, r, 1.0 - r)
    return target / (eta * lam_max_weighted(KA, w))


def stability_cap(KA, prob, eta, s, target, iters=30):
    """Largest replay weight cA with eta * lambda_max(K_A W(cA, s)) <= target (bisection)."""
    f = lambda cA: eta * lam_max_weighted(KA, sample_weights(prob.isA, prob.nA, prob.nB, cA, s))
    if f(0.0) > target:
        return 0.0
    hi = 1.0
    while f(hi) <= target and hi < 1e6:
        hi *= 2.0
    lo = 0.0 if hi == 1.0 else hi / 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) <= target else (lo, mid)
    return lo


def run_job(cfg: JobConfig, out_dir, device=None, log=print):
    t_start = time.time()
    device = pick_device(device)
    configure_torch(cfg.tf32)
    set_seed(cfg.seed)
    timer = Timer(device)
    jid = cfg.job_id()
    os.makedirs(out_dir, exist_ok=True)

    with timer('setup'):
        prob = TwoTaskProblem(cfg, device)
    nA, nB, C = prob.nA, prob.nB, prob.C
    m = min(nA, nB)
    grid = [float(r) for r in cfg.r_grid]
    log(f'[{jid}] n={prob.n} P={prob.P} kernel={prob.kind} device={device}')

    # ------------------------------------------------------------------ phase A
    with timer('kernel_full'):
        if cfg.etaA_rule == 'pilot':
            K0 = prob.kernel(prob.theta0)
            lam0 = K0.lam_max()
            etaA = cfg.lr_safety * m / lam0
        else:
            K0 = prob.kernel(prob.theta0, prob.idxA)
            lam0 = K0.lam_max()
            etaA = cfg.lr_safety * nA / lam0
        del K0
    LB_theta0 = float(0.5 * (prob.Y[~prob.isA] ** 2).sum() / nB)        # F(theta0) = 0 by centring
    target = cfg.targetA_rel * LB_theta0
    with timer('phaseA'):
        thA, infoA = train_A(prob, etaA, cfg.stepsA_max, target, momentum=0.9 if cfg.optA == 'momentum' else 0.0)
    EA = prob.F(thA) - prob.Y
    LA0, LB0 = [float(x) for x in prob.losses(EA)]
    tauA, tauB = LA0 + cfg.tolA * LB0, cfg.tolB * LB0
    t_prev = timer.t.get('kernel_full', 0.0)
    with timer('kernel_full'):
        KA = prob.kernel(thA)
        lamA = KA.lam_max()
    t_KA = timer.t['kernel_full'] - t_prev
    eta = cfg.lr_safety * m / lamA
    lamAA = _lam_block(prob.block_AA(KA))
    needs_cap = any(x in cfg.methods for x in ('normbal_lr', 'megaI'))
    stab_cap = stability_cap(KA, prob, eta, cfg.gstat_new_weight, cfg.edge_target) if needs_cap else None
    log(f'[{jid}] phase A: {infoA["steps"]} steps, L_A={LA0:.3g}, L_B={LB0:.3g}, eta={eta:.4g}')

    # ------------------------------------------------------------------ P_lin2
    sim_steps = cfg.H if cfg.H else cfg.H_max
    t_prev = timer.t.get('plan', 0.0)
    with timer('plan'):
        hits, viols = predict_curve(KA, EA.float(), grid, prob.isA, nA, nB, eta, sim_steps, tauA, tauB)
    t_curve = timer.t['plan'] - t_prev
    r_lin, T_hat = choose(torch.tensor(hits), torch.tensor(viols), grid)
    # the P_lin2 decision alone (what a practitioner runs): simulate the grid only up to the first predicted
    # arrival. The full curve above (every r until it arrives) is a diagnostic for the analysis and can cost
    # several times more; the wall-clock regret charges the decision, not the curve.
    t_prev = timer.t.get('plan_decide', 0.0)
    with timer('plan_decide'):
        p_dec = plan(KA, EA.float(), grid, prob.isA, nA, nB, eta, sim_steps, tauA, tauB)
    t_plin2 = timer.t['plan_decide'] - t_prev
    if p_dec['r'] != r_lin:
        log(f'[{jid}] warning: stop-first P_lin2 picked r={p_dec["r"]}, full curve r={r_lin}')
    if cfg.H:
        H = cfg.H
    else:
        H = cfg.H_max if T_hat is None else int(min(cfg.H_max, max(cfg.H_min, cfg.H_mult * T_hat)))
    M = max(cfg.M_min, int((T_hat if T_hat is not None else H) / cfg.n_checks))
    if cfg.batch_size > 0 and cfg.eval_every > 1:
        M = int(math.ceil(M / cfg.eval_every) * cfg.eval_every)
    rng = np.random.default_rng(cfg.seed)
    iA, iB = prob.idxA.cpu().numpy(), prob.idxB.cpu().numpy()
    ds = min(cfg.drift_subset, nA, nB)
    sub = torch.as_tensor(np.concatenate([rng.choice(iA, ds, replace=False), rng.choice(iB, ds, replace=False)]),
                          device=device)
    t_prev = timer.t.get('kernel_sub', 0.0)
    with timer('kernel_sub'):
        KsA = prob.kernel(thA, sub)
    t_KsA = timer.t['kernel_sub'] - t_prev

    ctx = Ctx()
    ctx.timer = timer
    ctx.eta, ctx.H, ctx.M = eta, H, M
    ctx.tauA, ctx.tauB, ctx.LA0, ctx.LB0 = tauA, tauB, LA0, LB0
    ctx.isA, ctx.nA, ctx.nB = prob.isA, nA, nB
    ctx.r_grid, ctx.r_lin, ctx.T_hat = grid, r_lin, T_hat
    ctx.K_A, ctx.Ks_A, ctx.E_A = KA, KsA, EA.detach().float()
    ctx.weight_cap = stab_cap

    def kernel_full(params):
        with timer('kernel_full'):
            return prob.kernel(params)

    def kernel_sub(params):
        with timer('kernel_sub'):
            return prob.kernel(params, sub)
    ctx.kernel_full, ctx.kernel_sub = kernel_full, kernel_sub

    common = dict(record_every=cfg.record_every, batch_size=cfg.batch_size, eval_every=cfg.eval_every,
                  lane_chunk=cfg.lane_chunk)
    npz = {}

    # ------------------------------------------------------------------ oracle grid
    bands = [(tauA, tauB)]
    if cfg.dest_tight:
        bands.append((LA0 + cfg.tight_tolA * LB0, cfg.tight_tolB * LB0))
    lanes = [Fixed(f'fixed_{r}', r) for r in grid]
    mu = []
    if 'oracle_edge' in cfg.methods:
        mu = [edge_mult(KA, prob, eta, r, cfg.edge_target) for r in grid]
        lanes += [Fixed(f'edge_{r}', r, mult=u) for r, u in zip(grid, mu)]
    with timer('oracle'):
        res_all, finals = run_lanes(prob, thA, lanes, ctx, H, bands=bands, snap_outputs=True, keep_final=True,
                                    seed=cfg.seed * 1000 + 1, **common)
    res_or, res_edge = res_all[:len(grid)], res_all[len(grid):]
    T_real = {r: ln['T'] for r, ln in zip(grid, res_or)}
    ok = {r: T for r, T in T_real.items() if T is not None}
    r_best = min(ok, key=ok.get) if ok else None
    T_best = ok[r_best] if ok else None
    oracle = dict(r=grid, T=[T_real[r] for r in grid], peak=[ln['peak'] for ln in res_or],
                  T_bands=[ln['T_bands'] for ln in res_or], diverged=[ln['diverged'] for ln in res_or],
                  r_best=r_best, T_best=T_best, test=[], test_tight=[])
    for i, ln in enumerate(res_or):
        npz[f'or_t_{i}'] = np.asarray(ln['rec_t'], np.int32)
        npz[f'or_LA_{i}'] = np.asarray(ln['LA'], np.float32)
        npz[f'or_LB_{i}'] = np.asarray(ln['LB'], np.float32)
        for b, s in enumerate(ln['snaps']):
            if s is None:
                (oracle['test'] if b == 0 else oracle['test_tight']).append(None)
                continue
            for k in ('A', 'B'):
                npz[f'or_F{k}_{i}_b{b}'] = s.pop(f'F_{k}')
            (oracle['test'] if b == 0 else oracle['test_tight']).append(s)
    oracle_edge = None
    if res_edge:
        TE = [ln['T'] for ln in res_edge]
        okE = {r: T for r, T in zip(grid, TE) if T is not None}
        rE = min(okE, key=okE.get) if okE else None
        oracle_edge = dict(r=grid, mu=mu, T=TE, peak=[ln['peak'] for ln in res_edge],
                           diverged=[ln['diverged'] for ln in res_edge], r_best=rE, T_best=okE[rE] if okE else None,
                           test=[ln['snaps'][0] and {k: v for k, v in ln['snaps'][0].items() if not k.startswith('F_')}
                                 for ln in res_edge])
        for i, ln in enumerate(res_edge):
            npz[f'edge_LA_{i}'] = np.asarray(ln['LA'], np.float32)
            npz[f'edge_LB_{i}'] = np.asarray(ln['LB'], np.float32)
            npz[f'edge_t_{i}'] = np.asarray(ln['rec_t'], np.int32)
    log(f'[{jid}] oracle: r_best={r_best} T_best={T_best}  r_lin={r_lin} T_hat={T_hat} H={H} M={M}' +
        (f'  edge: r_best={oracle_edge["r_best"]} T_best={oracle_edge["T_best"]}' if oracle_edge else ''))

    # drift along the best fixed-r run (laziness of this configuration)
    drift = {}
    if r_best is not None:
        th_best = finals[grid.index(r_best)]
        drift['sub_best'] = rel_change(kernel_sub(th_best), KsA)
        if prob.n * (C if KA.kind == 'full' else 1) <= cfg.full_drift_max_rows:
            Kb = kernel_full(th_best)
            drift['full_best'] = rel_change(Kb, KA)
            del Kb
    del finals

    # wall time of one training step of a single lane: the unit of the wall-clock regret (a short fixed-r run
    # from theta_A after a warm-up; the lane stops early only if it enters the band first)
    r_time = r_best if r_best is not None else r_lin
    n_time = int(max(1, min(TIMING_STEPS, H)))
    run_lanes(prob, thA, [Fixed('timing_warmup', r_time)], ctx, min(5, n_time), seed=cfg.seed * 1000 + 4, **common)
    t_prev = timer.t.get('step_timing', 0.0)
    with timer('step_timing'):
        res_t, _ = run_lanes(prob, thA, [Fixed('timing', r_time)], ctx, n_time, seed=cfg.seed * 1000 + 4, **common)
    steps_t = max(1, res_t[0]['t_end'] or n_time)
    timing = dict(step_sec=(timer.t['step_timing'] - t_prev) / steps_t, step_steps=steps_t, r=r_time,
                  KA_sec=t_KA, plin2_sec=t_plin2, plin2_curve_sec=t_curve, KsA_sec=t_KsA)

    def regret(T):
        return (T / T_best) if (T is not None and T_best) else None

    def regret_edge(T):
        TbE = oracle_edge['T_best'] if oracle_edge else None
        return (T / TbE) if (T is not None and TbE) else None

    methods = {}
    # ------------------------------------------------------------------ plin2 / probe
    if 'plin2' in cfg.methods:
        methods['plin2'] = dict(r=r_lin, T=T_real.get(r_lin), T_pred=T_hat, regret=regret(T_real.get(r_lin)))
    if 'probe_swap' in cfg.methods:
        steps = max(30, int(cfg.probe_frac * (T_hat if T_hat else H)))
        with timer('probe'):
            res_p, fin_p = run_lanes(prob, thA, [Fixed('probe', r_lin)], ctx, steps, keep_final=True,
                                     seed=cfg.seed * 1000 + 2, **common)
            Kp = kernel_full(fin_p[0])
            p = plan(Kp, ctx.E_A, grid, prob.isA, nA, nB, eta, H, tauA, tauB)
        methods['probe_swap'] = dict(r=p['r'], T=T_real.get(p['r']), regret=regret(T_real.get(p['r'])),
                                     probe_steps=steps, probe_drift_full=rel_change(Kp, KA))
        del Kp, fin_p

    # ------------------------------------------------------------------ online controllers
    mpc_ctrls, gstat_ctrls = build_controllers([m_ for m_ in cfg.methods
                                                if parse_method(m_)['kind'] not in ('oracle', 'plin2', 'probe_swap')],
                                               ctx, cfg)
    for c in mpc_ctrls:
        _clocked(c, device)
    mpc_by_name = {c.name: c for c in mpc_ctrls}
    lane_results = []
    gsize = cfg.mpc_group if cfg.mpc_group > 0 else max(1, len(mpc_ctrls))
    for g0 in range(0, len(mpc_ctrls), gsize):
        grp = mpc_ctrls[g0:g0 + gsize]
        with timer('mpc'):
            res, _ = run_lanes(prob, thA, grp, ctx, H, seed=cfg.seed * 1000 + 3 + g0, **common)
        for c in grp:
            c.release()
        lane_results += res
    if gstat_ctrls:
        with timer('gradstat'):
            res, _ = run_lanes(prob, thA, gstat_ctrls, ctx, H, split_grads=True, seed=cfg.seed * 1000 + 99,
                               **{**common, 'lane_chunk': 0})
        lane_results += res
    for ln in lane_results:
        name = ln['name']
        npz[f'm_{name}_t'] = np.asarray(ln['rec_t'], np.int32)
        npz[f'm_{name}_LA'] = np.asarray(ln['LA'], np.float32)
        npz[f'm_{name}_LB'] = np.asarray(ln['LB'], np.float32)
        snap = ln['snaps'][0]
        methods[name] = dict(T=ln['T'], regret=regret(ln['T']), regret_edge=regret_edge(ln['T']), peak=ln['peak'],
                             diverged=ln['diverged'], t_end=ln['t_end'], test=snap, **ln['ctrl'])
        if name in mpc_by_name:
            methods[name]['ctrl_sec'] = round(mpc_by_name[name].ctrl_sec, 4)

    out = dict(job_id=jid, cfg=cfg.to_dict(), host=host_tag(), gpu=gpu_name(device), torch=torch.__version__,
               problem=dict(nA=nA, nB=nB, C=C, P=prob.P, kernel=prob.kind, n_test=int(prob.Xte['A'].shape[0])),
               phaseA=dict(**infoA, LA0=LA0, LB0=LB0, tauA=tauA, tauB=tauB, target=target, lam0=lam0),
               eta=eta, lamA=lamA, lamAA=lamAA, stab_cap=stab_cap, H=H, M=M,
               plin2_curve=dict(r=grid, T_lin=hits, viol=viols), r_lin=r_lin, T_hat=T_hat,
               oracle=oracle, oracle_edge=oracle_edge, drift=drift, methods=methods, timing=timing,
               time_sec=round(time.time() - t_start, 1), timer=timer.as_dict())
    write_json_atomic(os.path.join(out_dir, f'{jid}.json'), out)
    savez_atomic(os.path.join(out_dir, f'{jid}.npz'), **npz)
    log(f'[{jid}] done in {out["time_sec"]} s; ' +
        ', '.join(f'{k}={v["regret"]:.3f}' for k, v in methods.items() if v.get('regret') is not None))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='run one job from a JSON config (or a jobs.jsonl line)')
    ap.add_argument('config')
    ap.add_argument('--out', default='results/manual')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    cfg = JobConfig.from_dict(json.load(open(a.config)))
    run_job(cfg, a.out, a.device)
