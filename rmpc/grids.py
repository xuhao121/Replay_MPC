"""Experiment grids and cost estimates.

    P0  main line: E1 (4 scenarios x 5 alphas) and E2 (6 alphas), seeds 0-2
    P1  more settings: 4 new E1 scenarios, E1 widths 256 / 4096, E1N (SGD noise), E2 extensions
    P2  E1W (width sweep at the interpolation threshold)
    P3  filler: 2 more E2 alphas, ResNet k = 64, E1 with n = 10000; more seeds only on request
The unit of analysis is the setting (everything except the seed), so seeds beyond 0-2 add no settings: they are
only added when the queue runs dry (make_jobs.py --seeds-p3-e1 / --seeds-p3-e2 FROM TO).

Edit ALPHAS after the day-1 calibration (scripts/calibrate.py writes the chosen alphas to calib.json, which
make_jobs.py --calib reads) or pass --alphas-e1 / --alphas-e2 to scripts/make_jobs.py.
"""
import json, math, os
from .config import JobConfig

ALPHAS = {
    'E1': [0.01, 0.03, 0.1, 0.3, 1.0],          # MLP h=1024: drift targets 1.0 0.5 0.2 0.07 0.02 (rich ... lazy)
    'E2': [0.3, 1.0, 2.0, 8.0, 18.0, 45.0],     # ResNet18-16: drift targets 0.8 0.5 0.3 0.1 0.05 0.02 (placeholders)
    'E1W': 10.0,                                # width sweep: lazy at every width (check drift at h=8)
    'E1N': [1.0, 0.03],                         # SGD noise: one lazy, one rich alpha
}
# 'E2_P1' (three of the E2 alphas: drift targets 0.8, 0.3, 0.05) and 'E2_P3' (two more: 0.4, 0.15) come from
# calib.json; without them they are derived from 'E2' by e2_subsets().
BENCH_ALPHA = {'E1': 0.1, 'E2': 3.0}            # reference jobs of scripts/bench.py (intermediate laziness)

E1_SCEN = {                                     # P0: symmetric baselines + two scenarios that move r* off 0.5
    'rot_sym':    dict(transform='rot90', nA=1000, nB=1000),
    'perm_sym':   dict(transform='perm', nA=1000, nB=1000),
    'rot_smallA': dict(transform='rot90', nA=500, nB=1500),
    'rot_tightA': dict(transform='rot90', nA=1000, nB=1000, tolA=0.01, tolB=0.15),
}
E1_SCEN_P1 = {                                  # P1: four more settings that spread r* over the grid
    'rot_tinyA':  dict(transform='rot90', nA=250, nB=1750),
    'rot_bigA':   dict(transform='rot90', nA=1500, nB=500),
    'rot_tolA02': dict(transform='rot90', nA=1000, nB=1000, tolA=0.02),
    'rot_tolA06': dict(transform='rot90', nA=1000, nB=1000, tolA=0.06),
}

E1_METHODS = ('oracle', 'plin2', 'mpc_0.05', 'mpcpe_0.2', 'mpc2_0.05', 'trig_0.05', 'replan_only', 'periodic',
              'normbal')
THRESHOLD_METHODS = ('mpc_0.02', 'mpc_0.1', 'mpcpe_0.1')       # appendix: threshold sensitivity (E1 rot_sym only)
E2_METHODS = ('oracle', 'plin2', 'mpc_0.05', 'mpcpe_0.2', 'trig_0.05', 'replan_only')
E1N_METHODS = ('oracle', 'plin2', 'mpc_0.05', 'mpcpe_0.2')
E1W_METHODS = ('oracle', 'plin2')
WIDTHS_E1W = [8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 128, 256, 1024]
E1W_BAND = dict(tolA=0.01, tolB=0.02)           # tight band only


def e1(scen, alpha, width=1024, seed=0, priority='P0', thresholds=False, **kw):
    d = dict(E1_SCEN[scen] if scen in E1_SCEN else E1_SCEN_P1[scen])
    d.setdefault('mpc_group', 4)                 # at most 4 MPC lanes (each may hold its own kernel) at a time
    d['methods'] = E1_METHODS + (THRESHOLD_METHODS if thresholds else ())
    d.update(kw)
    return JobConfig(exp='E1', tag=scen, priority=priority, seed=seed, dataset='mnist', model='mlp',
                     width=width, alpha=alpha, **d)


def e2(alpha, k=16, seed=0, transform='rot90', nA=500, nB=500, kernel='entk', priority='P0', methods=E2_METHODS,
       tag_extra='', **kw):
    d = dict(lane_chunk=5, jac_dtype='bfloat16', sample_chunk=500 if k >= 32 else 0, optA='momentum',
             targetA_rel=1e-2, stepsA_max=20000, H_max=12000, record_every=2)
    d.update(kw)
    return JobConfig(exp='E2', tag=f'{transform}_n{nA + nB}_{kernel}' + (f'_{tag_extra}' if tag_extra else ''),
                     priority=priority, seed=seed,
                     dataset='cifar10', transform=transform, nA=nA, nB=nB, model='resnet18k', width=k,
                     alpha=alpha, kernel=kernel, methods=methods, **d)


def e2_subsets(A):
    """(alphas of the P1 extensions, extra P3 alphas). With six E2 alphas sorted ascending (= drift targets 0.8,
    0.5, 0.3, 0.1, 0.05, 0.02) the P1 subset is the 1st, 3rd and 5th (drift 0.8, 0.3, 0.05) and the P3 alphas are
    the geometric means of the 2nd/3rd and 3rd/4th (drift about 0.4 and 0.15). Lists given in A win."""
    s = sorted(A['E2'])
    p1 = A.get('E2_P1') or (s[0:5:2] if len(s) >= 5 else s)
    if A.get('E2_P3') is not None:
        p3 = A['E2_P3']
    elif len(s) >= 4:
        p3 = [round(math.sqrt(s[1] * s[2]), 4), round(math.sqrt(s[2] * s[3]), 4)]
    else:
        p3 = []
    return sorted(p1), sorted(a for a in p3 if a not in s)


def build(phases, alphas=None, p3_seeds=None):
    """Job list of the given phases. p3_seeds = {'E1': (from, to), 'E2': (from, to)} adds those seeds (half-open
    range) to every main-line E1 setting (8 scenarios x 5 alphas, h = 1024) and every E2 rot90 / k = 16 / eNTK
    setting (P0 alphas + the P3 alphas)."""
    A = dict(ALPHAS)
    if alphas:
        A.update(alphas)
    a2p1, a2p3 = e2_subsets(A)
    S3 = dict(p3_seeds or {})
    jobs = []
    if 'P0' in phases:
        for scen in E1_SCEN:
            for a in A['E1']:
                for s in (0, 1, 2):
                    jobs.append(e1(scen, a, seed=s, thresholds=(scen == 'rot_sym')))
        for a in A['E2']:
            for s in (0, 1, 2):
                jobs.append(e2(a, seed=s))
    if 'P1' in phases:
        for scen in E1_SCEN_P1:                                  # four more main-line settings
            for a in A['E1']:
                for s in (0, 1, 2):
                    jobs.append(e1(scen, a, seed=s, priority='P1'))
        for w in (256, 4096):                                    # width as a second laziness knob
            for scen in ('rot_sym', 'perm_sym'):
                for a in A['E1']:
                    for s in (0, 1, 2):
                        jobs.append(e1(scen, a, width=w, seed=s, priority='P1'))
        for a in A['E1N']:                                       # Smith & Le: SGD noise scale g = eta N / B
            for B in (500, 128, 32):
                for s in range(5):
                    jobs.append(JobConfig(exp='E1N', tag=f'B{B}', priority='P1', seed=s, dataset='mnist',
                                          transform='rot90', nA=1000, nB=1000, model='mlp', width=1024, alpha=a,
                                          batch_size=B, eval_every=5, record_every=5, H_max=40000,
                                          methods=E1N_METHODS))
        for a in a2p1:                                           # E2 extensions at three laziness levels
            for s in (0, 1, 2):
                jobs.append(e2(a, seed=s, transform='neg', priority='P1'))
                jobs.append(e2(a, seed=s, kernel='pntk', priority='P1'))          # pNTK vs eNTK planner
                jobs.append(e2(a, k=32, seed=s, nA=1000, nB=1000, kernel='pntk', priority='P1'))
    if 'P2' in phases:
        for w in WIDTHS_E1W:                                     # Nakkiran: interpolation threshold in width
            for s in (0, 1, 2):
                jobs.append(JobConfig(exp='E1W', tag='tight', priority='P2', seed=s, dataset='mnist',
                                      transform='rot90', nA=1000, nB=1000, model='mlp', width=w, alpha=A['E1W'],
                                      methods=E1W_METHODS, H_max=100000, record_every=10, **E1W_BAND))
    if 'P3' in phases:                                           # filler, in the order it should run
        for a in a2p3:                                           # two more E2 settings (drift about 0.4, 0.15)
            for s in (0, 1, 2):
                jobs.append(e2(a, seed=s, priority='P3'))
        for a in a2p1:                                           # full-width ResNet18
            for s in (0, 1, 2):
                jobs.append(e2(a, k=64, seed=s, nA=1000, nB=1000, kernel='pntk', priority='P3', lane_chunk=2))
        for a in A['E1']:                                        # 5x more data (40 GB kernel, one MPC lane at a time)
            for s in (0, 1):
                jobs.append(e1('rot_sym', a, seed=s, priority='P3', nA=5000, nB=5000, mpc_group=1,
                               H_max=30000, full_drift_max_rows=0, record_every=5))
        if 'E2' in S3:                                           # seeds: only when the queue runs dry
            for a in sorted(A['E2']) + a2p3:
                for s in range(*S3['E2']):
                    jobs.append(e2(a, seed=s, priority='P3'))
        if 'E1' in S3:
            for s in range(*S3['E1']):
                for scen in list(E1_SCEN) + list(E1_SCEN_P1):
                    for a in A['E1']:
                        jobs.append(e1(scen, a, seed=s, priority='P3', thresholds=(scen == 'rot_sym')))
    for j in jobs:
        j.est_hours, j.est_mem_gb = estimate(j)
    return jobs


# ----------------------------------------------------------------------------- cost model
BENCH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bench.json')
DEFAULT_BENCH = {
    'E1_ref_hours': 0.13,   # E1 rot_sym, alpha 0.1, E1_METHODS (exclusive B200); prior, replaced by bench.py
    'E2_ref_hours': 0.49,   # E2 rot90, k = 16, nA = nB = 500, eNTK, E2_METHODS; prior, replaced by bench.py
}
# Rough relative cost of each part of a job (one ordinary MPC lane = 1). MLP: steps are nearly free, so the
# linearised planning dominates and mpc2 (1387 simulated columns per plan) outweighs everything else. ResNet:
# training dominates; the oracle trains 19 lanes and an MPC lane adds about three full eNTKs. Only ratios to
# the family's reference method list matter; bench.py measures the reference job itself.
METHOD_COST = {
    'mlp':       dict(base=2.5, oracle=2.5, oracle_edge=2.5, plin2=0.0, probe_swap=0.25, mpc=1.0, periodic=1.0,
                      mpc2=40.0, gstat=1.25),
    'resnet18k': dict(base=0.7, oracle=14.0, oracle_edge=14.0, plin2=0.0, probe_swap=0.2, mpc=1.15, periodic=1.36,
                      mpc2=3.0, gstat=2.0),
}
REF_METHODS = {'mlp': E1_METHODS, 'resnet18k': E2_METHODS}


def _kind(m):
    if m in ('oracle', 'oracle_edge', 'plin2', 'probe_swap', 'periodic'):
        return m
    if m.startswith('mpc2'):
        return 'mpc2'
    if m.startswith(('mpc', 'trig')) or m == 'replan_only':
        return 'mpc'
    return 'gstat'                                               # normbal, normbal_lr, megaI, fixed_<r>


def method_factor(model, methods):
    w = METHOD_COST[model]
    cost = lambda ms: w['base'] + sum(w[_kind(m)] for m in ms)
    return cost(methods) / cost(REF_METHODS[model])


def load_bench():
    b = dict(DEFAULT_BENCH)
    if os.path.exists(BENCH_PATH):
        b.update(json.load(open(BENCH_PATH)))
    return b


def estimate(cfg, bench=None):
    """(GPU hours on an exclusive B200, peak memory GB) - rough, rescaled by bench.json."""
    b = bench or load_bench()
    n = cfg.nA + cfg.nB
    rows = n * 10
    kern_gb = rows * rows * 4 / 1e9
    nm = sum(1 for m in cfg.methods if m.startswith(('mpc', 'trig', 'periodic', 'replan')))
    if cfg.model == 'mlp':
        s_n = n / 2000.0
        s_h = max(cfg.width / 1024.0, 0.25)
        hours = b['E1_ref_hours'] * (0.4 * s_n * s_h + 0.6 * s_n ** 2) * (cfg.H_max / 20000.0) ** 0.5
        hours *= method_factor('mlp', cfg.methods)
        group = min(cfg.mpc_group if cfg.mpc_group > 0 else nm, max(nm, 1))
        lanes = 38 if 'oracle_edge' in cfg.methods else 19
        mem = 3.0 + kern_gb * (group + 1) + lanes * n * cfg.width * 4 * 4 / 1e9
    else:
        s_n = n / 1000.0
        s_k = (cfg.width / 16.0) ** 2
        hours = b['E2_ref_hours'] * s_n * (0.7 * s_k + 0.3 * s_n) * method_factor('resnet18k', cfg.methods)
        if cfg.kernel == 'pntk':
            rows = n
            kern_gb = rows * rows * 4 / 1e9
        P = 11.2e6 * (cfg.width / 64.0) ** 2
        jac_gb = rows * P * (2 if cfg.jac_dtype == 'bfloat16' else 4) / 1e9
        lanes = cfg.lane_chunk if cfg.lane_chunk else 19
        per = min(n, cfg.sample_chunk) if cfg.sample_chunk else n
        act_gb = lanes * per * 1.6e6 * (cfg.width / 16.0) * 2 / 1e9
        mem = 3.0 + jac_gb + kern_gb * (nm + 1) + act_gb
    if cfg.batch_size > 0:
        hours *= 1.5
    return round(hours, 3), round(mem, 1)
