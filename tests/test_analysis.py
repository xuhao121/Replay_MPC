"""The analysis code on synthetic results: every P0-P2 job of the real grid gets a fake result built so that each
pre-registered hypothesis should be SUPPORTED; aggregate.py, figures.py and gate G3 must run on them and give
those verdicts. Then single hypotheses are broken on purpose and must turn into REJECT. No training, < 1 min.
Run:  python tests/test_analysis.py      or   python -m pytest -q tests/test_analysis.py
"""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from rmpc import grids
from rmpc.config import R_GRID

R_STAR = {'rot_sym': 0.5, 'perm_sym': 0.5, 'rot_smallA': 0.3, 'rot_tightA': 0.8, 'rot_tinyA': 0.2,
          'rot_bigA': 0.65, 'rot_tolA02': 0.55, 'rot_tolA06': 0.4}
E1_DRIFT = {0.01: 0.9, 0.03: 0.5, 0.1: 0.25, 0.3: 0.07, 1.0: 0.02}
E1W_T = {16: 30000, 20: 50000, 24: 20000, 28: 12000, 32: 9000, 40: 6000, 48: 5000, 64: 4000, 128: 3000,
         256: 2500, 1024: 2000}


def fake(cfg, rng):
    """A result dict in the format of rmpc/job.py with V-shaped T(r), exact P_lin2 in the lazy regime, ..."""
    lazy_alpha = sorted(grids.ALPHAS['E2'])[3:]
    if cfg.exp in ('E1', 'E1N'):
        drift = E1_DRIFT.get(cfg.alpha, 0.05)
    elif cfg.exp == 'E2':
        drift = 0.05 if cfg.alpha in lazy_alpha else 0.5
    else:
        drift = 0.01
    drift *= float(np.exp(rng.normal(0, 0.05)))
    lazy = drift < 0.2
    r_star = R_STAR.get(cfg.tag, 0.5)
    if cfg.exp == 'E1N':
        r_star = {500: 0.5, 128: 0.55, 32: 0.6}[cfg.batch_size]
    r_star = round(min(0.9, max(0.1, r_star + 0.05 * rng.integers(-1, 2))), 2)
    T0 = {'E1': 800, 'E2': 2000, 'E1N': 1500}.get(cfg.exp, E1W_T.get(cfg.width))
    grid = list(R_GRID)
    H = 20000 if cfg.exp != 'E1W' else 100000
    if T0 is None:                                               # E1W below the threshold: never in the band
        T = [None] * len(grid)
    else:
        T = [int(T0 * (1 + 2.0 * abs(r - r_star))) for r in grid]
    ok = [t for t in T if t is not None]
    T_best = min(ok) if ok else None
    r_best = grid[T.index(T_best)] if ok else None
    scale = 1.0 if lazy else 0.4
    r_lin = r_best if (lazy or r_best is None) else round(min(0.95, r_best + 0.05 * rng.integers(-1, 2)), 2)
    T_lin = [(-1 if t is None else int(t * scale)) for t in T]
    if r_best is not None and not lazy:                          # make the linear model's valley sit at r_lin
        T_lin[grid.index(r_lin)] = int(min(t for t in T_lin if t > 0) * 0.99)
    T_hat = None if r_best is None else T_lin[grid.index(r_lin)]
    spread = 0.004 if lazy else 0.03
    test = [None if t is None else dict(acc_A=0.9 + rng.uniform(0, spread), acc_B=0.88 + rng.uniform(0, spread),
                                        mse_A=0.1, mse_B=0.12) for t in T]
    step = 0.001 if cfg.model == 'mlp' else 0.03
    res = dict(job_id=cfg.job_id(), cfg=cfg.to_dict(), problem=dict(kernel='analytic', nA=cfg.nA, nB=cfg.nB, C=10),
               phaseA=dict(tauA=0.05, tauB=0.1), oracle=dict(r=grid, T=T, r_best=r_best, T_best=T_best, test=test),
               plin2_curve=dict(r=grid, T_lin=T_lin), r_lin=r_lin, T_hat=T_hat, H=H, M=max(20, (T_hat or H) // 10),
               drift=dict(sub_best=drift, full_best=drift), time_sec=60.0,
               timing=dict(step_sec=step, KA_sec=50 * step, plin2_sec=300 * step, KsA_sec=step), methods={})
    if T_best is None:
        res['drift'] = {}
    reg = dict(mpc=0.965 if lazy else 0.99, trig=1.0 if lazy else 1.01, replan=0.96 if lazy else 1.12,
               mpc2=0.88 if lazy else 0.97)
    for m in cfg.methods:
        if m == 'oracle':
            continue
        if m == 'plin2':
            Tm = T[grid.index(r_lin)] if r_best is not None else None
            res['methods'][m] = dict(r=r_lin, T=Tm, regret=(Tm / T_best) if Tm else None)
            continue
        if T_best is None or m == 'normbal':
            res['methods'][m] = dict(T=None, regret=None)
            continue
        kind = 'mpc2' if m.startswith('mpc2') else 'trig' if m.startswith('trig') else \
            'replan' if m == 'replan_only' else 'mpc'
        noise = float(np.exp(rng.normal(0, 0.004)))
        Tm = int(T_best * reg[kind] * noise)
        trig = 'drift' if m.startswith(('mpc_', 'mpc2', 'trig')) else 'prederr' if m.startswith('mpcpe') else 'none'
        res['methods'][m] = dict(T=Tm, regret=Tm / T_best, n_relin=0 if lazy else 3, n_plans=10, trigger=trig,
                                 ctrl_sec=(200 if kind == 'mpc2' else 3) * step * 100,
                                 plans2=[(0, 0.2, 0.7, 40)] if kind == 'mpc2' else [],
                                 trace=[(0, r_lin), (Tm // 2, r_best)], test=dict(acc_A=0.9, acc_B=0.88))
    return res


def write_results(jobs, d, seed=0):
    rng = np.random.default_rng(seed)
    for j in jobs:
        out = fake(j, rng)
        os.makedirs(os.path.join(d, j.exp), exist_ok=True)
        json.dump(out, open(os.path.join(d, j.exp, f'{j.job_id()}.json'), 'w'))
        arrs = {}
        for i in range(len(R_GRID)):
            arrs[f'or_t_{i}'] = np.arange(0, 100, 10); arrs[f'or_LA_{i}'] = np.linspace(1, 0.1, 10)
            arrs[f'or_LB_{i}'] = np.linspace(1, 0.05, 10)
        for m in j.methods:
            arrs[f'm_{m}_t'] = np.arange(0, 100, 10); arrs[f'm_{m}_LA'] = np.linspace(1, 0.1, 10)
            arrs[f'm_{m}_LB'] = np.linspace(1, 0.05, 10)
        np.savez(os.path.join(d, j.exp, f'{j.job_id()}.npz'), **arrs)


def verdicts(results):
    import aggregate
    from common import load_results, method_table, job_table
    jobs = load_results(results)
    df, jt = method_table(jobs), job_table(jobs)
    return df, jt, {o['id']: o for o in aggregate.confirmatory(df, jt)}


def test_synthetic_pipeline():
    jobs = grids.build(['P0', 'P1', 'P2'])
    with tempfile.TemporaryDirectory() as d:
        res, summ = os.path.join(d, 'results'), os.path.join(d, 'results', 'summary')
        write_results(jobs, res)
        env = dict(os.environ, RMPC_JOBS=os.path.join(d, 'jobs.jsonl'), RMPC_RESULTS=res)
        with open(env['RMPC_JOBS'], 'w') as f:
            for j in jobs:
                f.write(json.dumps(dict(j.to_dict(), owner='m1')) + '\n')
        for cmd in (['analysis/aggregate.py', '--results', res, '--out', summ],
                    ['analysis/figures.py', '--results', res, '--out', os.path.join(d, 'figs')]):
            p = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True)
            assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        tests = {o['id']: o for o in json.load(open(os.path.join(summ, 'tests.json')))}
        want = ['H1', 'H2a', 'H2b', 'H2c', 'H2d', 'H3', 'H2f', 'H5', 'H6a', 'H6b']
        assert sorted(tests) == sorted(want), sorted(tests)
        bad = {k: (v['verdict'], v['value']) for k, v in tests.items() if v['verdict'] != 'support'}
        assert not bad, bad
        e2 = {o['id']: o['verdict'] for o in json.load(open(os.path.join(summ, 'tests_e2.json')))}
        assert 'H3' not in e2 and e2['H1'] == 'support', e2
        md = open(os.path.join(summ, 'summary.md')).read()
        for part in ('wall-clock regret', 'Exploratory', 'threshold sensitivity', 'ratio rule', 'E1W', 'E1N'):
            assert part in md, part
        figs = sorted(os.listdir(os.path.join(d, 'figs')))
        assert all(f'fig{k}' in ' '.join(figs) for k in range(1, 8)), figs
        p = subprocess.run([sys.executable, 'analysis/gate_check.py', 'G3', '--summary', summ], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        assert p.returncode == 0 and '10 pre-registered hypotheses evaluated (10 support' in p.stdout, p.stdout
        # break hypotheses one at a time: each must become REJECT
        import aggregate
        df, jt, _ = verdicts(res)
        d2 = df.copy()
        sel = d2.method == 'mpcpe_0.2'
        d2.loc[sel, 'T'] = d2.loc[sel, 'T'] * 1.2                     # Jacobian-free trigger 20% slower
        assert {o['id']: o for o in aggregate.confirmatory(d2, jt)}['H2f']['verdict'] == 'reject'
        d3 = df.copy()
        sel = (d3.method == 'replan_only') & (d3.regime == 'rich')
        d3.loc[sel, 'T'] = d3.loc[sel, 'T'] * 0.8                     # stale kernel faster: H2c the other way
        assert {o['id']: o for o in aggregate.confirmatory(d3, jt)}['H2c']['verdict'] == 'reject'
        j2 = jt.copy()
        j2.loc[j2.regime == 'lazy', 'valley_accB_range'] = 0.05     # schedule changes generalisation
        assert {o['id']: o for o in aggregate.confirmatory(df, j2)}['H5']['verdict'] == 'reject'
        j3 = jt.copy()
        j3.loc[j3.regime == 'rich', 'argmin_offset'] = 4              # valley moves in the rich regime
        assert {o['id']: o for o in aggregate.confirmatory(df, j3)}['H1']['verdict'] == 'reject'
        d4 = df.copy()
        sel = d4.method == 'mpc2_0.05'
        d4.loc[sel, 'T'] = d4.loc[sel, 'T'] * 1.2                     # two-phase plans slower
        assert {o['id']: o for o in aggregate.confirmatory(d4, jt)}['H3']['verdict'] == 'reject'


if __name__ == '__main__':
    test_synthetic_pipeline()
    print('analysis tests passed')
