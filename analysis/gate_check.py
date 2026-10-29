"""Gate checks of the six-day plan.

    python analysis/gate_check.py G1     # before launch: calib.json (laziness range, E2 speed) and bench.json
    python analysis/gate_check.py G2     # after P0: completeness, failures, censoring, valley, lazy P_lin2, speed
    python analysis/gate_check.py G3     # P0-P2 complete (or failures documented) -> freeze the analysis code
Each line is PASS, CHECK (look at it, usually not blocking) or FAIL (fix before going on). The tests in
tests/ are run separately. Gates only look for implementation and resource problems: the pre-registered
criteria in analysis/aggregate.py are never changed because of what a gate shows.
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import load_results, method_table, job_table, LAZY_DRIFT, ROOT, SKIP        # noqa: E402
from rmpc import queue as Q, grids                                                       # noqa: E402
from rmpc.config import R_GRID                                                           # noqa: E402

out = {'FAIL': 0, 'CHECK': 0}


def say(status, msg):
    out[status] = out.get(status, 0) + 1
    print(f'{status:5s} {msg}')


def g1(calib_path, bench_path):
    if not os.path.exists(calib_path):
        say('FAIL', f'{calib_path} missing: run scripts/calibrate.py (E1 and E2) first')
        return
    c = json.load(open(calib_path))
    runs, chosen = c.get('runs', {}), c.get('chosen', {})
    drift = lambda r: r['drift_full'] if r.get('drift_full') is not None else r['drift_sub']
    for exp, need, n_want in (('E1', 2, 5), ('E2', 2, 6)):
        rows = {r['alpha']: r for r in runs.get(exp, [])}
        al = chosen.get(exp, [])
        if not al:
            say('FAIL', f'{exp}: no alphas chosen (calibrate.py --exp {exp})')
            continue
        d = [drift(rows[a]) for a in al]
        lazy, rich = sum(x < LAZY_DRIFT for x in d), sum(x >= LAZY_DRIFT for x in d)
        msg = f'{exp}: chosen alphas {al}, drift {[round(x, 3) for x in d]} -> {lazy} lazy, {rich} rich'
        say('PASS' if lazy >= need and rich >= need else 'FAIL', msg + f' (need >= {need} on each side of '
            f'{LAZY_DRIFT}; otherwise add smaller / larger alphas and calibrate again)')
        if len(set(al)) < n_want:
            say('CHECK', f'{exp}: {len(set(al))} distinct alphas for {n_want} drift targets (fewer settings than '
                f'planned; calibrate more alphas)')
        if exp == 'E2':
            p3 = chosen.get('E2_P3', [])
            if len(p3) < 2:
                say('CHECK', f'E2: {len(p3)} of 2 extra P3 alphas calibrated (drift 0.15, 0.4); make_jobs.py '
                    f'falls back to geometric means of neighbouring E2 alphas')
            T = [rows[a]['T_real'] for a in list(al) + [x for x in p3 if x in rows]]
            ok = all(t is not None and t < 10000 for t in T)
            say('PASS' if ok else 'FAIL', f'E2: steps to the band at the chosen alphas (P0, then P3) {T} (need < '
                f'10000; otherwise the E2 fallbacks of the plan: n = 250 + 250, tol_B = 0.2, k = 32 + pNTK)')
    w = runs.get('E1W', [])
    if w:
        aw = grids.ALPHAS['E1W']
        r = [x for x in w if x['alpha'] == aw]
        if r:
            dw = drift(r[0])
            say('PASS' if dw < LAZY_DRIFT else 'CHECK', f'E1W: drift at width {r[0]["width"]}, alpha {aw}: '
                f'{dw:.3f} (the width sweep should stay lazy; otherwise make_jobs.py --alpha-e1w <larger>)')
    else:
        say('CHECK', 'E1W not calibrated (optional): calibrate.py --exp E1W --alphas 3 10 30 --width 8')
    n = runs.get('E1N', [])
    if n:
        T = [x['T_real'] for x in n]
        say('PASS' if all(t is not None for t in T) else 'CHECK', f'E1N: SGD reaches the band: T = {T}')
    if not os.path.exists(bench_path):
        say('FAIL', f'{bench_path} missing: run scripts/bench.py single and concurrent first')
        return
    b = json.load(open(bench_path))
    for k, ref in grids.DEFAULT_BENCH.items():
        v = b.get(k)
        if v is None:
            say('FAIL', f'bench.json has no {k}')
            continue
        say('PASS' if v <= 2 * ref else 'CHECK', f'{k} = {v:.3f} h (plan assumed {ref}; above 2x -> cut P3 first '
            f'(incl. E1 n = 10000), then ResNet k = 32 in P1)')
    conc = b.get('concurrent_E1') or {}
    if conc:
        sp = {int(k): v.get('speedup') for k, v in conc.items()}
        say('PASS', f'concurrency (lanes: throughput gain) {sp}; use the lane count where the gain flattens')
    else:
        say('CHECK', 'no concurrency measurement (bench.py concurrent --procs 2 4 6 8)')


def phase_jobs(phases):
    jobs = [j for j in Q.load_jobs() if j.priority in phases]
    if not jobs:
        say('FAIL', f'no {"/".join(phases)} jobs in {Q.JOBS} (make_jobs.py)')
    return jobs


def completeness(jobs, label):
    done = [j for j in jobs if Q.is_done(j)]
    failed = [j for j in jobs if not Q.is_done(j) and Q.is_failed(j)]
    left = len(jobs) - len(done) - len(failed)
    say('PASS' if left == 0 else 'FAIL', f'{label}: {len(done)} done, {len(failed)} failed, {left} pending or '
        f'running (pull the other machine\'s results first: git pull)')
    for j in failed[:10]:
        say('CHECK' if label != 'P0' else 'FAIL', f'failed: {j.job_id()} -> {Q.FAILED}/{j.job_id()}.txt')
    return {j.job_id() for j in done}


def g2():
    jobs = phase_jobs(['P0'])
    if not jobs:
        return
    ids = completeness(jobs, 'P0')
    res = [d for d in load_results(Q.RESULTS) if d['job_id'] in ids]
    if not res:
        return
    mt, jt = method_table(res), job_table(res)
    cens = int(jt['T_best'].isna().sum())
    say('PASS' if cens == 0 else 'FAIL', f'oracle reached the band in every job: {len(jt) - cens}/{len(jt)} '
        f'(a job with no T(r*) points to a too small H or a broken phase A)')
    rmin, rmax = min(R_GRID), max(R_GRID)
    edge = jt[(jt['r_best'] == rmin) | (jt['r_best'] == rmax)]
    say('PASS' if len(edge) <= 0.1 * len(jt) else 'CHECK', f'r* at a grid end in {len(edge)}/{len(jt)} jobs '
        f'(a valley outside the grid would bias every regret)')
    for exp, g in jt.groupby('exp'):
        lz, rc = int((g['regime'] == 'lazy').sum()), int((g['regime'] == 'rich').sum())
        say('PASS' if lz and rc else 'CHECK', f'{exp}: {lz} lazy and {rc} rich jobs by measured drift')
    lazy = mt[(mt.method == 'plin2') & (mt.regime == 'lazy')]['regret'].dropna()
    if len(lazy):
        med = float(lazy.median())
        say('PASS' if med <= 1.10 else 'FAIL', f'lazy regime, P_lin2 median regret {med:.3f} (the linear model is '
            f'exact there; far above 1 means an implementation problem, compare with tests/test_job_pilot.py)')
    div = mt[mt['diverged'] == True]                                                     # noqa: E712
    say('PASS' if len(div) == 0 else 'CHECK', f'diverged method lanes: {len(div)} '
        f'({sorted(set(div["method"]))[:6]}; rich regime: lower lr_safety for that alpha)')
    ratios = {}
    for d in res:
        est = d['cfg'].get('est_hours') or 0
        if est > 0:
            ratios.setdefault(d['cfg']['exp'], []).append(d['time_sec'] / 3600 / est)
    for exp, v in sorted(ratios.items()):
        m = float(np.median(v))
        say('PASS' if m <= 2.0 else 'CHECK', f'{exp}: wall time / estimate, median {m:.2f} (lanes assume 2.0; if '
            f'higher, relaunch the lanes with --slowdown {max(2.0, 1.2 * m):.1f})')
    print('next: python scripts/make_jobs.py --phases P0 P1 P2 P3 --calib calib.json   (if nothing FAILs)')


def g3(summary_dir):
    jobs = phase_jobs(['P0', 'P1', 'P2'])
    if not jobs:
        return
    for p in ('P0', 'P1', 'P2'):
        completeness([j for j in jobs if j.priority == p], p)
    tests = os.path.join(summary_dir, 'tests.json')
    skip = SKIP | {'summary', 'figures'}
    newest = max((os.path.getmtime(p) for p in glob.glob(os.path.join(Q.RESULTS, '*', '*.json'))
                  if os.path.basename(os.path.dirname(p)) not in skip), default=0)
    if os.path.exists(tests) and os.path.getmtime(tests) >= newest:
        t = json.load(open(tests))
        n = {v: sum(o.get('verdict') == v for o in t) for v in ('support', 'reject', 'inconclusive')}
        say('PASS', f'aggregate.py is up to date: {len(t)} pre-registered hypotheses evaluated '
            f'({n["support"]} support, {n["reject"]} reject, {n["inconclusive"]} inconclusive)')
    else:
        say('CHECK', 'run python analysis/aggregate.py && python analysis/figures.py on the merged results')
    print('after G3: analysis/aggregate.py and analysis/figures.py are frozen (bug fixes only); P3 keeps running')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('gate', choices=['G1', 'G2', 'G3'])
    ap.add_argument('--calib', default=os.path.join(ROOT, 'calib.json'))
    ap.add_argument('--bench', default=grids.BENCH_PATH)
    ap.add_argument('--summary', default=os.path.join(Q.RESULTS, 'summary'), help='output folder of aggregate.py')
    a = ap.parse_args()
    {'G1': lambda: g1(a.calib, a.bench), 'G2': g2, 'G3': lambda: g3(a.summary)}[a.gate]()
    print(f'\n{a.gate}: {"FAIL" if out["FAIL"] else "PASS"} ({out["FAIL"]} fail, {out["CHECK"]} to check)')
    sys.exit(1 if out['FAIL'] else 0)
