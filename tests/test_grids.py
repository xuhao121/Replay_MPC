"""The job grid must match the experiment document (configuration tables of E1 and E2): job counts, method lists
and seeds per block; memory estimates must fit the lane budgets; calibration must pick one distinct alpha per
drift target; make_jobs.py must add seeds only on request. Fast (no training).
Run:  python tests/test_grids.py      or   python -m pytest -q tests/test_grids.py
"""
import collections, json, os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from rmpc import grids
from rmpc.config import parse_method

EXPECTED = {('P0', 'E1'): 60, ('P0', 'E2'): 18,
            ('P1', 'E1'): 120, ('P1', 'E1N'): 30, ('P1', 'E2'): 27,
            ('P2', 'E1W'): 39,
            ('P3', 'E1'): 10, ('P3', 'E2'): 15}


def _all():
    return grids.build(['P0', 'P1', 'P2', 'P3'])


def test_counts():
    got = collections.Counter((j.priority, j.exp) for j in _all())
    assert dict(got) == EXPECTED, dict(got)
    ids = [j.job_id() for j in _all()]
    assert len(ids) == len(set(ids)), 'duplicate job ids'


def test_methods_and_settings():
    jobs = _all()
    for j in jobs:
        for m in j.methods:
            parse_method(m)                                       # every name is known
    e1 = [j for j in jobs if j.exp == 'E1']
    for j in e1:
        extra = set(j.methods) - set(grids.E1_METHODS)
        main_rot = j.tag == 'rot_sym' and j.width == 1024 and j.nA + j.nB == 2000
        assert extra == (set(grids.THRESHOLD_METHODS) if main_rot else set()), (j.tag, j.width, extra)
        assert len(set(j.methods) & set(grids.E1_METHODS)) == 9
    assert all(set(j.methods) == set(grids.E2_METHODS) for j in jobs if j.exp == 'E2')
    assert all(j.methods == grids.E1N_METHODS and j.priority == 'P1' for j in jobs if j.exp == 'E1N')
    assert all(j.methods == ('oracle', 'plin2') and (j.tolA, j.tolB) == (0.01, 0.02) for j in jobs if j.exp == 'E1W')
    big = [j for j in e1 if j.nA + j.nB == 10000]
    assert len(big) == 10 and all(j.priority == 'P3' and j.mpc_group == 1 for j in big)
    setting = lambda j: (j.exp, j.tag, j.width, j.alpha, j.kernel, j.transform, j.nA, j.nB, j.tolA, j.tolB, j.batch_size)
    main = {setting(j) for j in e1 if j.width == 1024 and j.nA + j.nB == 2000}
    assert len(main) == 40, len(main)                             # 8 scenarios x 5 alphas
    assert {j.seed for j in jobs if j.exp in ('E1', 'E2', 'E1W')} == {0, 1, 2}
    e2p0 = {j.alpha for j in jobs if j.exp == 'E2' and j.priority == 'P0'}
    p1, p3 = grids.e2_subsets(grids.ALPHAS)
    assert len(e2p0) == 6 and set(p1) <= e2p0 and len(p1) == 3 and len(p3) == 2 and not set(p3) & e2p0
    s = sorted(e2p0)
    assert p1 == [s[0], s[2], s[4]], (p1, s)                      # drift targets 0.8, 0.3, 0.05
    assert {j.alpha for j in jobs if j.exp == 'E2' and j.priority == 'P1'} == set(p1)


def test_memory_fits_lanes():
    lanes = (90, 38, 13, 13, 13)
    for j in _all():
        assert j.est_hours > 0 and j.est_mem_gb <= lanes[0], (j.job_id(), j.est_mem_gb)
        if j.model == 'mlp' and j.width <= 1024 and j.nA + j.nB == 2000:
            assert j.est_mem_gb <= 13, (j.job_id(), j.est_mem_gb)
        if j.model == 'resnet18k' and j.width <= 32:
            assert j.est_mem_gb <= 38, (j.job_id(), j.est_mem_gb)


def test_seeds_only_on_request():
    base = grids.build(['P3'])
    more = grids.build(['P3'], p3_seeds={'E1': (3, 4), 'E2': (3, 5)})
    add = collections.Counter(j.exp for j in more if j.seed >= 3)
    assert len(base) == 25 and add == {'E1': 40, 'E2': 16}, add


def test_calibration_assign():
    import calibrate
    row = lambda a, d: dict(alpha=a, drift_sub=d, drift_full=d, T_real=3000)
    alphas = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
    rows = [row(a, min(0.9, 0.6 / a)) for a in alphas]            # drift ~ 1/alpha, saturating
    main = calibrate.assign(rows, calibrate.TARGETS['E2'])
    al = [main[t]['alpha'] for t in sorted(main)]
    assert len(set(al)) == 6 and al == sorted(al, reverse=True), al
    extra = calibrate.assign(rows, calibrate.EXTRA_P3['E2'], exclude=set(al))
    assert len(extra) == 2 and not {r['alpha'] for r in extra.values()} & set(al)
    few = calibrate.assign(rows[:4], calibrate.TARGETS['E2'])
    assert len({r['alpha'] for r in few.values()}) == 4


def test_make_jobs_cli():
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, RMPC_JOBS=os.path.join(d, 'jobs.jsonl'), RMPC_RESULTS=os.path.join(d, 'results'))
        run = lambda *a: subprocess.run([sys.executable, 'scripts/make_jobs.py', *a], cwd=ROOT, env=env,
                                        check=True, capture_output=True, text=True).stdout
        run('--phases', 'P0', 'P1', 'P2')
        rows = [json.loads(l) for l in open(env['RMPC_JOBS'])]
        assert len(rows) == 294 and {r['owner'] for r in rows} == {'m1', 'm2'}
        out = run('--seeds-p3-e1', '3', '4')
        rows = [json.loads(l) for l in open(env['RMPC_JOBS'])]
        assert len(rows) == 294 + 25 + 40, (len(rows), out)
        bad = subprocess.run([sys.executable, 'scripts/make_jobs.py', '--seeds-p3-e2', '0', '3'], cwd=ROOT, env=env,
                             capture_output=True, text=True)
        assert bad.returncode != 0


if __name__ == '__main__':
    for f in [test_counts, test_methods_and_settings, test_memory_fits_lanes, test_seeds_only_on_request,
              test_calibration_assign, test_make_jobs_cli]:
        f()
        print('ok', f.__name__)
    print('grid tests passed')
