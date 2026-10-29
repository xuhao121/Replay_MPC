"""End-to-end job on the pilot setting (digits, h=256, float64, pilot r grid, H=5000) must reproduce the
pilot's oracle curve and the online controllers' hitting times (results*.jsonl in e0/pilot).
Run:  python tests/test_job_pilot.py        (about 5-10 min on 2 CPU cores)
"""
import json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from rmpc.config import JobConfig, PILOT_GRID
from rmpc.job import run_job

PILOT = os.path.join(ROOT, 'e0', 'pilot')
METHODS = ('oracle', 'plin2', 'probe_swap', 'mpc_0.05', 'trig_0.05', 'periodic', 'replan_only', 'mpcpe_0.2',
           'normbal', 'normbal_lr', 'megaI')


def pilot_ref(alpha, seed=0, scen='rot_sym'):
    key = (scen, alpha, seed)
    base = {(r['scen'], r['alpha'], r['seed']): r for r in map(json.loads, open(os.path.join(PILOT, 'results.jsonl')))}[key]
    ex = {(r['scen'], r['alpha'], r['seed']): r for r in map(json.loads, open(os.path.join(PILOT, 'results_extra.jsonl')))}[key]
    pe = {(r['scen'], r['alpha'], r['seed']): r for r in map(json.loads, open(os.path.join(PILOT, 'results_prederr.jsonl')))}[key]
    truth = {float(k): v for k, v in base['truth'].items()}
    return dict(truth=truth, r_best=base['r_best'], T_best=base['T_best'], M=base['check_every'],
                plin2=base['P_lin2']['T'], probe_swap_r=base['probe_swap']['r'],
                trig=base['trig_0.05']['T'], periodic=base['periodic']['T'],
                replan_only=ex['replan_only']['T'], mpc=ex['trig_replan_0.05']['T'], mpcpe=pe['pe_replan_0.2']['T'])


def run(alpha, out_dir):
    cfg = JobConfig(exp='pilot', tag='rot_sym', dataset='digits', nA=150, nB=150, n_test=50, transform='rot90',
                    model='mlp', width=256, alpha=alpha, init='numpy_pilot', dtype='float64', seed=0,
                    r_grid=PILOT_GRID, H=5000, methods=METHODS)
    return run_job(cfg, out_dir, 'cpu')


def check(alpha):
    torch.set_num_threads(2)
    ref = pilot_ref(alpha)
    with tempfile.TemporaryDirectory() as d:
        out = run(alpha, d)
    T = {r: t for r, t in zip(out['oracle']['r'], out['oracle']['T'])}
    bad = {r: (T[r], ref['truth'][r]) for r in T if (T[r] if T[r] is not None else -1) != ref['truth'][r]}
    assert not bad, f'oracle T(r) differs from pilot: {bad}'
    assert out['M'] == ref['M'], (out['M'], ref['M'])
    meth = out['methods']
    got = dict(plin2=meth['plin2']['T'], probe_swap_r=meth['probe_swap']['r'], trig=meth['trig_0.05']['T'],
               periodic=meth['periodic']['T'], replan_only=meth['replan_only']['T'], mpc=meth['mpc_0.05']['T'],
               mpcpe=meth['mpcpe_0.2']['T'])
    want = {k: ref[k] for k in got}
    assert got == want, f'methods differ: got {got} want {want}'
    print(f'alpha={alpha}: oracle and all online controllers match the pilot; T_best={out["oracle"]["T_best"]}; '
          f'normbal T={meth["normbal"]["T"]} normbal_lr T={meth["normbal_lr"]["T"]} megaI T={meth["megaI"]["T"]}')
    return out


def test_job_pilot_lazy():
    check(1.0)


def test_job_pilot_rich():
    check(0.05)


if __name__ == '__main__':
    check(0.05)
    check(1.0)
