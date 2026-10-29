"""End-to-end smoke tests on tiny synthetic problems (CPU, a few minutes): every code path of a job
(MLP with all methods incl. oracle_edge, ResNet with eNTK + lane chunking, pNTK + sample chunking, SGD),
the job queue with a worker, aggregation, figures and the gate checks.
Run:  python -m pytest -q tests/test_smoke.py      or   python tests/test_smoke.py
"""
import json, os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from rmpc.config import JobConfig, DEFAULT_METHODS
from rmpc.grids import E1_METHODS, E2_METHODS, E1N_METHODS
from rmpc.job import run_job

torch.set_num_threads(2)


def _check(out, methods):
    assert out['oracle']['T_best'] is not None, 'no fixed r reached the band'
    for m in methods:
        if m in ('oracle', 'oracle_edge'):
            continue
        assert m in out['methods'], m
    json.dumps(out)


def mlp_cfg(**kw):
    d = dict(exp='smoke', tag='mlp', dataset='synth', nA=60, nB=60, n_test=40, transform='perm', model='mlp',
             width=64, alpha=0.1, H_max=800, stepsA_max=4000, seed=0)
    d.update(kw)
    return JobConfig(**d)


def test_mlp_all_methods(tmp=None):
    with tempfile.TemporaryDirectory() as d:
        out = run_job(mlp_cfg(methods=DEFAULT_METHODS), d, 'cpu')
        assert out['oracle_edge'] is not None and out['oracle_edge']['T_best'] is not None
        _check(out, DEFAULT_METHODS)
        z = np.load(os.path.join(d, out['job_id'] + '.npz'))
        assert 'or_LA_0' in z.files and 'm_mpc_0.05_LA' in z.files
        tm = out['timing']
        assert tm['step_sec'] > 0 and tm['KA_sec'] > 0 and tm['plin2_sec'] > 0, tm
        assert all(out['methods'][m]['ctrl_sec'] >= 0 for m in out['methods'] if m.startswith(('mpc', 'trig', 'periodic', 'replan')))


def test_resnet_entk_lane_chunks():
    cfg = JobConfig(exp='smoke', tag='cnn', dataset='synth_img', nA=24, nB=24, n_test=16, transform='rot90',
                    model='resnet18k', width=8, alpha=10.0, kernel='entk', jac_dtype='bfloat16', lane_chunk=4,
                    H_max=3000, stepsA_max=3000, optA='momentum', targetA_rel=0.05, tolB=0.3,
                    methods=E2_METHODS, r_grid=(0.2, 0.35, 0.5, 0.65, 0.8))
    with tempfile.TemporaryDirectory() as d:
        out = run_job(cfg, d, 'cpu')
        _check(out, E2_METHODS)


def test_resnet_pntk_sample_chunks():
    cfg = JobConfig(exp='smoke', tag='cnnp', dataset='synth_img', nA=24, nB=24, n_test=16, transform='neg',
                    model='resnet18k', width=8, alpha=10.0, kernel='pntk', sample_chunk=16, lane_chunk=2,
                    H_max=3000, stepsA_max=3000, optA='momentum', targetA_rel=0.05, tolB=0.3,
                    methods=('oracle', 'plin2', 'mpc_0.05', 'mpcpe_0.2', 'mpc2_0.05', 'replan_only'),
                    r_grid=(0.2, 0.5, 0.8))
    with tempfile.TemporaryDirectory() as d:
        out = run_job(cfg, d, 'cpu')
        _check(out, cfg.methods)


def test_sgd():
    cfg = mlp_cfg(batch_size=16, eval_every=5, record_every=5, methods=E1N_METHODS, H_max=1500, tag='sgd')
    with tempfile.TemporaryDirectory() as d:
        out = run_job(cfg, d, 'cpu')
        assert out['M'] % 5 == 0
        _check(out, E1N_METHODS)


def test_queue_worker_aggregate():
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, RMPC_JOBS=os.path.join(d, 'jobs.jsonl'), RMPC_RESULTS=os.path.join(d, 'results'),
                   RMPC_STOP=os.path.join(d, 'STOP'))
        jobs = [mlp_cfg(seed=s, methods=E1_METHODS, owner='m1', exp='E1', tag='rot_sym') for s in (0, 1)]
        with open(env['RMPC_JOBS'], 'w') as f:
            for j in jobs:
                f.write(json.dumps(j.to_dict()) + '\n')
        subprocess.run([sys.executable, 'scripts/worker.py', '--owner', 'm1', '--device', 'cpu', '--exit-when-empty'],
                       cwd=ROOT, env=env, check=True)
        res = os.listdir(os.path.join(d, 'results', 'E1'))
        assert sum(r.endswith('.json') for r in res) == 2, res
        subprocess.run([sys.executable, 'scripts/status.py'], cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, 'analysis/aggregate.py', '--results', os.path.join(d, 'results'),
                        '--out', os.path.join(d, 'summary')], cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, 'analysis/figures.py', '--results', os.path.join(d, 'results'),
                        '--out', os.path.join(d, 'figs')], cwd=ROOT, env=env, check=True)
        assert os.path.exists(os.path.join(d, 'summary', 'summary.md'))
        # gate checks: G1 on synthetic calibration / bench files, G2 and G3 on the two finished jobs
        row = lambda a, dr, T=500: dict(alpha=a, width=1024, drift_sub=dr, drift_full=dr, T_real=T)
        e2a = (0.3, 1.0, 1.5, 2.0, 4.0, 8.0, 18.0, 45.0)
        calib = dict(runs=dict(E1=[row(a, dr) for a, dr in zip((0.01, 0.03, 0.1, 0.3, 1.0), (1.0, 0.5, 0.2, 0.07, 0.02))],
                               E2=[row(a, dr, 3000) for a, dr in zip(e2a, (0.8, 0.5, 0.4, 0.3, 0.15, 0.1, 0.05, 0.02))],
                               E1W=[dict(row(10.0, 0.05), width=8)]),
                     chosen=dict(E1=[0.01, 0.03, 0.1, 0.3, 1.0], E2=[0.3, 1.0, 2.0, 8.0, 18.0, 45.0],
                                 E2_P1=[0.3, 2.0, 18.0], E2_P3=[1.5, 4.0]))
        json.dump(calib, open(os.path.join(d, 'calib.json'), 'w'))
        json.dump(dict(E1_ref_hours=0.1, E2_ref_hours=0.6, concurrent_E1={'2': dict(speedup=1.8)}),
                  open(os.path.join(d, 'bench.json'), 'w'))
        g = subprocess.run([sys.executable, 'analysis/gate_check.py', 'G1', '--calib', os.path.join(d, 'calib.json'),
                            '--bench', os.path.join(d, 'bench.json')], cwd=ROOT, env=env, capture_output=True, text=True)
        print(g.stdout)
        assert g.returncode == 0, g.stdout + g.stderr
        assert 'CHECK E2' not in g.stdout, g.stdout
        for gate in ('G2', 'G3'):
            g = subprocess.run([sys.executable, 'analysis/gate_check.py', gate, '--summary', os.path.join(d, 'summary')],
                               cwd=ROOT, env=env, capture_output=True, text=True)
            print(g.stdout)
            assert g.returncode in (0, 1) and f'{gate}: ' in g.stdout, g.stdout + g.stderr
            assert 'P0: 2 done, 0 failed, 0 pending' in g.stdout, g.stdout


if __name__ == '__main__':
    for f in [test_mlp_all_methods, test_resnet_entk_lane_chunks, test_resnet_pntk_sample_chunks, test_sgd,
              test_queue_worker_aggregate]:
        print('==', f.__name__, flush=True)
        f()
    print('smoke tests passed')
