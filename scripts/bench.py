"""Day-1 speed measurement on the real GPU; writes bench.json used by make_jobs.py --update-estimates.

    python scripts/bench.py single                    # reference E1 and E2 jobs, exclusive GPU
    python scripts/bench.py single --exp E1
    python scripts/bench.py concurrent --procs 2 4 8  # N copies of the E1 reference job at once
Pick the number of lanes N where adding processes stops raising throughput by more than ~10%.
If throughput at 8 lanes is below ~3x, try CUDA MPS (see README) and measure again.
"""
import argparse, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from rmpc import grids
from rmpc.job import run_job
from rmpc.utils import write_json_atomic

BENCH = os.path.join(ROOT, 'bench.json')


def ref_cfg(exp):
    """Reference job of each family: standard method list, intermediate laziness (grids.BENCH_ALPHA)."""
    if exp == 'E1':
        return grids.e1('rot_sym', grids.BENCH_ALPHA['E1'], seed=900)
    return grids.e2(grids.BENCH_ALPHA['E2'], seed=900)


def load():
    return json.load(open(BENCH)) if os.path.exists(BENCH) else {}


def single(exps, device):
    b = load()
    for e in exps:
        cfg = ref_cfg(e)
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = run_job(cfg, os.path.join(ROOT, 'results', '_bench'), device)
        h = (time.time() - t0) / 3600
        b[f'{e}_ref_hours'] = round(h, 4)
        b[f'{e}_ref_timer'] = out['timer']
        if device.startswith('cuda'):
            b[f'{e}_ref_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        b['gpu'] = out['gpu']
        print(f'{e}: {h * 60:.1f} min, timer {out["timer"]}')
    write_json_atomic(BENCH, b)
    print('written', BENCH)


def concurrent(procs, device):
    b = load()
    base = b.get('E1_ref_hours')
    res = {}
    for n in procs:
        t0 = time.time()
        ps = [subprocess.Popen([sys.executable, __file__, 'one', '--device', device], cwd=ROOT) for _ in range(n)]
        for p in ps:
            p.wait()
        wall = (time.time() - t0) / 3600
        res[n] = dict(wall_hours=round(wall, 4), speedup=round(n * base / wall, 2) if base else None)
        print(f'{n} lanes: wall {wall * 60:.1f} min, throughput x{res[n]["speedup"]}')
    b['concurrent_E1'] = res
    write_json_atomic(BENCH, b)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['single', 'concurrent', 'one'])
    ap.add_argument('--exp', nargs='+', default=['E1', 'E2'])
    ap.add_argument('--procs', nargs='+', type=int, default=[2, 4, 8])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.mode == 'single':
        single(a.exp, a.device)
    elif a.mode == 'concurrent':
        concurrent(a.procs, a.device)
    else:
        run_job(ref_cfg('E1'), os.path.join(ROOT, 'results', '_bench'), a.device, log=lambda s: None)
