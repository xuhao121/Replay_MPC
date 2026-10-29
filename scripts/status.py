"""Queue progress: done / running / failed / pending per priority and owner, GPU hours used, ETA.

    python scripts/status.py                 # all owners
    python scripts/status.py --owner m1 --lanes 6
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmpc import queue as Q

ap = argparse.ArgumentParser()
ap.add_argument('--owner', default=None)
ap.add_argument('--lanes', type=int, default=1, help='concurrent lanes (for a rough ETA)')
ap.add_argument('--speedup', type=float, default=None, help='measured throughput gain of N lanes (bench.py)')
a = ap.parse_args()

jobs = [j for j in Q.load_jobs() if a.owner in (None, j.owner)]
rows = {}
used = 0.0
for j in jobs:
    k = (j.priority, j.owner)
    r = rows.setdefault(k, dict(done=0, running=0, failed=0, pending=0, est_left=0.0, est_done=0.0))
    if Q.is_done(j):
        r['done'] += 1
        r['est_done'] += j.est_hours
        try:
            used += json.load(open(Q.result_path(j)))['time_sec'] / 3600
        except Exception:
            pass
    elif Q.claim_live(j, 20):
        r['running'] += 1
        r['est_left'] += j.est_hours / 2
    elif Q.is_failed(j):
        r['failed'] += 1
    else:
        r['pending'] += 1
        r['est_left'] += j.est_hours
print(f'{"prio":5s} {"owner":5s} {"done":>6s} {"run":>5s} {"fail":>5s} {"pend":>6s} {"est h left":>11s}')
tot_left = 0.0
for (p, o), r in sorted(rows.items()):
    print(f'{p:5s} {o:5s} {r["done"]:6d} {r["running"]:5d} {r["failed"]:5d} {r["pending"]:6d} {r["est_left"]:11.1f}')
    tot_left += r['est_left']
sp = a.speedup or max(1.0, 0.55 * a.lanes)
print(f'wall time of finished jobs: {used:.1f} h (summed over lanes); estimated work left {tot_left:.1f} '
      f'exclusive GPU-h -> about {tot_left / sp:.1f} h wall with {a.lanes} lanes (speedup {sp:.1f}x)')
fails = sorted(glob.glob(os.path.join(Q.FAILED, '*.txt')))
if fails:
    print(f'{len(fails)} failed jobs, e.g. {fails[-1]}  (inspect, fix, then run workers with --retry-failed)')
