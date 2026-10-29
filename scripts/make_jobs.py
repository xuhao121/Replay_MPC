"""Create / extend jobs/jobs.jsonl and split the jobs between the two owners (balanced by estimated hours).

    python scripts/make_jobs.py --phases P0 P1 P2 --calib calib.json --update-estimates   # gate G1 (day 1)
    python scripts/make_jobs.py --phases P0 P1 P2 P3 --calib calib.json                   # gate G2: add the filler
    python scripts/make_jobs.py --phases P3 --calib calib.json --seeds-p3-e2 3 6 --seeds-p3-e1 3 5   # queue dry
    python scripts/make_jobs.py --update-estimates               # re-estimate all jobs with bench.json
    python scripts/make_jobs.py --phases P0 --alphas-e1 0.01 0.03 0.1 0.3 1 --alphas-e2 0.3 1 2 8 18 45
Seeds 3 and up add no settings, so they come last: --seeds-p3-e2 / --seeds-p3-e1 FROM TO add seeds [FROM, TO) to
every E2 rot90 / k = 16 setting and every main-line E1 setting, only when the rest of P3 is queued.
Existing jobs keep their owner; only new jobs are added. One member runs this and commits jobs/jobs.jsonl,
the other pulls it, so both machines see the same owner split.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmpc import grids
from rmpc import queue as Q
from rmpc.config import JobConfig

ap = argparse.ArgumentParser()
ap.add_argument('--phases', nargs='*', default=[])
ap.add_argument('--owners', nargs='+', default=['m1', 'm2'])
ap.add_argument('--alphas-e1', nargs='*', type=float, default=None)
ap.add_argument('--alphas-e2', nargs='*', type=float, default=None)
ap.add_argument('--alphas-e2-p1', nargs='*', type=float, default=None,
                help='E2 alphas of the P1 extensions (default: from calib.json, else the 1st/3rd/5th E2 alpha)')
ap.add_argument('--alphas-e2-p3', nargs='*', type=float, default=None,
                help='two extra E2 alphas of P3 (default: from calib.json, else geometric means of E2 neighbours)')
ap.add_argument('--alpha-e1w', type=float, default=None)
ap.add_argument('--calib', default=None, help='calib.json written by scripts/calibrate.py')
ap.add_argument('--update-estimates', action='store_true')
ap.add_argument('--seeds-p3-e1', nargs=2, type=int, default=None, metavar=('FROM', 'TO'),
                help='add seeds [FROM, TO) to every main-line E1 setting (start at 3; none by default)')
ap.add_argument('--seeds-p3-e2', nargs=2, type=int, default=None, metavar=('FROM', 'TO'),
                help='add seeds [FROM, TO) to every E2 rot90 / k = 16 setting (start at 3; none by default)')
ap.add_argument('--out', default=Q.JOBS)
a = ap.parse_args()
p3 = {}
if a.seeds_p3_e1:
    p3['E1'] = tuple(a.seeds_p3_e1)
if a.seeds_p3_e2:
    p3['E2'] = tuple(a.seeds_p3_e2)
for k, (lo, hi) in p3.items():
    if lo < 3:
        sys.exit(f'--seeds-p3-{k.lower()}: seeds 0-2 are already in P0-P2; start at 3')
if p3 and 'P3' not in a.phases:
    a.phases = list(a.phases) + ['P3']
    print('extra seeds belong to P3: added P3 to --phases')

alphas = {}
if a.calib:
    c = json.load(open(a.calib))
    alphas.update({k: v for k, v in c.get('chosen', {}).items()})
if a.alphas_e1:
    alphas['E1'] = a.alphas_e1
if a.alphas_e2:
    alphas['E2'] = a.alphas_e2
    alphas.pop('E2_P1', None)                   # subsets of calib.json belong to its own E2 list
    alphas.pop('E2_P3', None)
if a.alphas_e2_p1:
    alphas['E2_P1'] = a.alphas_e2_p1
if a.alphas_e2_p3:
    alphas['E2_P3'] = a.alphas_e2_p3
if a.alpha_e1w:
    alphas['E1W'] = a.alpha_e1w
if 'P3' in a.phases and 'E2' in alphas and 'E2_P3' not in alphas:
    print(f'E2 P3 alphas not calibrated: using {grids.e2_subsets(dict(grids.ALPHAS, **alphas))[1]} '
          '(geometric means of neighbouring E2 alphas)')

old = Q.load_jobs(a.out)
by_id = {j.job_id(): j for j in old}
new = [j for j in grids.build(a.phases, alphas, p3) if j.job_id() not in by_id]
bench = grids.load_bench()
if a.update_estimates:
    for j in old:
        j.est_hours, j.est_mem_gb = grids.estimate(j, bench)

load = {}
for j in old:
    load[(j.priority, j.owner)] = load.get((j.priority, j.owner), 0.0) + j.est_hours
for j in sorted(new, key=lambda j: -j.est_hours):            # longest-processing-time-first
    o = min(a.owners, key=lambda o: load.get((j.priority, o), 0.0))
    j.owner = o
    load[(j.priority, o)] = load.get((j.priority, o), 0.0) + j.est_hours
new.sort(key=lambda j: Q.PRIO.get(j.priority, 9))            # stable: longest first within a priority
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out + '.tmp', 'w') as f:
    for j in old + new:
        f.write(json.dumps(j.to_dict()) + '\n')
os.replace(a.out + '.tmp', a.out)

alljobs = old + new
print(f'{len(new)} new jobs, {len(alljobs)} total -> {a.out}')
for p in sorted({j.priority for j in alljobs}):
    for o in a.owners:
        js = [j for j in alljobs if j.priority == p and j.owner == o]
        if js:
            exps = sorted({j.exp for j in js})
            print(f'  {p} {o}: {len(js):4d} jobs, est {sum(j.est_hours for j in js):7.1f} GPU-h (exclusive), '
                  f'max mem {max(j.est_mem_gb for j in js):5.1f} GB, exps {exps}')
