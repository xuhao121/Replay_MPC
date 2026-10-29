import json, sys, numpy as np
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
methods = ['P_lin2', 'probe_swap', 'probe_piece', 'probe_cont', 'trig_0.05', 'trig_0.2', 'periodic']
groups = defaultdict(list)
for r in rows:
    if 'error' in r: print('ERROR', r['scen'], r['alpha'], r['seed'], r['error'][-300:]); continue
    groups[(r['scen'], r['alpha'])].append(r)
def fmt(x): return 'NA' if x is None else f'{x:.2f}'
for (scen, alpha), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
    print(f"\n== {scen}  alpha={alpha}  seeds={[r['seed'] for r in rs]}")
    print('  best fixed r / T_best:', [(r['r_best'], r['T_best']) for r in rs],
          ' kernel drift (best run):', [round(r['kernel_drift_best_run'], 2) for r in rs],
          ' probe drift:', [round(r['probe_kernel_drift'], 2) for r in rs])
    for m in methods:
        regs = [r[m]['regret'] for r in rs]
        extra = ''
        if m.startswith('trig') or m == 'periodic':
            extra = f"  relin={[r[m]['n_relin'] for r in rs]} max_drift={[round(r[m]['max_drift'],2) for r in rs]}"
        rsel = [r[m].get('r') for r in rs]
        print(f"  {m:12s} regret={[fmt(x) for x in regs]}  r={rsel}{extra}")
