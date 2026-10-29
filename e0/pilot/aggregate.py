import json, numpy as np, sys
rows = [json.loads(l) for l in open('results.jsonl') if l.strip() and 'error' not in l]
extra = {}
try:
    for l in open('results_extra.jsonl'):
        e = json.loads(l); extra[(e['scen'], e['alpha'], e['seed'])] = e
except FileNotFoundError:
    pass
methods = ['P_lin2', 'probe_swap', 'probe_piece', 'probe_cont', 'trig_0.05', 'trig_0.2', 'periodic', 'replan_only', 'trig_replan_0.05', 'trig_replan_0.2']
def reg(r, m):
    if m in r: return r[m]['regret']
    e = extra.get((r['scen'], r['alpha'], r['seed']))
    return e[m]['regret'] if e and m in e else None
def nrel(r, m):
    if m in r: return r[m].get('n_relin')
    e = extra.get((r['scen'], r['alpha'], r['seed']))
    return e[m]['n_relin'] if e and m in e else None
groups = {'lazy (alpha=1)': [r for r in rows if r['alpha'] == 1.0], 'rich (alpha<=0.05)': [r for r in rows if r['alpha'] < 1.0], 'all': rows}
for gname, rs in groups.items():
    print(f"\n== {gname}: {len(rs)} configs; median kernel drift along best run {np.median([r['kernel_drift_best_run'] for r in rs]):.2f}")
    for m in methods:
        v = [reg(r, m) for r in rs]
        if any(x is None for x in v):
            if all(x is None for x in v): continue
            print(f"  {m:18s} (missing {sum(x is None for x in v)})"); v = [x for x in v if x is not None]
        v = np.array(v); nr = [nrel(r, m) for r in rs]
        nr_s = f"  mean relin {np.mean([x for x in nr if x is not None]):.1f}" if any(x is not None for x in nr) else ''
        print(f"  {m:18s} mean {v.mean():.3f}  median {np.median(v):.3f}  max {v.max():.3f}  min {v.min():.3f}  share within 2% {np.mean(v <= 1.02):.0%}{nr_s}")
