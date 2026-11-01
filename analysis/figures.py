"""Report figures (PNG + PDF) from results/.

    python analysis/figures.py               # -> results/figures/
Figures
  fig1_curves      T(r): real training vs linearised prediction, a lazy and a rich configuration (E1, E2)
  fig2_regret      regret vs kernel drift, one panel per method (P_lin2, MPC, trigger only, stale kernel)
  fig3_cost        re-linearisations per run and wall-clock regret (kernels and plans included) vs drift
  fig4_traces      r(t) of mpc_0.05 / replan_only and the losses in the richest E1 job, plus r(t) of mpc_0.05 /
                   mpc2_0.05 and the first two-phase plan in a typical lazy E1 job (plan shape, H3)
  fig5_gradstat    appendix: gradient-statistics rules vs the best fixed r, each at its own tuned step size
                   (E0, e0/results/sim2.json; no GPU results needed)
  fig6_width       E1W: best time-to-joint T* and test accuracy vs width (interpolation threshold)
  fig7_noise       E1N: best fixed r and regrets vs minibatch size
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_results, method_table, job_table, ROOT, C, INK, INK2, GRID, LAZY_DRIFT

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'legend.frameon': False, 'savefig.dpi': 200})


def save(fig, out, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), bbox_inches='tight')    # keeps figure-level legends
    plt.close(fig)
    print('wrote', name)


def _drift(d):
    return d['drift'].get('full_best', d['drift'].get('sub_best'))


def _legend_below(fig, axes, ncol):
    """One legend under the panels, collected from every panel (first occurrence of each label)."""
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc='upper center', bbox_to_anchor=(0.5, 0.0),
                   ncol=ncol, fontsize=7)


def _curve(d):
    r = np.array(d['oracle']['r'])
    T = np.array([np.nan if t is None else t for t in d['oracle']['T']], float)
    L = np.array([np.nan if (t is None or t < 0) else t for t in d['plin2_curve']['T_lin']], float)
    return r, T, L


def fig1(jobs, out):
    picks = []
    for exp in ('E1', 'E2'):
        js = [d for d in jobs if d['cfg']['exp'] == exp and d.get('drift') and d['oracle']['T_best']]
        if not js:
            continue
        js.sort(key=_drift)
        picks += [(exp, 'lazy', js[0]), (exp, 'rich', js[-1])]
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 2.9), squeeze=False)
    for ax, (exp, lab, d) in zip(axes[0], picks):
        r, T, L = _curve(d)
        ax.plot(r, T, 'o-', color=C[0], ms=4, label='real training')
        ax.plot(r, L, 's--', color=C[1], ms=4, label='linearised at θ_A')
        miss = ~np.isfinite(T)                    # fixed r that never reached the band within H
        if miss.any():
            ax.scatter(r[miss], np.full(int(miss.sum()), float(d['H'])), marker='x', color=C[0], s=22,
                       linewidths=1.2, zorder=3, label='real: not in band within H')
        ax.axvline(d['oracle']['r_best'], color=C[0], lw=1, alpha=0.5, label='r* (best fixed r)')
        ax.axvline(d['r_lin'], color=C[1], lw=1, ls=':', alpha=0.8, label='r̂ (P_lin2)')
        cfg = d['cfg']
        name = f"{exp} {cfg['tag']}, " + (f"h={cfg['width']}" if cfg['model'] == 'mlp' else f"k={cfg['width']}")
        ax.set_title(f"{name}\nα={cfg['alpha']:g}, drift {_drift(d):.2f}", fontsize=8, color=INK)
        ax.set_yscale('log')
        ax.set_xlabel('replay ratio r')
    axes[0][0].set_ylabel('steps to joint band T(r)')
    _legend_below(fig, axes[0], ncol=5)
    save(fig, out, 'fig1_curves')


def fig2(df, out):
    meths = [('plin2', 'P_lin2 (plan once)'), ('mpc_0.05', 'MPC (re-plan, drift trigger)'),
             ('trig_0.05', 'trigger only (keep r)'), ('replan_only', 're-plan, stale kernel')]
    meths = [m for m in meths if (df.method == m[0]).any()]
    if not meths:
        return
    fig, axes = plt.subplots(1, len(meths), figsize=(2.7 * len(meths), 2.6), sharey=True, squeeze=False)
    bins = np.logspace(-2, 0.3, 8)
    groups = [('E1', lambda s: s.exp == 'E1', C[0], 0.45),
              ('E2, eNTK planner', lambda s: (s.exp == 'E2') & (s.kernel != 'pntk'), C[2], 0.6),
              ('E2, pNTK planner', lambda s: (s.exp == 'E2') & (s.kernel == 'pntk'), C[1], 0.75),
              ('E1N, E1W', lambda s: s.exp.isin(['E1N', 'E1W']), INK2, 0.3)]
    for ax, (m, lab) in zip(axes[0], meths):
        s = df[(df.method == m) & df.regret.notna() & df.drift.notna()]
        for name, sel, col, al in groups:
            g = s[sel(s)]
            if len(g):
                ax.scatter(g.drift, g.regret, s=9, color=col, alpha=al, linewidths=0, label=name)
        idx = np.digitize(s.drift, bins)
        xs = [s.drift[idx == k].median() for k in np.unique(idx)]
        ys = [np.exp(np.log(s.regret[idx == k]).mean()) for k in np.unique(idx)]
        ax.plot(xs, ys, '-', color=INK, lw=1.5, label='geometric mean per drift bin (all)')
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.axvline(LAZY_DRIFT, color=INK2, lw=0.8, ls=':', label='lazy / rich boundary (drift 0.2)')
        ax.set_xscale('log')
        ax.set_title(lab, fontsize=8, color=INK)
        ax.set_xlabel('kernel drift (best fixed-r run)')
    axes[0][0].set_ylabel('regret  T / T_best')
    _legend_below(fig, axes[0], ncol=5)
    save(fig, out, 'fig2_regret')


def fig3(df, out):
    meths = [('mpc_0.05', 'drift trigger ε=0.05', C[0]), ('mpcpe_0.2', 'prediction-error trigger ε=0.2', C[1]),
             ('periodic', 'every check', C[2])]
    meths = [m for m in meths if (df.method == m[0]).any()]
    walls = [('plin2', 'P_lin2 (plan once)', C[3])] + meths
    walls = [m for m in walls if df[(df.method == m[0])]['wall_regret'].notna().any()] if 'wall_regret' in df else []
    if not meths:
        return
    fig, axes = plt.subplots(1, 2 if walls else 1, figsize=(8.4 if walls else 4.2, 2.8), squeeze=False)
    a1 = axes[0][0]
    for m, lab, col in meths:
        s = df[(df.method == m) & df.drift.notna()].sort_values('drift')
        a1.scatter(s.drift, s.n_relin + np.random.default_rng(0).uniform(-0.15, 0.15, len(s)), s=9, color=col,
                   alpha=0.5, linewidths=0, label=lab)
    a1.set_xscale('log')
    a1.set_xlabel('kernel drift (best fixed-r run)')
    a1.set_ylabel('re-linearisations per run')
    a1.legend(fontsize=7)
    if walls:
        a2 = axes[0][1]
        for m, lab, col in walls:
            s = df[(df.method == m) & df.drift.notna() & df.wall_regret.notna()]
            a2.scatter(s.drift, s.wall_regret, s=9, color=col, alpha=0.5, linewidths=0, label=lab)
        a2.axhline(1.0, color=INK2, lw=0.8)
        a2.set_xscale('log'); a2.set_yscale('log')
        plain = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')
        lead = lambda v: round(v / 10 ** np.floor(np.log10(v)), 6) if v > 0 else 0
        short = lambda: np.log10(a2.get_ylim()[1] / a2.get_ylim()[0]) < 1.2     # less than ~a decade shown
        minor = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}' if (lead(v) in (2, 5) or short()) else '')
        a2.yaxis.set_major_formatter(plain); a2.yaxis.set_minor_formatter(minor)   # label 1, 2, 5, 10, 20, 50 ...
        a2.set_xlabel('kernel drift (best fixed-r run)')
        a2.set_ylabel('wall-clock regret (kernels, plans incl.)')
        a2.legend(fontsize=7, loc='best')
    save(fig, out, 'fig3_cost')


def _trace_xy(m, H):
    tr = m['trace']
    T = m.get('T') or H
    return [p[0] for p in tr] + [T], [p[1] for p in tr] + [tr[-1][1]]


def _relin_ticks(ax, m, col):
    rt = m.get('relin_t') or []
    if rt:                                        # re-linearisations: ticks along the top edge of the panel
        ax.plot(rt, [0.97] * len(rt), '|', color=col, ms=7, mew=1.5, transform=ax.get_xaxis_transform(),
                label='re-linearisation')


def fig4(jobs, out, results_dir):
    """(a, b) the richest E1 job: r(t) of mpc_0.05 and replan_only, and the losses; (c) a typical lazy E1 job
    (mpc2 / mpc step ratio closest to the median): r(t) of mpc_0.05 and mpc2_0.05 and the first two-phase plan."""
    E1 = [d for d in jobs if d['cfg']['exp'] == 'E1' and d.get('drift') and d['oracle']['T_best']]
    js = [d for d in E1 if 'mpc_0.05' in d['methods'] and d['methods']['mpc_0.05'].get('trace')]
    if not js:
        return
    d = max(js, key=_drift)
    ok = lambda x, m: bool((x['methods'].get(m) or {}).get('T')) and bool(x['methods'][m].get('trace'))
    lz = [x for x in E1 if _drift(x) < LAZY_DRIFT and ok(x, 'mpc_0.05') and ok(x, 'mpc2_0.05')]
    lazy = None
    if lz:
        rat = [x['methods']['mpc2_0.05']['T'] / x['methods']['mpc_0.05']['T'] for x in lz]
        lazy = lz[int(np.argmin(np.abs(np.array(rat) - np.median(rat))))]
    fig, axs = plt.subplots(1, 3 if lazy else 2, figsize=(11.1 if lazy else 7.4, 2.9))
    a1, a2 = axs[0], axs[1]
    z = np.load(os.path.join(results_dir, d['cfg']['exp'], f"{d['job_id']}.npz"))
    r_best = d['oracle']['r_best']
    ib = d['oracle']['r'].index(r_best)
    t = z[f'or_t_{ib}']
    a1.axhline(r_best, color=C[0], lw=1.5, label=f'fixed r* = {r_best:g}')
    a2.plot(t, z[f'or_LB_{ib}'], color=C[0])
    a2.plot(t, z[f'or_LA_{ib}'], color=C[0], ls='--')
    for m, col in [('mpc_0.05', C[1]), ('replan_only', C[2])]:
        if m not in d['methods'] or not d['methods'][m].get('trace'):
            continue
        xs, ys = _trace_xy(d['methods'][m], d['H'])
        a1.step(xs, ys, where='post', color=col, label=m)
        _relin_ticks(a1, d['methods'][m], col)
        a2.plot(z[f'm_{m}_t'], z[f'm_{m}_LB'], color=col)
        a2.plot(z[f'm_{m}_t'], z[f'm_{m}_LA'], color=col, ls='--')
    a1.set_xlabel('step'); a1.set_ylabel('replay ratio r(t)')
    a1.set_title(f"rich: {d['cfg']['tag']}, h={d['cfg']['width']}, α={d['cfg']['alpha']:g}, drift {_drift(d):.2f}",
                 fontsize=8, color=INK)
    a2.axhline(d['phaseA']['tauB'], color=INK2, lw=0.8, ls=':')
    a2.axhline(d['phaseA']['tauA'], color=INK2, lw=0.8, ls='-.')
    style = [plt.Line2D([], [], color=INK2, lw=1.5), plt.Line2D([], [], color=INK2, lw=1.5, ls='--'),
             plt.Line2D([], [], color=INK2, lw=0.8, ls=':'), plt.Line2D([], [], color=INK2, lw=0.8, ls='-.')]
    a2.legend(style, ['L_B', 'L_A', 'band edge τ_B', 'band edge τ_A'], fontsize=7, ncol=2)
    a2.set_yscale('log'); a2.set_xlabel('step'); a2.set_ylabel('training loss (colours as left)')
    a2.set_title('same job: training losses', fontsize=8, color=INK)
    if lazy:
        a3 = axs[2]
        rb = lazy['oracle']['r_best']
        a3.axhline(rb, color=C[0], lw=1.5)
        for m, col in [('mpc_0.05', C[1]), ('mpc2_0.05', C[3])]:
            xs, ys = _trace_xy(lazy['methods'][m], lazy['H'])
            a3.step(xs, ys, where='post', color=col, label=m)
        p2 = lazy['methods']['mpc2_0.05'].get('plans2') or []
        if p2:                                    # the first two-phase plan: r1 for t_switch steps, then r2
            t0, r1, r2, ts = p2[0]
            end = lazy['methods']['mpc2_0.05']['T']
            a3.plot([t0, t0 + ts, t0 + ts, end], [r1, r1, r2, r2], color=C[3], lw=1, ls='--', alpha=0.8,
                    label='first two-phase plan')
        a3.set_xlabel('step'); a3.set_ylabel('replay ratio r(t)')
        a3.set_title(f"lazy: {lazy['cfg']['tag']}, h={lazy['cfg']['width']}, α={lazy['cfg']['alpha']:g}, "
                     f"drift {_drift(lazy):.2f}", fontsize=8, color=INK)
    _legend_below(fig, [a1] + ([axs[2]] if lazy else []), ncol=6)
    save(fig, out, 'fig4_traces')


def fig5(out, sim2_path=os.path.join(ROOT, 'e0', 'results', 'sim2.json')):
    """Appendix: steps to the joint band of each gradient-statistics rule / steps of the best fixed r, every
    method at its own tuned step size (E0 linear problems, written by e0/sim2.py)."""
    if not os.path.exists(sim2_path):
        return
    res = json.load(open(sim2_path))
    meths = [('normbal(ratio only)', 'equal-norm ratio rule', C[1]), ('normbal_lr', 'LR-compensated rule', C[2]),
             ('MEGA-I(eps=1e-3)', 'MEGA-I', C[3])]
    groups = {}                                   # problem (alpha, task-B scale) -> its seeds
    for key in res:
        prob, _, seed = key.rpartition(',seed=')
        groups.setdefault(prob, []).append(key)
    pts, top = [], 1.0
    for g, (prob, keys) in enumerate(groups.items()):
        for s, key in enumerate(keys):
            rs = res[key]
            best = min(v['T_hit'] for m, v in rs.items() if m.startswith('fixed_r') and v.get('T_hit'))
            for j, (m, lab, col) in enumerate(meths):
                T = (rs.get(m) or {}).get('T_hit')
                pts.append((4 * g + j + (s - 1) * 0.12, T / best if T else None, col))
                top = max(top, T / best if T else top)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    cap = top * 2.0
    for x, y, col in pts:
        if y is None:
            ax.scatter(x, cap, s=18, color=col, marker='x', linewidths=1.2)
        else:
            ax.scatter(x, y, s=14, color=col, linewidths=0)
    for m, lab, col in meths:
        ax.scatter([], [], s=14, color=col, label=lab)
    ax.scatter([], [], s=18, color=INK2, marker='x', label='did not reach the band')
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_yscale('log')
    ax.set_xticks([4 * g + 1 for g in range(len(groups))])
    ax.set_xticklabels([p.replace('alpha=', 'spectrum α ').replace(',scaleB=', '\ntask-B scale ') for p in groups],
                       fontsize=7)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('T / T(best tuned fixed r)')
    ax.set_title('E0: replay-weighting rules, each at its own tuned step size (3 seeds per problem)',
                 fontsize=8, color=INK)
    ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.3))
    save(fig, out, 'fig5_gradstat')


def fig6(jt, out):
    w = jt[jt.exp == 'E1W']
    if not len(w):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for k, (tag, g) in enumerate(w.groupby('tag')):
        rows = []
        for h, gh in g.groupby('width'):
            hit = gh.T_best.notna()
            rows.append(dict(h=h, n=len(gh), n_ok=int(hit.sum()), H=float(gh.H.max()),
                             T=float(np.exp(np.log(gh.T_best[hit].astype(float)).mean())) if hit.any() else np.nan,
                             acc=float(gh.acc_B_best[hit].mean()) if hit.any() else np.nan))
        t = pd.DataFrame(rows).sort_values('h')
        some = t[t.n_ok > 0]
        full, part, none = t[t.n_ok == t.n], t[(t.n_ok > 0) & (t.n_ok < t.n)], t[t.n_ok == 0]
        col = C[k]
        for a, y in ((a1, 'T'), (a2, 'acc')):
            a.plot(some.h, some[y], '-', color=col, lw=1.5)
            a.plot(full.h, full[y], 'o', color=col, ms=4, label=f'band {tag}: every seed in the band')
            a.plot(part.h, part[y], 'o', color=col, mfc='white', ms=5,
                   label='some seeds in the band (mean of those)')
            for _, p in part.iterrows():
                a.annotate(f"{int(p.n_ok)}/{int(p.n)} seeds", (p.h, p[y]), textcoords='offset points', xytext=(5, -3),
                           fontsize=7, color=col)
        if len(none):
            a1.plot(none.h, none.H, 'v', color=INK2, ms=4, label=f'no seed in the band within H')
    for a in (a1, a2):
        a.set_xscale('log', base=2); a.set_xlabel('hidden width h')
        a.axvline(20000 / 794, color=INK2, lw=0.8, ls=':', label='parameter count = (n_A + n_B) C, h ≈ 25')
    a1.set_yscale('log'); a1.set_ylabel('T* (best fixed r)')
    a2.set_ylabel('task-B test accuracy at T*')
    _legend_below(fig, [a1, a2], ncol=2)
    save(fig, out, 'fig6_width')


def fig7(df, jt, out):
    n = jt[jt.exp == 'E1N']
    if not len(n):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for k, (al, g) in enumerate(n.groupby('alpha')):
        g2 = g.groupby('batch').r_best.mean()
        a1.plot(g2.index, g2.values, 'o-', color=C[k], ms=4, label=f'α={al:g}')
        for m, ls in [('plin2', '--'), ('mpc_0.05', '-')]:
            s = df[(df.exp == 'E1N') & (df.alpha == al) & (df.method == m)].groupby('batch').regret.median()
            a2.plot(s.index, s.values, ls, color=C[k], marker='o', ms=4, label=f'{m} α={al:g}')
    for a in (a1, a2):
        a.set_xscale('log', base=2); a.set_xlabel('minibatch size B')
    a1.set_ylabel('best fixed r'); a1.legend(fontsize=7)
    a2.set_ylabel('median regret'); a2.axhline(1, color=INK2, lw=0.8); a2.legend(fontsize=7)
    save(fig, out, 'fig7_noise')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'figures'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig5(a.out)                                   # appendix figure from the E0 results, no GPU results needed
    jobs = load_results(a.results)
    if not jobs:
        print('no results')
        return
    df, jt = method_table(jobs), job_table(jobs)
    fig1(jobs, a.out)
    fig2(df, a.out)
    fig3(df, a.out)
    fig4(jobs, a.out, a.results)
    fig6(jt, a.out)
    fig7(df, jt, a.out)


if __name__ == '__main__':
    main()
