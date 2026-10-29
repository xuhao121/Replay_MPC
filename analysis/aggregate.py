"""Tables and pre-registered tests.

    python analysis/aggregate.py                    # all results -> results/summary/
    python analysis/aggregate.py --exp E1 E2
Unit of analysis: the setting (everything except the seed). Seeds are averaged on the log scale first
(geometric mean), confidence intervals come from a bootstrap over settings, paired tests are Wilcoxon
signed-rank tests over settings, and the confirmatory family (H2b, H2c, H2d, H3) is Holm-corrected.
Every hypothesis gets a verdict as pre-registered in the experiment document: SUPPORT when its support
criterion holds, REJECT when its rejection criterion holds, INCONCLUSIVE otherwise. E2 is also evaluated on
its own (descriptive, no Holm correction). Exploratory and appendix tables carry no verdicts.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_results, method_table, job_table, ROOT

MAIN = ['fixed_0.5', 'plin2', 'mpc_0.05', 'mpcpe_0.2', 'trig_0.05', 'replan_only', 'periodic', 'mpc2_0.05', 'normbal']
COST = ['fixed_0.5', 'plin2', 'mpc_0.05', 'mpcpe_0.2', 'trig_0.05', 'replan_only', 'periodic', 'mpc2_0.05']
THRESHOLDS = ['mpc_0.02', 'mpc_0.05', 'mpc_0.1', 'mpcpe_0.1', 'mpcpe_0.2']     # appendix: E1 rot_sym jobs
FAMILY = ('H2b', 'H2c', 'H2d', 'H3')                                          # Holm-corrected together
RNG = np.random.default_rng(0)


def gmean_ci(values, n_boot=10000, level=0.95):
    v = np.log(np.asarray(values, float))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    bs = RNG.choice(v, (n_boot, len(v)), replace=True).mean(1)
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(np.exp(v.mean())), float(np.exp(lo)), float(np.exp(hi))


def per_setting(df, method, col='regret', censor_as=None):
    s = df[df.method == method].copy()
    if censor_as is not None:
        s.loc[s[col].isna(), col] = censor_as
    s = s[s[col].notna() & (s[col] > 0)]
    return s.groupby('setting')[col].apply(lambda x: float(np.exp(np.log(x.astype(float)).mean())))


def paired(df, a, b, col='T', H_fill=True):
    """Setting-level geometric-mean ratio T_a / T_b over jobs where both exist (censored -> H + 1)."""
    x = df[df.method == a].set_index('job_id')
    y = df[df.method == b].set_index('job_id')
    j = x.index.intersection(y.index)
    if len(j) == 0:
        return pd.Series(dtype=float)
    ta = x.loc[j, col].astype(float).fillna(x.loc[j, 'H'].astype(float) + 1) if H_fill else x.loc[j, col]
    tb = y.loc[j, col].astype(float).fillna(y.loc[j, 'H'].astype(float) + 1) if H_fill else y.loc[j, col]
    lr = np.log(ta / tb)
    return lr.groupby(x.loc[j, 'setting']).mean().apply(np.exp)


def one_sided(ratios, alternative):
    """Wilcoxon signed-rank p-value of log ratios over settings (needs >= 6 settings)."""
    v = np.log(np.asarray(ratios, float))
    v = v[np.isfinite(v)]
    if len(v) < 6 or not np.any(v != 0):
        return np.nan
    try:
        return float(wilcoxon(v, alternative=alternative).pvalue)
    except ValueError:
        return np.nan


def gm(r):
    return float(np.exp(np.log(np.asarray(r, float)).mean())) if len(r) else np.nan


def f3(x):
    return 'n/a' if x is None or x != x else f'{x:.3f}'


def holm(ps):
    """Holm-adjusted p-values (NaN entries stay NaN and are not counted)."""
    idx = [i for i, p in enumerate(ps) if p == p]
    out = [np.nan] * len(ps)
    prev = 0.0
    for k, i in enumerate(sorted(idx, key=lambda i: ps[i])):
        prev = min(1.0, max(prev, (len(idx) - k) * ps[i]))
        out[i] = prev
    return out


def summary_table(df, methods=MAIN):
    rows = []
    for (exp, regime), g in df.groupby(['exp', 'regime'], dropna=False):
        for m in methods:
            s = g[g.method == m]
            if not len(s):
                continue
            reg = s['regret'].dropna()
            gmn, lo, hi = gmean_ci(per_setting(g, m).values)
            rows.append(dict(exp=exp, regime=regime, method=m, jobs=len(s), settings=s.setting.nunique(),
                             censored=int(s['T'].isna().sum()), gmean=gmn, ci_lo=lo, ci_hi=hi,
                             median=reg.median() if len(reg) else np.nan, worst=reg.max() if len(reg) else np.nan,
                             within_2pct=float((reg <= 1.02).mean()) if len(reg) else np.nan,
                             relin=s['n_relin'].mean() if 'n_relin' in s and s['n_relin'].notna().any() else np.nan))
    return pd.DataFrame(rows)


def cost_table(df):
    """Wall-clock regret: (T x single-lane step time + the method's kernels and plans) / (T(r*) x step time),
    and the same wall time relative to a full fixed-r grid search."""
    rows = []
    for (exp, regime), g in df.groupby(['exp', 'regime'], dropna=False):
        for m in COST:
            s = g[(g.method == m)]
            if not len(s) or not s['wall_regret'].notna().any():
                continue
            gmn, lo, hi = gmean_ci(per_setting(g, m, col='wall_regret').values)
            w = s['wall_regret'].dropna()
            rows.append(dict(exp=exp, regime=regime, method=m, jobs=len(s), steps_regret=gm(per_setting(g, m).values),
                             wall_gmean=gmn, wall_ci_lo=lo, wall_ci_hi=hi, wall_median=w.median(), wall_worst=w.max(),
                             vs_grid_median=s['wall_vs_grid'].median(), relin=s['n_relin'].mean(),
                             ctrl_sec_median=s['ctrl_sec'].median() if 'ctrl_sec' in s else np.nan))
    return pd.DataFrame(rows)


def confirmatory(df, jt, exps=('E1', 'E2'), family=True, extensions=True):
    """Verdicts of the pre-registered hypotheses (see the experiment document, 'hypotheses').
    family=False: descriptive run on a subset (raw p-values, no Holm correction)."""
    out = []
    E = df[df.exp.isin(exps)]
    J = jt[jt.exp.isin(exps)]
    lazy, rich = E[E.regime == 'lazy'], E[E.regime == 'rich']
    jl, jr = J[J.regime == 'lazy'], J[J.regime == 'rich']

    def add(hid, text, value, support, reject, **kw):
        v = 'support' if support else ('reject' if reject else 'inconclusive')
        out.append(dict(id=hid, test=text, value=value, verdict=v, passed=(v == 'support'), **kw))

    # H1: P_lin2 predicts r* - exact in the lazy regime, valley position kept in the rich regime
    gmn, lo, hi = gmean_ci(per_setting(lazy, 'plin2').values)
    share1 = float((jl.argmin_offset <= 1).mean()) if len(jl) else np.nan
    rho = float(jr.spearman.median()) if len(jr) else np.nan
    share2 = float((jr.argmin_offset <= 2).mean()) if len(jr) else np.nan
    far = float((jr.argmin_offset > 2).mean()) if len(jr) else np.nan
    lazy_ok = hi == hi and hi <= 1.05 and share1 >= 0.9
    rich_ok = rho == rho and rho >= 0.9 and share2 >= 0.8
    add('H1', 'P_lin2: lazy gmean regret CI upper <= 1.05 and offset <= 1 step in >= 90% of jobs; rich median '
              'Spearman(T, T_lin) >= 0.9 and offset <= 2 steps in >= 80%  |  reject: lazy CI upper > 1.10 or '
              'offset > 2 steps in > 50% of rich jobs',
        f'lazy gmean {f3(gmn)} [{f3(lo)}, {f3(hi)}], offset<=1 {share1:.0%} ({len(jl)} jobs); rich Spearman '
        f'{f3(rho)}, offset<=2 {share2:.0%}, >2 {far:.0%} ({len(jr)} jobs)',
        lazy_ok and rich_ok, (hi == hi and hi > 1.10) or (far == far and far > 0.5))
    # H2a: the main method is close to or better than the best fixed r everywhere
    gmn, lo, hi = gmean_ci(per_setting(E, 'mpc_0.05').values)
    worst = per_setting(E, 'mpc_0.05').max()
    add('H2a', 'mpc_0.05: gmean regret CI upper <= 1.02 and worst setting <= 1.10  |  reject: CI lower > 1.05',
        f'gmean {f3(gmn)} [{f3(lo)}, {f3(hi)}], worst setting {f3(worst)}',
        hi == hi and hi <= 1.02 and worst <= 1.10, lo == lo and lo > 1.05)
    # H2b-H2d: paired one-sided Wilcoxon over settings, Holm over the family (with H3)
    tests = [('H2b', rich, 'mpc_0.05', 'plin2', 'less', 'rich: T(mpc_0.05) < T(P_lin2)'),
             ('H2c', rich, 'replan_only', 'mpc_0.05', 'greater', 'rich: T(replan_only) > T(mpc_0.05)'),
             ('H2d', E, 'mpc_0.05', 'trig_0.05', 'less', 'all: T(mpc_0.05) < T(trig_0.05)')]
    for hid, data, a, b, alt, text in tests:
        r = paired(data, a, b)
        opp = 'greater' if alt == 'less' else 'less'
        add(hid, f'{text}  [Wilcoxon one-sided{", Holm" if family else ""}]  |  reject: significant the other way',
            f'gmean ratio {f3(gm(r))}, n={len(r)} settings', False, False,
            p=one_sided(r.values, alt), p_opp=one_sided(r.values, opp), confirmatory=True)
    # H3: two-phase plans beat constant-r plans in the lazy regime (E1 only: mpc2 runs there)
    r = paired(lazy, 'mpc2_0.05', 'mpc_0.05')
    s = lazy[(lazy.method == 'mpc2_0.05') & lazy.first_up.notna()]
    up = float(s.first_up.astype(float).mean()) if len(s) else np.nan
    if len(r):
        add('H3', f'lazy: gmean T(mpc2_0.05) / T(mpc_0.05) <= 0.97 [Wilcoxon{", Holm" if family else ""}] and first '
                  'two-phase plan low->high in >= 70%  |  reject: ratio >= 1.0',
            f'gmean ratio {f3(gm(r))}, n={len(r)} settings, low->high {up:.0%}', False, gm(r) >= 1.0,
            p=one_sided(r.values, 'less'), p_opp=None, confirmatory=True,
            extra_ok=bool(gm(r) <= 0.97 and up >= 0.7))
    # Holm over the confirmatory family, then the verdicts of the family members
    fam = [o for o in out if o.get('confirmatory')]
    adj = holm([o['p'] if o['p'] is not None else np.nan for o in fam]) if family else [o['p'] for o in fam]
    adj_opp = holm([o['p_opp'] if o.get('p_opp') is not None else np.nan for o in fam]) if family \
        else [o.get('p_opp') for o in fam]
    for o, pa, po in zip(fam, adj, adj_opp):
        o['p_holm'] = pa if family else None
        support = pa is not None and pa == pa and pa < 0.05 and o.get('extra_ok', True)
        reject = o['verdict'] == 'reject' or (po is not None and po == po and po < 0.05)
        o['verdict'] = 'support' if support else ('reject' if reject else 'inconclusive')
        o['passed'] = o['verdict'] == 'support'
    # H2f: the Jacobian-free trigger is equivalent to the drift trigger
    r = paired(E, 'mpcpe_0.2', 'mpc_0.05')
    gmn, lo, hi = gmean_ci(r.values, level=0.90)
    add('H2f', 'mpcpe_0.2 vs mpc_0.05: 90% CI of gmean T ratio inside [0.97, 1.03]  |  reject: CI excludes 1 and '
               'the ratio is off by > 5%',
        f'{f3(gmn)} [{f3(lo)}, {f3(hi)}], n={len(r)} settings', lo == lo and lo >= 0.97 and hi <= 1.03,
        gmn == gmn and (lo > 1 or hi < 1) and abs(gmn - 1) > 0.05)
    # H5: in the lazy regime the schedule does not change generalisation
    dl = float(jl.valley_accB_range.median()) if len(jl) else np.nan
    dr = float(jr.valley_accB_range.median()) if len(jr) else np.nan
    add('H5', 'valley (T <= 1.2 T*) spread of task-B test accuracy: lazy median <= 1 point and below rich  |  '
              'reject: lazy median > 2 points',
        f'lazy {f3(dl)} vs rich {f3(dr)}', dl == dl and dr == dr and dl <= 0.01 and dl < dr, dl == dl and dl > 0.02)
    if extensions:
        out += h6(df, jt)
    return out


def h6(df, jt):
    out = []
    w = jt[jt.exp == 'E1W']
    if len(w):
        rows = []
        for h, g in w.groupby('width'):
            T = g.T_best.astype(float).fillna(g.H.astype(float) + 1)
            rows.append(dict(width=h, reached=float(g.T_best.notna().mean()), T=float(np.exp(np.log(T).mean()))))
        t = pd.DataFrame(rows).sort_values('width')
        ok = t[t.reached >= 0.5]
        pl = df[(df.exp == 'E1W') & (df.method == 'plin2')].regret.dropna()
        if len(ok):
            peak = float(ok.loc[ok['T'].idxmax(), 'width'])
            no_peak = peak == float(t.width.min())
            v = (f'T* largest at h={peak:g} among widths reaching the band (unreached: '
                 f'{[int(x) for x in t[t.reached < 0.5].width]}); P_lin2 regret median {f3(pl.median())}, max {f3(pl.max())}')
            support = 16 <= peak <= 40 and len(pl) and pl.max() <= 1.05
            reject = peak >= 64 or no_peak
        else:
            v, support, reject = 'no width reached the band in most seeds', False, False
        out.append(dict(id='H6a', test='width sweep (tight band): T* peaks at h in [16, 40]; P_lin2 regret <= 1.05 at '
                                       'every width  |  reject: peak at h >= 64 or no peak (largest at the narrowest width)',
                        value=v, verdict='support' if support else ('reject' if reject else 'inconclusive'),
                        passed=bool(support)))
    n = jt[(jt.exp == 'E1N') & jt.r_best.notna()]
    if len(n) >= 6:
        rho = spearmanr(1.0 / n.batch, n.r_best).correlation
        m_ = df[(df.exp == 'E1N') & (df.method == 'mpc_0.05')].regret
        support = rho == rho and rho > 0 and m_.median() <= 1.05
        reject = rho == rho and rho < 0
        out.append(dict(id='H6b', test='SGD noise: r* increases with 1/B (Spearman > 0); mpc_0.05 regret median <= 1.05'
                                       '  |  reject: Spearman < 0',
                        value=f'Spearman(1/B, r*) {f3(rho)}; mpc_0.05 median {f3(m_.median())}',
                        verdict='support' if support else ('reject' if reject else 'inconclusive'), passed=bool(support)))
    return out


def mechanism(jt):
    J = jt[jt.exp.isin(['E1', 'E2'])]
    if not len(J):
        return pd.DataFrame()
    return J.groupby(['exp', 'regime']).agg(
        jobs=('job_id', 'count'), drift=('drift', 'median'), spearman=('spearman', 'median'),
        offset_le1=('argmin_offset', lambda x: (x <= 1).mean()), time_scale=('time_scale', 'median'),
        ratio_cv=('ratio_cv', 'median'), accB_valley_range=('valley_accB_range', 'median')).reset_index()


def grid_accuracy(jt):
    """Exploratory (no verdict): does less replay reach a better joint solution? Test accuracy at T of every r
    that reached the band, relative to r*."""
    J = jt[jt.exp.isin(['E1', 'E2']) & (jt.grid_n > 0)]
    rows = []
    for (exp, regime), g in J.groupby(['exp', 'regime']):
        rows.append(dict(exp=exp, regime=regime, jobs=len(g), r_in_band=g.grid_n.median(),
                         spearman_r_acc=g.spearman_r_acc.median(), d_acc_smallest_r=g.d_acc_small_r.median(),
                         d_acc_best_r=g.d_acc_best_r.median(),
                         best_r_below_rstar=float((g.r_acc_best < g.r_best).mean()),
                         d_mse_smallest_r=g.d_mse_small_r.median()))
    return pd.DataFrame(rows)


def extensions(df, jt):
    out = {}
    w = jt[jt.exp == 'E1W']
    if len(w):
        t = w.groupby(['tag', 'width']).agg(
            T_best=('T_best', lambda x: float(np.exp(np.log(x.dropna()).mean())) if x.notna().any() else np.nan),
            reached=('T_best', lambda x: float(x.notna().mean())),
            accB=('acc_B_best', 'mean'), accA=('acc_A_best', 'mean'), drift=('drift', 'median'))
        pl = df[(df.exp == 'E1W') & (df.method == 'plin2')].groupby(['tag', 'width']).regret.median()
        out['E1W'] = t.join(pl.rename('plin2_regret')).reset_index()
    n = jt[jt.exp == 'E1N']
    if len(n):
        rows = []
        for (al, B), g in n.groupby(['alpha', 'batch']):
            d = df[(df.exp == 'E1N') & (df.alpha == al) & (df.batch == B)]
            rows.append(dict(alpha=al, batch=B, r_best_mean=g.r_best.mean(), r_lin=g.r_lin.mean(),
                             plin2=d[d.method == 'plin2'].regret.median(), mpc=d[d.method == 'mpc_0.05'].regret.median(),
                             mpcpe=d[d.method == 'mpcpe_0.2'].regret.median(),
                             mpcpe_relin=d[d.method == 'mpcpe_0.2'].n_relin.mean(),
                             censored=int(g.T_best.isna().sum())))
        out['E1N'] = pd.DataFrame(rows)
    return out


def appendix(df):
    out = {}
    t = df[(df.exp == 'E1') & (df.tag == 'rot_sym') & (df.width == 1024) & (df.nA + df.nB == 2000)]
    if len(t) and t.method.isin(['mpc_0.02', 'mpc_0.1', 'mpcpe_0.1']).any():
        out['Appendix: threshold sensitivity (E1 rot_sym)'] = summary_table(t, THRESHOLDS)
    s = df[df.method == 'normbal']
    if len(s):
        rows = []
        for (exp, regime), g in s.groupby(['exp', 'regime'], dropna=False):
            rows.append(dict(exp=exp, regime=regime, jobs=len(g), censored=int(g['T'].isna().sum()),
                             stall_share=float((g['T'].isna() | (g['regret'] >= 3)).mean()),
                             regret_median=g['regret'].median(), regret_worst=g['regret'].max()))
        out['Appendix: equal-norm ratio rule (normbal), descriptive'] = pd.DataFrame(rows)
    return out


def verdict_lines(conf):
    lines = []
    for o in conf:
        pv = ''
        if o.get('p') is not None and o['p'] == o['p']:
            pv = f"  p={o['p']:.3g}" + (f", Holm {o['p_holm']:.3g}" if o.get('p_holm') is not None and o['p_holm'] == o['p_holm'] else '')
        if o.get('p_opp') is not None and o['p_opp'] == o['p_opp']:
            pv += f"; other way p={o['p_opp']:.3g}"
        lines.append(f"- **{o['verdict'].upper()}** {o['id']}: {o['test']}. Value: {o['value']}{pv}")
    return lines


def md(t):
    return t.to_markdown(index=False, floatfmt='.3f') if len(t) else '(no data)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None)
    ap.add_argument('--exp', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'summary'))
    a = ap.parse_args()
    jobs = load_results(a.results, a.exp)
    if not jobs:
        print('no results found')
        return
    os.makedirs(a.out, exist_ok=True)
    df, jt = method_table(jobs), job_table(jobs)
    df.to_csv(os.path.join(a.out, 'methods.csv'), index=False)
    jt.to_csv(os.path.join(a.out, 'jobs.csv'), index=False)
    st = summary_table(df)
    st.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    ct = cost_table(df)
    ct.to_csv(os.path.join(a.out, 'cost.csv'), index=False)
    conf = confirmatory(df, jt)
    conf_e2 = confirmatory(df, jt, exps=('E2',), family=False, extensions=False) if (df.exp == 'E2').any() else []
    lines = [f'# Summary ({len(jobs)} jobs)\n']
    for (exp, regime), g in st.groupby(['exp', 'regime'], dropna=False):
        lines.append(f'\n## {exp} / {regime}\n')
        lines.append(md(g.drop(columns=['exp', 'regime'])))
    lines.append('\n## Pre-registered tests (E1 + E2; H6 on E1W / E1N)\n')
    lines += verdict_lines(conf)
    if conf_e2:
        lines.append('\n## E2 alone (descriptive: raw p-values, no Holm correction)\n')
        lines += verdict_lines(conf_e2)
    lines.append('\n## Cost: wall-clock regret (T x single-lane step + the method\'s kernels and plans) / (T(r*) x step); '
                 'vs_grid = the same wall time / a full fixed-r grid search\n')
    lines.append(md(ct))
    lines.append('\n## Mechanism (per job; ratio_cv = CV of T / T_lin over the grid, descriptive)\n')
    lines.append(md(mechanism(jt)))
    lines.append('\n## Exploratory: test accuracy at T over the whole r grid, relative to r* (no verdict)\n')
    lines.append(md(grid_accuracy(jt)))
    for k, v in extensions(df, jt).items():
        lines.append(f'\n## {k}\n')
        lines.append(md(v))
    for k, v in appendix(df).items():
        lines.append(f'\n## {k}\n')
        lines.append(md(v))
        v.to_csv(os.path.join(a.out, ('thresholds' if 'threshold' in k else 'normbal') + '.csv'), index=False)
    txt = '\n'.join(lines)
    open(os.path.join(a.out, 'summary.md'), 'w').write(txt)
    json.dump(conf, open(os.path.join(a.out, 'tests.json'), 'w'), indent=1, default=str)
    json.dump(conf_e2, open(os.path.join(a.out, 'tests_e2.json'), 'w'), indent=1, default=str)
    print(txt)


if __name__ == '__main__':
    main()
