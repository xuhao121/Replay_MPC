"""Load job results into flat tables (pandas)."""
import glob, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {'_claims', '_failed', '_logs', '_bench', 'manual', 'summary', 'figures'}
LAZY_DRIFT = 0.2          # pre-registered: lazy if kernel drift along the best fixed-r run < 0.2

# categorical slots (fixed order, never cycled)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

ONLINE = ('mpc', 'trig', 'periodic', 'replan')      # name prefixes of the online (MPC-type) controllers


def load_results(results_dir=None, exps=None):
    results_dir = results_dir or os.path.join(ROOT, 'results')
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, '*', '*.json'))):
        if os.path.basename(os.path.dirname(p)) in SKIP:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or 'job_id' not in d or 'cfg' not in d:
            continue                              # not a job result (e.g. results/summary/tests.json)
        if exps and d['cfg']['exp'] not in exps:
            continue
        out[d['job_id']] = d                      # duplicates (both owners ran a job) collapse here
    return list(out.values())


def drift_of(d):
    dr = d.get('drift') or {}
    return dr.get('full_best', dr.get('sub_best'))


def regime_of(drift):
    if drift is None or drift != drift:
        return None
    return 'lazy' if drift < LAZY_DRIFT else 'rich'


def setting_key(cfg):
    keys = ['exp', 'tag', 'model', 'width', 'alpha', 'kernel', 'transform', 'nA', 'nB', 'tolA', 'tolB', 'batch_size']
    return '|'.join(str(cfg[k]) for k in keys)


def wall_time(d, name, m, T):
    """Wall time a practitioner would spend with this method in one training run (seconds): T single-lane steps
    plus the method's own overhead - K_A and the P_lin2 prediction for every planning method, the subset kernel
    at theta_A for drift triggers, and the controller's measured kernels and plans. None if not measured."""
    tm = d.get('timing') or {}
    step = tm.get('step_sec')
    if step is None or T is None:
        return None
    if name.startswith('fixed_'):
        return T * step
    shared = (tm.get('KA_sec') or 0.0) + (tm.get('plin2_sec') or 0.0)
    if name == 'plin2':
        return T * step + shared
    if name.startswith(ONLINE) and m.get('ctrl_sec') is not None:
        sub = (tm.get('KsA_sec') or 0.0) if m.get('trigger') == 'drift' else 0.0
        return T * step + shared + sub + m['ctrl_sec']
    return None


def method_table(jobs):
    rows = []
    for d in jobs:
        cfg = d['cfg']
        o = d['oracle']
        step = (d.get('timing') or {}).get('step_sec')
        best_wall = o['T_best'] * step if (step and o['T_best']) else None
        grid_wall = sum((t if t is not None else d['H']) for t in o['T']) * step if step else None
        base = dict(job_id=d['job_id'], setting=setting_key(cfg), exp=cfg['exp'], tag=cfg['tag'], model=cfg['model'],
                    width=cfg['width'], alpha=cfg['alpha'], seed=cfg['seed'], transform=cfg['transform'],
                    nA=cfg['nA'], nB=cfg['nB'], tolA=cfg['tolA'], tolB=cfg['tolB'], batch=cfg['batch_size'],
                    kernel=d['problem']['kernel'], drift=drift_of(d), r_best=o['r_best'], T_best=o['T_best'],
                    T_hat=d['T_hat'], r_lin=d['r_lin'], H=d['H'], M=d['M'], time_sec=d['time_sec'], step_sec=step)
        base['regime'] = regime_of(base['drift'])

        def walls(name, m, T):
            w = wall_time(d, name, m, T)
            return dict(wall_sec=w, wall_regret=(w / best_wall) if (w is not None and best_wall) else None,
                        wall_vs_grid=(w / grid_wall) if (w is not None and grid_wall) else None)
        if 0.5 in o['r']:
            T05 = o['T'][o['r'].index(0.5)]
            rows.append(dict(base, method='fixed_0.5', T=T05,
                             regret=(T05 / o['T_best']) if (T05 and o['T_best']) else None,
                             **walls('fixed_0.5', {}, T05)))
        for name, m in d['methods'].items():
            test = m.get('test') or {}
            p2 = [p for p in (m.get('plans2') or []) if p[1] != p[2]]
            rows.append(dict(base, method=name, T=m.get('T'), regret=m.get('regret'), regret_edge=m.get('regret_edge'),
                             r=m.get('r'), n_relin=m.get('n_relin'), n_plans=m.get('n_plans'), peak=m.get('peak'),
                             diverged=m.get('diverged'), acc_A=test.get('acc_A'), acc_B=test.get('acc_B'),
                             first_up=(p2[0][1] < p2[0][2]) if p2 else None, trigger=m.get('trigger'),
                             ctrl_sec=m.get('ctrl_sec'), **walls(name, m, m.get('T'))))
    df = pd.DataFrame(rows)
    if len(df):
        df['censored'] = df['T'].isna()
        for c in ('n_relin', 'n_plans', 'wall_sec', 'wall_regret', 'wall_vs_grid', 'ctrl_sec', 'step_sec'):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _grid_accuracy(o, r):
    """Exploratory: test performance at T over every r that reached the band (including r outside the valley)."""
    from scipy.stats import spearmanr
    tests = o.get('test') or []
    pts = [(r[i], 0.5 * (s['acc_A'] + s['acc_B']), s['mse_A'] + s['mse_B'])
           for i, s in enumerate(tests) if s and s.get('acc_A') is not None]
    out = dict(grid_n=len(pts), spearman_r_acc=np.nan, d_acc_small_r=np.nan, d_acc_best_r=np.nan,
               d_mse_small_r=np.nan, r_acc_best=np.nan)
    if not pts or o['r_best'] is None:
        return out
    star = [p for p in pts if p[0] == o['r_best']]
    if not star:
        return out
    _, acc_star, mse_star = star[0]
    small = min(pts, key=lambda p: p[0])
    best = max(pts, key=lambda p: (p[1], -abs(p[0] - o['r_best'])))
    out.update(d_acc_small_r=small[1] - acc_star, d_mse_small_r=small[2] - mse_star,
               d_acc_best_r=best[1] - acc_star, r_acc_best=best[0])
    if len(pts) >= 4:
        out['spearman_r_acc'] = spearmanr([p[0] for p in pts], [p[1] for p in pts]).correlation
    return out


def job_table(jobs):
    """One row per job with mechanism measurements."""
    from scipy.stats import spearmanr
    rows = []
    for d in jobs:
        cfg, o = d['cfg'], d['oracle']
        r = o['r']
        Tr = np.array([np.nan if t is None else t for t in o['T']], float)
        Tl = np.array([np.nan if (t is None or t < 0) else t for t in d['plin2_curve']['T_lin']], float)
        ok = np.isfinite(Tr) & np.isfinite(Tl)
        rho = spearmanr(Tr[ok], Tl[ok]).correlation if ok.sum() >= 4 else np.nan
        step = (r[1] - r[0]) if len(r) > 1 else 0.05
        off = abs(d['r_lin'] - o['r_best']) / step if o['r_best'] is not None else np.nan
        scale = (o['T_best'] / d['T_hat']) if (o['T_best'] and d['T_hat']) else np.nan
        ratio = Tr[ok] / Tl[ok]
        valley = [i for i, t in enumerate(Tr) if np.isfinite(t) and o['T_best'] and t <= 1.2 * o['T_best']]
        accs = [(o['test'][i] or {}).get('acc_B') for i in valley] if o.get('test') else []
        accs = [a for a in accs if a is not None]
        accA = [(o['test'][i] or {}).get('acc_A') for i in valley] if o.get('test') else []
        accA = [a for a in accA if a is not None]
        rows.append(dict(job_id=d['job_id'], setting=setting_key(cfg), exp=cfg['exp'], tag=cfg['tag'],
                         width=cfg['width'], alpha=cfg['alpha'], seed=cfg['seed'], batch=cfg['batch_size'],
                         tolB=cfg['tolB'], H=d['H'], drift=drift_of(d), r_best=o['r_best'], r_lin=d['r_lin'],
                         T_best=o['T_best'], T_hat=d['T_hat'], spearman=rho, argmin_offset=off, time_scale=scale,
                         ratio_cv=float(np.std(ratio) / np.mean(ratio)) if len(ratio) > 1 else np.nan,
                         valley_n=len(valley), valley_accB_range=(max(accs) - min(accs)) if accs else np.nan,
                         valley_accA_range=(max(accA) - min(accA)) if accA else np.nan,
                         acc_A_best=((o['test'][r.index(o['r_best'])] or {}).get('acc_A') if o['r_best'] is not None and o.get('test') else np.nan),
                         acc_B_best=((o['test'][r.index(o['r_best'])] or {}).get('acc_B') if o['r_best'] is not None and o.get('test') else np.nan),
                         time_sec=d['time_sec'], **_grid_accuracy(o, r)))
    df = pd.DataFrame(rows)
    if len(df):
        df['drift'] = pd.to_numeric(df['drift'], errors='coerce')
        df['regime'] = [regime_of(x) for x in df['drift']]
    return df
