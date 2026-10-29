"""Extra online variant: re-plan from the current state at every check using the current kernel,
but recompute (re-linearise) the kernel only when the subset-kernel drift exceeds eps.
This separates the two possible benefits of online control:
  (1) correcting kernel drift (needs re-linearisation), and
  (2) a time-varying r from state feedback (needs only re-planning).
Regret uses T_best from results.jsonl (same seeds -> identical problems)."""
import numpy as np, json, sys, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool


def work(args):
    import scan
    from core import Problem
    from exp import RS
    row, eps_list = args
    P = Problem(seed=row['seed'], alpha=row['alpha'], **scan.SCEN[row['scen']])
    m = min(P.nA, P.nB)
    P.train_A(0.9 * m / np.linalg.eigvalsh(P.kernel(P.net.theta0))[-1])
    KA = P.kernel(P.thetaA)
    eta = 0.9 * m / np.linalg.eigvalsh(KA)[-1]
    H = 5000
    r_lin = row['P_lin2']['r']
    M = row['check_every']
    rng = np.random.default_rng(row['seed'])
    sub = np.concatenate([rng.choice(np.where(P.isA)[0], 30, replace=False),
                          rng.choice(np.where(~P.isA)[0], 30, replace=False)])
    out = dict(scen=row['scen'], alpha=row['alpha'], seed=row['seed'], T_best=row['T_best'])
    for eps in eps_list:
        state = dict(Kref_sub=P.kernel(P.thetaA, sub), K=KA, n_relin=0)

        def on_check(t, th, F, eps=eps, state=state):
            Ks = P.kernel(th, sub)
            drift = float(np.linalg.norm(Ks - state['Kref_sub']) / np.linalg.norm(state['Kref_sub']))
            if drift > eps:
                state['Kref_sub'] = Ks; state['K'] = P.kernel(th); state['n_relin'] += 1
            r_new, _ = P.plan([(state['K'], 0)], (F - P.Y).reshape(-1), RS, eta, H - t)
            return r_new
        o = P.run_real(eta, H, r_lin, check_every=M, on_check=on_check)
        name = 'replan_only' if eps == np.inf else f'trig_replan_{eps}'
        out[name] = dict(T=o['T'], regret=(o['T'] / row['T_best']) if o['T'] else None,
                         n_relin=state['n_relin'], r_trace=o['r_trace'])
    return out


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('results.jsonl') if l.strip()]
    rows = [r for r in rows if 'error' not in r]
    eps_list = [0.05, 0.2, np.inf]
    with Pool(2) as pool, open('results_extra.jsonl', 'w') as f:
        for out in pool.imap_unordered(work, [(r, eps_list) for r in rows]):
            f.write(json.dumps(out) + '\n'); f.flush()
            print(json.dumps({k: (v['regret'], v['n_relin']) if isinstance(v, dict) else v for k, v in out.items()}), flush=True)
    print('EXTRA DONE', flush=True)
