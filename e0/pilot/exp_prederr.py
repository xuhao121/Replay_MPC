"""Jacobian-free trigger for the online controller (pilot check before the GPU runs).

Same controller as trig_replan_eps in exp_extra.py (re-plan r at every check from the current
state, re-linearise only when triggered), but the trigger is the linear model's own one-window
prediction error instead of kernel drift on a sample subset:

    delta_k = || E_real(t_k + M) - E_pred(t_k + M) || / || E_real(t_k + M) - E_real(t_k) ||

where E_pred is the linearised function-space prediction made at t_k with the current reference
kernel and the r chosen at t_k. delta needs no Jacobian; a kernel is computed only when
delta > eps. Regret uses T_best from results.jsonl (same seeds -> identical problems)."""
import numpy as np, json, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool


def predict(P, K, E0, r, eta, M):
    w = np.repeat(P.weights(r), 10).astype(np.float32)
    E = E0.astype(np.float32).copy()
    K = K.astype(np.float32)
    for _ in range(M):
        E -= eta * (K @ (w * E))
    return E


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
    EA = (P.FA - P.Y).reshape(-1)
    out = dict(scen=row['scen'], alpha=row['alpha'], seed=row['seed'], T_best=row['T_best'])
    for eps in eps_list:
        state = dict(K=KA, n_relin=0, E_prev=EA, E_pred=predict(P, KA, EA, r_lin, eta, M), deltas=[])

        def on_check(t, th, F, eps=eps, state=state):
            E = (F - P.Y).reshape(-1)
            delta = float(np.linalg.norm(E - state['E_pred']) / max(np.linalg.norm(E - state['E_prev']), 1e-12))
            state['deltas'].append(round(delta, 3))
            if delta > eps:
                state['K'] = P.kernel(th); state['n_relin'] += 1
            r_new, _ = P.plan([(state['K'], 0)], E, RS, eta, H - t)
            state['E_prev'] = E
            state['E_pred'] = predict(P, state['K'], E, r_new, eta, M)
            return r_new
        o = P.run_real(eta, H, r_lin, check_every=M, on_check=on_check)
        out[f'pe_replan_{eps}'] = dict(T=o['T'], regret=(o['T'] / row['T_best']) if o['T'] else None,
                                       n_relin=state['n_relin'], r_trace=o['r_trace'], deltas=state['deltas'])
    return out


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('results.jsonl') if l.strip()]
    rows = [r for r in rows if 'error' not in r]
    eps_list = [0.05, 0.1, 0.2]
    with Pool(2) as pool, open('results_prederr.jsonl', 'w') as f:
        for out in pool.imap_unordered(work, [(r, eps_list) for r in rows]):
            f.write(json.dumps(out) + '\n'); f.flush()
            print(json.dumps({k: (v['regret'], v['n_relin']) if isinstance(v, dict) else v for k, v in out.items()}), flush=True)
    print('PREDERR DONE', flush=True)
