"""Does a probe + re-linearisation pick a better replay ratio than the lazy prediction?

For one (scenario, alpha, seed) this runs
  oracle      : real training for every r on the grid (ground truth T(r))
  P_lin2      : linearise both tasks at theta_A, pick r once
  probe_swap  : short real probe at r_lin2, recompute the kernel there, re-predict T(r) from the
                start with the new kernel, pick r once (offline, the version proposed earlier)
  probe_piece : same, but old kernel for the probe length and new kernel afterwards
  probe_cont  : keep the probe's progress, re-plan once from the probe state (one online re-plan)
  trig_e      : online: every M steps measure kernel drift on a fixed sample subset since the last
                linearisation; if drift > e, re-linearise at the current point and re-plan r for the
                rest of the run; otherwise keep r (the user's version)
  periodic    : online: re-linearise and re-plan at every check (drift threshold 0)
T = first step at which both tasks are inside the tolerance band. Regret = T / best fixed-r T.
"""
import numpy as np, json, sys, time
from core import Problem

RS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]


def run_config(scen, alpha, seed, H=5000, eps_list=(0.05, 0.2), probe_frac=0.15, n_checks=10):
    t0 = time.time()
    P = Problem(seed=seed, alpha=alpha, **scen)
    K0 = P.kernel(P.net.theta0)
    m = min(P.nA, P.nB)
    P.train_A(0.9 * m / np.linalg.eigvalsh(K0)[-1])
    KA = P.kernel(P.thetaA)
    eta = 0.9 * m / np.linalg.eigvalsh(KA)[-1]
    EA = (P.FA - P.Y).reshape(-1)
    res = dict(alpha=alpha, seed=seed)

    # ground truth
    truth = {}
    for r in RS:
        o = P.run_real(eta, H, r)
        truth[r] = (o['T'] if o['T'] is not None else -1, o['peak'])
    ok = {r: v[0] for r, v in truth.items() if v[0] >= 0}
    r_best = min(ok, key=ok.get); T_best = ok[r_best]
    res.update(truth={r: v[0] for r, v in truth.items()}, r_best=r_best, T_best=T_best)
    o = P.run_real(eta, H, r_best)
    res['kernel_drift_best_run'] = float(np.linalg.norm(P.kernel(o['th']) - KA) / np.linalg.norm(KA))

    def T_of(r):
        return truth[r][0] if truth[r][0] >= 0 else None

    # P_lin2
    r_lin, That = P.plan([(KA, 0)], EA, RS, eta, H)
    res['P_lin2'] = dict(r=r_lin, T=T_of(r_lin), T_pred=That)

    # offline probe at r_lin
    Kp_steps = max(30, int(probe_frac * (That if That else H)))
    probe = P.run_real(eta, Kp_steps, r_lin)
    th_p = probe['th']
    Kp = P.kernel(th_p)
    res['probe_steps'] = Kp_steps
    res['probe_kernel_drift'] = float(np.linalg.norm(Kp - KA) / np.linalg.norm(KA))
    r_swap, _ = P.plan([(Kp, 0)], EA, RS, eta, H)
    r_piece, _ = P.plan([(KA, Kp_steps), (Kp, 0)], EA, RS, eta, H)
    res['probe_swap'] = dict(r=r_swap, T=T_of(r_swap))
    res['probe_piece'] = dict(r=r_piece, T=T_of(r_piece))
    if probe['T'] is not None:            # band already reached during the probe
        res['probe_cont'] = dict(r=r_lin, T=probe['T'])
    else:
        Ep = (P.F(th_p) - P.Y).reshape(-1)
        r_cont, _ = P.plan([(Kp, 0)], Ep, RS, eta, H - Kp_steps)
        o = P.run_real(eta, H - Kp_steps, r_cont, start=th_p)
        res['probe_cont'] = dict(r=r_cont, T=(Kp_steps + o['T']) if o['T'] is not None else None)

    # online controllers
    M = max(20, int((That if That else H) / n_checks))
    rng = np.random.default_rng(seed)
    sub = np.concatenate([rng.choice(np.where(P.isA)[0], 30, replace=False),
                          rng.choice(np.where(~P.isA)[0], 30, replace=False)])
    for name, eps in [(f'trig_{e}', e) for e in eps_list] + [('periodic', 0.0)]:
        state = dict(Kref=P.kernel(P.thetaA, sub), n_relin=0, drifts=[])

        def on_check(t, th, F, eps=eps, state=state):
            Ks = P.kernel(th, sub)
            drift = float(np.linalg.norm(Ks - state['Kref']) / np.linalg.norm(state['Kref']))
            state['drifts'].append(round(drift, 3))
            if drift > eps:
                state['Kref'] = Ks; state['n_relin'] += 1
                r_new, _ = P.plan([(P.kernel(th), 0)], (F - P.Y).reshape(-1), RS, eta, H - t)
                return r_new
            return None
        o = P.run_real(eta, H, r_lin, check_every=M, on_check=on_check)
        res[name] = dict(T=o['T'], n_relin=state['n_relin'], r_trace=o['r_trace'],
                         max_drift=max(state['drifts']) if state['drifts'] else 0.0)
    res['check_every'] = M
    for k in ['P_lin2', 'probe_swap', 'probe_piece', 'probe_cont'] + [f'trig_{e}' for e in eps_list] + ['periodic']:
        T = res[k]['T']
        res[k]['regret'] = (T / T_best) if T is not None else None
    res['sec'] = round(time.time() - t0)
    return res


if __name__ == '__main__':
    import scan
    name = sys.argv[1]; alpha = float(sys.argv[2]); seed = int(sys.argv[3])
    out = run_config(scan.SCEN[name], alpha, seed)
    out['scen'] = name
    print(json.dumps(out), flush=True)
