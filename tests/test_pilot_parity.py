"""The torch implementation must reproduce the numpy CPU pilot (e0/pilot/core.py) exactly on sklearn digits:
same data, init, kernels, step sizes, phase-A solution, linearised prediction and real hitting times T(r).
Run:  python -m pytest -q tests/test_pilot_parity.py      (about 1-2 min on CPU)
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'e0', 'pilot'))

from rmpc.config import JobConfig, PILOT_GRID
from rmpc.problem import TwoTaskProblem
from rmpc.trainer import train_A, run_lanes
from rmpc.linsim import predict_curve, choose
from rmpc.controllers import Fixed
import core as pilot


class _Ctx:
    pass


def _setup(alpha, h=256, transform='rot90', seed=0):
    cfg = JobConfig(exp='parity', dataset='digits', nA=150, nB=150, n_test=50, transform=transform,
                    model='mlp', width=h, alpha=alpha, init='numpy_pilot', dtype='float64', seed=seed,
                    r_grid=PILOT_GRID, H=5000)
    prob = TwoTaskProblem(cfg, torch.device('cpu'))
    P = pilot.Problem(seed=seed, alpha=alpha, nA=150, nB=150, transform=transform, h=h)
    return cfg, prob, P


def check(alpha, rs=(0.4, 0.5, 0.6), h=256):
    torch.set_num_threads(2)
    cfg, prob, P = _setup(alpha, h)
    assert np.allclose(prob.X.numpy(), P.X, atol=1e-12), 'data differ'
    assert np.allclose(prob.Y.numpy(), P.Y), 'targets differ'
    assert np.allclose(prob.theta0['W1'].numpy(), P.net.theta0['W1']), 'init differs'
    K0 = prob.kernel(prob.theta0).mat.numpy()
    K0p = P.kernel(P.net.theta0)
    assert np.allclose(K0, K0p, rtol=1e-10, atol=1e-10), 'NTK at theta0 differs'
    m = min(P.nA, P.nB)
    etaA = 0.9 * m / np.linalg.eigvalsh(K0p)[-1]
    stepsA_p = P.train_A(etaA)
    LB_theta0 = float(0.5 * (prob.Y[~prob.isA] ** 2).sum() / prob.nB)
    thA, info = train_A(prob, 0.9 * m / prob.kernel(prob.theta0).lam_max(), 20000, 2e-3 * LB_theta0)
    assert info['steps'] == stepsA_p, (info['steps'], stepsA_p)
    assert np.allclose(thA['W1'].numpy(), P.thetaA['W1'], rtol=1e-8, atol=1e-10), 'theta_A differs'
    KA = prob.kernel(thA)
    eta = 0.9 * m / KA.lam_max()
    KAp = P.kernel(P.thetaA)
    eta_p = 0.9 * m / np.linalg.eigvalsh(KAp)[-1]
    assert abs(eta - eta_p) / eta_p < 1e-10
    EA = prob.F(thA) - prob.Y
    LA0, LB0 = [float(x) for x in prob.losses(EA)]
    tauA, tauB = LA0 + 0.03 * LB0, 0.1 * LB0
    assert abs(tauA - P.tauA) < 1e-12 and abs(tauB - P.tauB) < 1e-12
    hits, viols = predict_curve(KA, EA.float(), list(PILOT_GRID), prob.isA, prob.nA, prob.nB, eta, 5000, tauA, tauB)
    sim_p = P.simulate([(KAp, 0)], (P.FA - P.Y).reshape(-1), list(PILOT_GRID), eta_p, 5000)
    assert list(hits) == [int(x) for x in sim_p['hit']], (hits, sim_p['hit'])
    r_lin, T_hat = choose(torch.tensor(hits), torch.tensor(viols), list(PILOT_GRID))
    r_lin_p, T_hat_p = P.plan([(KAp, 0)], (P.FA - P.Y).reshape(-1), list(PILOT_GRID), eta_p, 5000)
    assert (r_lin, T_hat) == (r_lin_p, T_hat_p), ((r_lin, T_hat), (r_lin_p, T_hat_p))
    ctx = _Ctx()
    ctx.eta, ctx.tauA, ctx.tauB, ctx.LA0, ctx.LB0 = eta, tauA, tauB, LA0, LB0
    res, _ = run_lanes(prob, thA, [Fixed(f'fixed_{r}', r) for r in rs], ctx, 5000)
    T_torch = [ln['T'] for ln in res]
    T_pilot = [P.run_real(eta_p, 5000, r)['T'] for r in rs]
    assert T_torch == T_pilot, (T_torch, T_pilot)
    return dict(alpha=alpha, stepsA=stepsA_p, r_lin=r_lin, T_hat=T_hat, T=T_torch)


def test_parity_lazy():
    print(check(1.0))


def test_parity_rich():
    print(check(0.05))


if __name__ == '__main__':
    print(check(1.0))
    print(check(0.05))
