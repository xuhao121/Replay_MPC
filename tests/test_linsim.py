"""The function-space simulator must equal explicit GD of a linear model in parameter space.
Run:  python -m pytest -q tests/test_linsim.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from rmpc.ntk import Kernel
from rmpc.linsim import simulate, plan, predict_window, r_coefs, choose


def _setup(n=40, C=3, P=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    Phi = torch.randn(n * C, P, generator=g, dtype=torch.float64) / P ** 0.5     # rows (sample, output)
    theta_star = torch.randn(P, generator=g, dtype=torch.float64)
    theta0 = torch.randn(P, generator=g, dtype=torch.float64) * 0.3
    isA = torch.zeros(n, dtype=torch.bool)
    isA[: n // 2] = True
    y = (Phi @ theta_star).view(n, C)
    return Phi, theta0, y, isA


def test_simulator_equals_parameter_gd():
    Phi, th, y, isA = _setup()
    n, C = y.shape
    nA = nB = n // 2
    K = Kernel(Phi @ Phi.T, 'full', n, C)
    E0 = (Phi @ th).view(n, C) - y
    eta = 0.5 * nA / float(torch.linalg.eigvalsh(K.mat)[-1])
    rs = [0.2, 0.5, 0.8]
    LB0 = float(0.5 * (E0[~isA] ** 2).sum() / nB)
    LA0 = float(0.5 * (E0[isA] ** 2).sum() / nA)
    tauA, tauB = LA0 + 0.0, 0.1 * LB0
    res = simulate(K, E0, r_coefs(rs), isA, nA, nB, eta, 400, 1e9, -1.0)       # no hits: run all steps
    for k, r in enumerate(rs):
        w = torch.where(isA, torch.tensor(r / nA, dtype=torch.float64), torch.tensor((1 - r) / nB, dtype=torch.float64))
        t = th.clone()
        for _ in range(400):
            E = (Phi @ t).view(n, C) - y
            t = t - eta * Phi.T @ (w[:, None] * E).reshape(-1)
        E_ref = ((Phi @ t).view(n, C) - y).float()
        assert torch.allclose(res['E'][..., k], E_ref, rtol=1e-3, atol=1e-4), (res['E'][..., k] - E_ref).abs().max()


def test_plan_and_window_consistency():
    Phi, th, y, isA = _setup(seed=1)
    n, C = y.shape
    nA = nB = n // 2
    K = Kernel(Phi @ Phi.T, 'full', n, C)
    E0 = (Phi @ th).view(n, C) - y
    eta = 0.5 * nA / float(torch.linalg.eigvalsh(K.mat)[-1])
    LB0 = float(0.5 * (E0[~isA] ** 2).sum() / nB)
    LA0 = float(0.5 * (E0[isA] ** 2).sum() / nA)
    grid = [0.1 * i for i in range(1, 10)]
    tauA, tauB = LA0 + 0.5 * LB0, 0.2 * LB0
    full = simulate(K, E0, r_coefs(grid), isA, nA, nB, eta, 5000, tauA, tauB)
    r_full, T_full = choose(full['hit'], full['viol'], grid)
    p = plan(K, E0, grid, isA, nA, nB, eta, 5000, tauA, tauB)
    assert (p['r'], p['T_hat']) == (r_full, T_full)
    Ew = predict_window(K, E0, (0.3, 0.7), isA, nA, nB, eta, 37)
    ref = simulate(K, E0, [[0.3, 0.7]], isA, nA, nB, eta, 37, -1.0, -1.0)['E'][..., 0]
    assert torch.allclose(Ew, ref)




def test_two_phase_columns():
    from rmpc.linsim import plan2
    Phi, th, y, isA = _setup(seed=2)
    n, C = y.shape
    nA = nB = n // 2
    K = Kernel(Phi @ Phi.T, 'full', n, C)
    E0 = (Phi @ th).view(n, C) - y
    eta = 0.5 * nA / float(torch.linalg.eigvalsh(K.mat)[-1])
    # switching column == two consecutive constant simulations
    mid = simulate(K, E0, [[0.2, 0.8]], isA, nA, nB, eta, 25, -1.0, -1.0)['E'][..., 0]
    seq = simulate(K, mid, [[0.7, 0.3]], isA, nA, nB, eta, 40, -1.0, -1.0)['E'][..., 0]
    sw = simulate(K, E0, [[0.2, 0.8]], isA, nA, nB, eta, 65, -1.0, -1.0, coefs2=[[0.7, 0.3]], ts=[25])['E'][..., 0]
    assert torch.allclose(seq, sw, rtol=1e-4, atol=1e-5)
    # plan2 = brute force over its own candidate set
    LB0 = float(0.5 * (E0[~isA] ** 2).sum() / nB)
    LA0 = float(0.5 * (E0[isA] ** 2).sum() / nA)
    tauA, tauB = LA0 + 0.5 * LB0, 0.2 * LB0
    grid, M = [0.2, 0.4, 0.6, 0.8], 10
    p = plan2(K, E0, grid, isA, nA, nB, eta, 3000, tauA, tauB, M, js=(1, 3))
    best = None
    for a in grid:
        for b in grid:
            for ts in ([0] if a == b else [M, 3 * M]):
                h = int(simulate(K, E0, [[a, 1 - a]], isA, nA, nB, eta, 3000, tauA, tauB,
                                 coefs2=[[b, 1 - b]], ts=[ts])['hit'][0])
                if h >= 0 and (best is None or h < best[0]):
                    best = (h, a)
    assert p['T_hat'] == best[0], (p, best)


if __name__ == '__main__':
    test_simulator_equals_parameter_gd(); test_plan_and_window_consistency(); test_two_phase_columns()
    print('linsim tests passed')
