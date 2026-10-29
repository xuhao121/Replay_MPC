"""Kernel correctness: analytic MLP NTK = generic eNTK; pNTK identity; eNTK action = J (J^T v); weighted lambda_max.
Run:  python -m pytest -q tests/test_ntk.py      (seconds)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from torch.func import functional_call, vjp, jvp
from rmpc.models import MLP1, ResNet18k
from rmpc.ntk import kernel_matrix, Kernel
from rmpc.linsim import lam_max_weighted

torch.set_default_dtype(torch.float64)


def _params(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def test_mlp_analytic_equals_autodiff():
    m = MLP1(10, 7, 3).double().init_(0)
    X = torch.randn(6, 10)
    p = _params(m)
    Ka = kernel_matrix(m, p, X, 'analytic', 3).mat
    Ke = kernel_matrix(m, p, X, 'entk', 3, jac_chunk=4, jac_dtype='float64').mat
    assert torch.allclose(Ka, Ke, atol=1e-10), (Ka - Ke).abs().max()


def test_pntk_identity_resnet():
    m = ResNet18k(4, 5, in_ch=3).double().init_(0)
    X = torch.randn(4, 3, 8, 8)
    p = _params(m)
    Ke = kernel_matrix(m, p, X, 'entk', 5, jac_chunk=2, jac_dtype='float64').mat.view(4, 5, 4, 5)
    Kp = kernel_matrix(m, p, X, 'pntk', 5, jac_chunk=2, jac_dtype='float64').mat
    assert torch.allclose(Kp, Ke.sum((1, 3)) / 5, atol=1e-9)


def test_entk_action_resnet():
    m = ResNet18k(4, 5, in_ch=3).double().init_(1)
    X = torch.randn(3, 3, 8, 8)
    p = _params(m)
    K = kernel_matrix(m, p, X, 'entk', 5, jac_chunk=3, jac_dtype='float64')
    v = torch.randn(3, 5)
    f = lambda q: functional_call(m, q, (X,))
    _, pull = vjp(f, p)
    (g,) = pull(v)                                  # J^T v
    _, Jg = jvp(f, (p,), (g,))                      # J J^T v
    Kv = K.mv(v[:, :, None])[..., 0]
    assert torch.allclose(Kv, Jg, atol=1e-8), (Kv - Jg).abs().max()


def test_weighted_lambda_max_power_iteration():
    torch.manual_seed(0)
    n, C = 450, 10                                   # 4500 rows -> power-iteration path
    A = torch.randn(n * C, 60)
    K = Kernel(A @ A.T, 'full', n, C)
    w = torch.rand(n) + 0.1
    lam_pi = lam_max_weighted(K, w, iters=400)
    s = w.sqrt().repeat_interleave(C)
    lam_ex = float(torch.linalg.eigvalsh(s[:, None] * K.mat * s[None, :])[-1])
    assert abs(lam_pi - lam_ex) / lam_ex < 1e-3, (lam_pi, lam_ex)


if __name__ == '__main__':
    test_mlp_analytic_equals_autodiff(); test_pntk_identity_resnet(); test_entk_action_resnet()
    test_weighted_lambda_max_power_iteration()
    print('ntk tests passed')
