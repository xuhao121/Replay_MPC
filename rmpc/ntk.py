"""Empirical NTK of the raw network g at given parameters.

kind = 'analytic'  exact NTK of MLP1 in closed form (fast; used for all E1 runs)
       'entk'      exact empirical NTK of any model: per-sample Jacobians with torch.func (vmap(jacrev)),
                   K = J J^T with rows ordered (sample, output)
       'pntk'      pseudo-NTK (Mohamadi, Bae & Sutherland 2023): Theta = (1/C) J_s J_s^T with J_s the gradient
                   of the summed logits; used as Theta (x) I_C. C times cheaper in Jacobian rows, for large models.

The Kernel object acts on residual arrays of shape (n, C, R) (R = number of replay ratios simulated at once).
"""
import math
import torch
from torch.func import functional_call, vmap, jacrev, grad


class Kernel:
    def __init__(self, mat, kind, n, C):
        self.mat = mat
        self.kind = 'pntk' if kind == 'pntk' else 'full'
        self.n, self.C = n, C

    def mv(self, V):
        n, C, R = V.shape
        if self.kind == 'pntk':
            return (self.mat @ V.reshape(n, C * R)).reshape(n, C, R)
        return (self.mat @ V.reshape(n * C, R)).reshape(n, C, R)

    def fro(self):
        return float(torch.linalg.norm(self.mat))

    def nbytes(self):
        return self.mat.numel() * self.mat.element_size()

    def lam_max(self, iters=300, seed=0):
        M = self.mat
        N = M.shape[0]
        if N <= 4000:
            return float(torch.linalg.eigvalsh(M.double())[-1])
        g = torch.Generator(device='cpu').manual_seed(seed)
        v = torch.randn(N, 1, generator=g, dtype=torch.float64).to(M.device, M.dtype)
        v /= torch.linalg.norm(v)
        lam = 0.0
        for _ in range(iters):
            w = M @ v
            lam = float((v * w).sum())
            v = w / torch.linalg.norm(w)
        return float((v * (M @ v)).sum())


def rel_change(K, Kref):
    return float(torch.linalg.norm(K.mat - Kref.mat) / torch.linalg.norm(Kref.mat))


def _single(model):
    def f(p, x):
        return functional_call(model, p, (x.unsqueeze(0),)).squeeze(0)
    return f


def jacobian_rows(model, params, X, chunk=8, store_dtype=torch.float32):
    """(n*C, P) Jacobian of the raw network outputs; rows ordered (sample, output)."""
    names = list(params.keys())
    jf = vmap(jacrev(_single(model)), in_dims=(None, 0))
    rows = []
    for i in range(0, X.shape[0], chunk):
        jac = jf(params, X[i:i + chunk])
        b = X[i:i + chunk].shape[0]
        Jb = torch.cat([jac[k].reshape(b, jac[k].shape[1], -1) for k in names], dim=2)   # (b, C, P)
        rows.append(Jb.reshape(-1, Jb.shape[-1]).to(store_dtype))
        del jac, Jb
    return torch.cat(rows)


def sumlogit_rows(model, params, X, chunk=32, store_dtype=torch.float32):
    """(n, P) gradients of the summed logits (pseudo-NTK)."""
    names = list(params.keys())
    f = _single(model)
    gf = vmap(grad(lambda p, x: f(p, x).sum()), in_dims=(None, 0))
    rows = []
    for i in range(0, X.shape[0], chunk):
        gr = gf(params, X[i:i + chunk])
        b = X[i:i + chunk].shape[0]
        rows.append(torch.cat([gr[k].reshape(b, -1) for k in names], dim=1).to(store_dtype))
    return torch.cat(rows)


def _gram(J, out_dtype, block=4096):
    if J.dtype == out_dtype:
        return J @ J.T
    N = J.shape[0]
    K = torch.empty(N, N, dtype=out_dtype, device=J.device)
    for i in range(0, N, block):
        Ji = J[i:i + block].to(out_dtype)
        for j in range(0, N, block):
            K[i:i + block, j:j + block] = Ji @ J[j:j + block].to(out_dtype).T
    return K


@torch.no_grad()
def _analytic(model, params, X):
    return model.analytic_ntk(params, X)


def kernel_matrix(model, params, X, kind, C, jac_chunk=8, jac_dtype='float32'):
    """Kernel of the raw network at `params` on inputs X (float dtype of params)."""
    n = X.shape[0]
    out_dtype = next(iter(params.values())).dtype
    sd = getattr(torch, jac_dtype)
    params = {k: v.detach() for k, v in params.items()}
    if kind == 'analytic':
        return Kernel(_analytic(model, params, X), 'full', n, C)
    if kind == 'entk':
        J = jacobian_rows(model, params, X, jac_chunk, sd)
        K = _gram(J, out_dtype)
        del J
        return Kernel(K, 'full', n, C)
    if kind == 'pntk':
        J = sumlogit_rows(model, params, X, max(jac_chunk * 4, 8), sd)
        K = _gram(J, out_dtype) / C
        del J
        return Kernel(K, 'pntk', n, C)
    raise ValueError(kind)
