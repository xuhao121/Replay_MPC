"""TwoTaskProblem: data + model + output centring/scaling, shared by phase A, phase B and the kernels.

Output model (Chizat, Oyallon & Bach 2019):  f(x; theta) = alpha * (g(x; theta) - g(x; theta0)),
loss scaled by 1/alpha^2. One GD step with step size eta on the scaled loss moves the training-set
function by  -eta K W E  to first order, independent of alpha; alpha only controls how far the real kernel
drifts during training (large alpha = lazy / NTK regime, small alpha = rich / feature learning).
In code: the parameter step is  theta <- theta - (eta / alpha^2) * grad_theta [ sum_s w_s 0.5 ||F_s - y_s||^2 ].
"""
import math
import numpy as np
import torch
from torch.func import functional_call, vmap
from .data import make_task_pair
from .models import build_model, n_params
from .ntk import kernel_matrix


class TwoTaskProblem:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.dtype = getattr(torch, cfg.dtype)
        tp = make_task_pair(cfg)
        dt = self.dtype
        self.X = tp.X.to(device, dt)
        self.Y = tp.Y.to(device, dt)
        self.isA = tp.isA.to(device)
        self.idxA = torch.nonzero(self.isA).flatten()
        self.idxB = torch.nonzero(~self.isA).flatten()
        self.nA, self.nB, self.C = tp.nA, tp.nB, tp.C
        self.n = self.nA + self.nB
        self.Xte = {'A': tp.Xte_A.to(device, dt), 'B': tp.Xte_B.to(device, dt)}
        self.Yte = {'A': tp.Yte_A.to(device, dt), 'B': tp.Yte_B.to(device, dt)}
        self.model = build_model(cfg, tp.input_shape, tp.C).to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.theta0 = {k: v.detach().clone() for k, v in self.model.named_parameters()}
        self.P = n_params(self.model)
        self.alpha = float(cfg.alpha)
        self.kind = cfg.resolved_kernel()
        self.chunk = cfg.sample_chunk if cfg.sample_chunk > 0 else None
        self.g0 = self.raw(self.theta0, self.X)
        self.g0te = {k: self.raw(self.theta0, v) for k, v in self.Xte.items()}

    # ------------------------------------------------------------------ single-model evaluation
    @torch.no_grad()
    def raw(self, params, X):
        if self.chunk is None or X.shape[0] <= self.chunk:
            return functional_call(self.model, params, (X,))
        return torch.cat([functional_call(self.model, params, (X[i:i + self.chunk],))
                          for i in range(0, X.shape[0], self.chunk)])

    def F(self, params):
        return self.alpha * (self.raw(params, self.X) - self.g0)

    def losses(self, E):
        """E: (..., n, C) residual -> (LA, LB) each (...)"""
        E2 = (E * E).sum(-1)
        LA = 0.5 * E2[..., self.isA].sum(-1) / self.nA
        LB = 0.5 * E2[..., ~self.isA].sum(-1) / self.nB
        return LA, LB

    @torch.no_grad()
    def test_metrics(self, params, keep_outputs=False):
        out = {}
        for k in ('A', 'B'):
            Fte = self.alpha * (self.raw(params, self.Xte[k]) - self.g0te[k])
            Y = self.Yte[k]
            out[f'mse_{k}'] = float(0.5 * ((Fte - Y) ** 2).sum(-1).mean())
            out[f'acc_{k}'] = float((Fte.argmax(-1) == Y.argmax(-1)).float().mean())
            if keep_outputs:
                out[f'F_{k}'] = Fte.float().cpu().numpy()
        return out

    # ------------------------------------------------------------------ stacked lanes
    def fwd_stacked(self, stacked, X, per_lane=False):
        """stacked: dict name -> (R, *shape). X: (n, ...) shared or (R, b, ...) per lane. -> (R, n|b, C)"""
        f = lambda p, x: functional_call(self.model, p, (x,))
        return vmap(f, in_dims=(0, 0 if per_lane else None))(stacked, X)

    # ------------------------------------------------------------------ kernels
    def kernel(self, params, idx=None):
        X = self.X if idx is None else self.X[idx]
        return kernel_matrix(self.model, params, X, self.kind, self.C, self.cfg.jac_chunk, self.cfg.jac_dtype)

    def block_AA(self, K):
        m = self.nA * (self.C if K.kind == 'full' else 1)
        return K.mat[:m, :m]
