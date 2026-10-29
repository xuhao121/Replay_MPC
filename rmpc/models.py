"""Models. Both are used through torch.func.functional_call so that several lanes (copies with different
replay schedules) can be stacked and trained in one vmapped call.

MLP1       one hidden layer, tanh, NTK parameterisation, no biases (as in the CPU pilot):
           g(x) = W2 tanh(W1 x / sqrt(d)) / sqrt(h).  Biases are left out on purpose: they add a large
           near-constant NTK eigenvalue that dictates the step size and slows everything else down.
ResNet18k  CIFAR-style pre-activation ResNet-18 with base width k (k, 2k, 4k, 8k), GroupNorm instead of
           BatchNorm (no running statistics, so the function of one sample does not depend on the batch;
           vmap-friendly), a CENTRED activation after every GroupNorm, no bias in the classifier.
           Why centred: the positive mean of ReLU features makes the gradients of all images point the same
           way, which puts C huge eigenvalues (one constant mode per output) on top of the NTK
           (Karakida et al. 2019, "pathological sharpness"). They cap the step size while carrying almost no
           information. On 60 random 32x32 inputs (k = 8) the ratio lambda_max / median eigenvalue drops from
           165 (post-activation ResNet, plain ReLU) to 4.1 (pre-activation, relu(z) - 1/sqrt(2 pi)).
           Why GELU (act='gelu', the default): laziness needs a smooth network. With ReLU the Jacobian changes by
           activation flips, so the kernel drift falls only like alpha^(-1/2) although the parameter displacement
           falls like 1/alpha (measured on a CIFAR-like 32x32 stand-in, k = 8: drift 0.70 / 0.26 / 0.09 at
           alpha = 10 / 100 / 1000). With centred GELU it falls like 1/alpha (0.25 / 0.025 / 0.002), so the alpha
           grid reaches the low drift targets. act='relu' keeps the centred ReLU.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP1(nn.Module):
    def __init__(self, d, h, C):
        super().__init__()
        self.d, self.h, self.C = d, h, C
        self.W1 = nn.Parameter(torch.empty(h, d))
        self.W2 = nn.Parameter(torch.empty(C, h))

    def init_(self, seed, mode='torch'):
        with torch.no_grad():
            if mode == 'numpy_pilot':          # bit-for-bit the pilot's Net.__init__
                rng = np.random.default_rng(10_000 + seed)
                self.W1.copy_(torch.as_tensor(rng.standard_normal((self.h, self.d))))
                self.W2.copy_(torch.as_tensor(rng.standard_normal((self.C, self.h))))
            else:
                g = torch.Generator().manual_seed(10_000 + seed)
                self.W1.copy_(torch.randn(self.h, self.d, generator=g, dtype=torch.float64))
                self.W2.copy_(torch.randn(self.C, self.h, generator=g, dtype=torch.float64))
        return self

    def forward(self, x):
        P = torch.tanh(x @ self.W1.T / math.sqrt(self.d))
        return P @ self.W2.T / math.sqrt(self.h)

    def analytic_ntk(self, params, X):
        """Exact empirical NTK of the raw network at `params` on X; rows ordered (sample, output)."""
        W1, W2 = params['W1'], params['W2']
        n, C, h, d = X.shape[0], self.C, self.h, self.d
        P = torch.tanh(X @ W1.T / math.sqrt(d))                       # (n, h)
        G = ((1.0 - P * P)[:, None, :] * W2[None, :, :]).reshape(n * C, h) / math.sqrt(h)
        K = G @ G.T                                                    # (nC, nC)
        K4 = K.view(n, C, n, C)
        K4.mul_((X @ X.T / d)[:, None, :, None])
        PP = P @ P.T / h
        for i in range(C):
            K4[:, i, :, i] += PP
        return K


def _gn(ch):
    return nn.GroupNorm(min(32, max(1, ch // 4)), ch)


_C0 = 1.0 / math.sqrt(2.0 * math.pi)       # E[relu(z)] for z ~ N(0, 1)


def crelu(z):
    """Centred ReLU: zero mean when its input is standard normal (every use follows a GroupNorm)."""
    return F.relu(z) - _C0


_C_GELU = 1.0 / (2.0 * math.sqrt(math.pi))     # E[gelu(z)] = E[z Phi(z)] for z ~ N(0, 1)


def cgelu(z):
    """Centred GELU (exact erf form): smooth, zero mean for standard normal input."""
    return F.gelu(z) - _C_GELU


ACTS = {'gelu': cgelu, 'relu': crelu}


class _PreBlock(nn.Module):
    """Pre-activation basic block: x + conv(act(gn(conv(act(gn(x)))))), act = centred GELU or ReLU."""

    def __init__(self, cin, cout, stride, act=cgelu):
        super().__init__()
        self.act = act
        self.n1 = _gn(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n2 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        o = self.act(self.n1(x))
        shortcut = x if self.sc is None else self.sc(o)
        o = self.c1(o)
        o = self.c2(self.act(self.n2(o)))
        return o + shortcut


class ResNet18k(nn.Module):
    def __init__(self, k=64, C=10, in_ch=3, act='gelu'):
        super().__init__()
        self.C = C
        self.act = ACTS[act]
        self.stem = nn.Conv2d(in_ch, k, 3, 1, 1, bias=False)
        layers, cin = [], k
        for w, s in zip([k, 2 * k, 4 * k, 8 * k], [1, 2, 2, 2]):
            layers += [_PreBlock(cin, w, s, self.act), _PreBlock(w, w, 1, self.act)]
            cin = w
        self.layers = nn.Sequential(*layers)
        self.nf = _gn(8 * k)
        self.fc = nn.Linear(8 * k, C, bias=False)

    def init_(self, seed, mode='torch'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10_000 + seed)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    m.reset_parameters()
                elif isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        return self

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = self.act(self.nf(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

    analytic_ntk = None


def build_model(cfg, input_shape, C):
    if cfg.model == 'mlp':
        d = int(np.prod(input_shape))
        m = MLP1(d, cfg.width, C)
    elif cfg.model == 'resnet18k':
        m = ResNet18k(cfg.width, C, in_ch=input_shape[0], act=cfg.act)
    else:
        raise ValueError(cfg.model)
    m = m.to(getattr(torch, cfg.dtype))       # cast before init so a float64 pilot init stays exact
    m.init_(cfg.seed, cfg.init)
    return m


def n_params(model):
    return sum(p.numel() for p in model.parameters())
