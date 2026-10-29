"""Two-task data: task A = original inputs, task B = disjoint inputs under a fixed transform, same labels.

All sources return a TaskPair with
    X     (n, ...) float64 (cast to the training dtype by the job), task A rows first, then task B
    Y     (n, C)   centred one-hot targets e_y - 1/C (MSE loss)
    isA   (n,)     bool
    Xte_A, Yte_A, Xte_B, Yte_B   held-out test sets (task B test inputs get the same transform)
Inputs are standardised with statistics of the training pool (A and B together); test sets reuse them.

Sources
    digits     sklearn digits (8x8), identical to the CPU pilot (make_data in e0/pilot/core.py)
    mnist      MNIST via data/mnist.npz (scripts/prepare_data.py) or torchvision
    cifar10    CIFAR-10 via data/cifar10.npz or torchvision
    synth      Gaussian class clusters on 8x8 "images", flattened (tests, no download)
    synth_img  Gaussian class clusters as 3x8x8 images (ResNet tests, no download)
"""
from dataclasses import dataclass
import os
import numpy as np
import torch

DATA_ROOT = os.environ.get('RMPC_DATA', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))


@dataclass
class TaskPair:
    X: torch.Tensor
    Y: torch.Tensor
    isA: torch.Tensor
    Xte_A: torch.Tensor
    Yte_A: torch.Tensor
    Xte_B: torch.Tensor
    Yte_B: torch.Tensor
    C: int
    input_shape: tuple

    @property
    def nA(self):
        return int(self.isA.sum())

    @property
    def nB(self):
        return int((~self.isA).sum())


# ----------------------------------------------------------------------------- raw loaders
def _load_npz_or_torchvision(name):
    path = os.path.join(DATA_ROOT, f'{name}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['x_train'], z['y_train'].astype(np.int64), z['x_test'], z['y_test'].astype(np.int64)
    try:
        import torchvision
    except ImportError as e:
        raise FileNotFoundError(f'{path} not found and torchvision missing; run scripts/prepare_data.py') from e
    os.makedirs(DATA_ROOT, exist_ok=True)
    if name == 'mnist':
        tr = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True)
        te = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True)
        out = (tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy())
    elif name == 'cifar10':
        tr = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True)
        te = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True)
        out = (tr.data, np.array(tr.targets), te.data, np.array(te.targets))
    else:
        raise ValueError(name)
    np.savez_compressed(path, x_train=out[0], y_train=out[1], x_test=out[2], y_test=out[3])
    return out


def _transform(x, kind, spatial_axes):
    """x: float array in [0, 1] (or standardised); spatial_axes = (h_axis, w_axis)."""
    if kind == 'rot90':
        return np.ascontiguousarray(np.rot90(x, k=1, axes=spatial_axes))
    if kind == 'neg':
        return 1.0 - x
    if kind == 'perm':
        h_ax, w_ax = spatial_axes
        H, W = x.shape[h_ax], x.shape[w_ax]
        perm = np.random.default_rng(777).permutation(H * W)
        xs = np.moveaxis(x, (h_ax, w_ax), (-2, -1))
        shp = xs.shape
        xs = xs.reshape(*shp[:-2], H * W)[..., perm].reshape(shp)
        return np.ascontiguousarray(np.moveaxis(xs, (-2, -1), (h_ax, w_ax)))
    if kind == 'none':
        return x
    raise ValueError(f'unknown transform {kind}')


def _targets(y, C):
    return np.eye(C)[y] - 1.0 / C


def _finish(XA, XB, yA, yB, XteA, XteB, yteA, yteB, C, flatten, channel_axis=None):
    """Standardise with train-pool statistics; flatten for MLPs (pilot convention) or per-channel for CNNs."""
    Xall = np.concatenate([XA, XB]).astype(np.float64)
    Xte = [XteA.astype(np.float64), XteB.astype(np.float64)]
    if flatten:
        Xall = Xall.reshape(len(Xall), -1)
        Xte = [t.reshape(len(t), -1) for t in Xte]
        mu = Xall.mean(0)
        Xall = Xall - mu
        sd = Xall.std()
        Xall = Xall / sd
        Xte = [(t - mu) / sd for t in Xte]
    else:
        axes = tuple(i for i in range(Xall.ndim) if i != channel_axis)
        mu = Xall.mean(axis=axes, keepdims=True)
        sd = Xall.std(axis=axes, keepdims=True)
        Xall = (Xall - mu) / sd
        Xte = [(t - mu) / sd for t in Xte]
    nA = len(XA)
    isA = np.zeros(len(Xall), bool)
    isA[:nA] = True
    Y = np.concatenate([_targets(yA, C), _targets(yB, C)])
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)   # cast later to cfg.dtype
    return TaskPair(X=t(Xall), Y=torch.as_tensor(Y), isA=torch.as_tensor(isA),
                    Xte_A=t(Xte[0]), Yte_A=torch.as_tensor(_targets(yteA, C)),
                    Xte_B=t(Xte[1]), Yte_B=torch.as_tensor(_targets(yteB, C)),
                    C=C, input_shape=tuple(Xall.shape[1:]))


# ----------------------------------------------------------------------------- sources
def digits_pair(seed, nA, nB, transform, n_test):
    """Bit-identical training data to the CPU pilot (e0/pilot/core.py make_data); test sets are drawn from
    the remaining images (the pilot had none)."""
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    iA, iB = idx[:nA], idx[nA:nA + nB]
    rest = idx[nA + nB:]
    nt = min(n_test, len(rest) // 2)
    tA, tB = rest[:nt], rest[nt:2 * nt]
    imgs = X.reshape(-1, 8, 8) / 16.0
    XA, XB = imgs[iA], _transform(imgs[iB], transform, (1, 2))
    XteA, XteB = imgs[tA], _transform(imgs[tB], transform, (1, 2))
    return _finish(XA, XB, y[iA], y[iB], XteA, XteB, y[tA], y[tB], 10, flatten=True)


def mnist_pair(seed, nA, nB, transform, n_test):
    xtr, ytr, xte, yte = _load_npz_or_torchvision('mnist')
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(ytr))
    iA, iB = p[:nA], p[nA:nA + nB]
    q = rng.permutation(len(yte))
    tA, tB = q[:n_test], q[n_test:2 * n_test]
    f = lambda a: a.astype(np.float64) / 255.0
    XA, XB = f(xtr[iA]), _transform(f(xtr[iB]), transform, (1, 2))
    XteA, XteB = f(xte[tA]), _transform(f(xte[tB]), transform, (1, 2))
    return _finish(XA, XB, ytr[iA], ytr[iB], XteA, XteB, yte[tA], yte[tB], 10, flatten=True)


def cifar10_pair(seed, nA, nB, transform, n_test):
    xtr, ytr, xte, yte = _load_npz_or_torchvision('cifar10')      # (N, 32, 32, 3) uint8
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(ytr))
    iA, iB = p[:nA], p[nA:nA + nB]
    q = rng.permutation(len(yte))
    tA, tB = q[:n_test], q[n_test:2 * n_test]
    f = lambda a: np.transpose(a.astype(np.float64) / 255.0, (0, 3, 1, 2))   # -> (N, 3, 32, 32)
    XA, XB = f(xtr[iA]), _transform(f(xtr[iB]), transform, (2, 3))
    XteA, XteB = f(xte[tA]), _transform(f(xte[tB]), transform, (2, 3))
    return _finish(XA, XB, ytr[iA], ytr[iB], XteA, XteB, yte[tA], yte[tB], 10, flatten=False, channel_axis=1)


def _synth_images(seed, n, shape, C, rng_data, noise):
    proto = np.random.default_rng(1234).standard_normal((C,) + shape)
    y = rng_data.integers(0, C, n)
    x = proto[y] + noise * rng_data.standard_normal((n,) + shape)
    return x, y


def synth_pair(seed, nA, nB, transform, n_test, image=False):
    """Class prototypes + Gaussian noise (tests only). Images use less noise so that tiny CNNs fit them fast."""
    rng = np.random.default_rng(seed)
    shape = (3, 8, 8) if image else (8, 8)
    sp = (2, 3) if image else (1, 2)
    noise = 0.3 if image else 0.8
    xA, yA = _synth_images(seed, nA, shape, 10, rng, noise)
    xB, yB = _synth_images(seed, nB, shape, 10, rng, noise)
    xtA, ytA = _synth_images(seed, n_test, shape, 10, rng, noise)
    xtB, ytB = _synth_images(seed, n_test, shape, 10, rng, noise)
    XB, XtB = _transform(xB, transform, sp), _transform(xtB, transform, sp)
    return _finish(xA, XB, yA, yB, xtA, XtB, ytA, ytB, 10, flatten=not image, channel_axis=1 if image else None)


def make_task_pair(cfg):
    n_test = max(int(cfg.n_test), 1)
    if cfg.dataset == 'digits':
        return digits_pair(cfg.seed, cfg.nA, cfg.nB, cfg.transform, n_test)
    if cfg.dataset == 'mnist':
        return mnist_pair(cfg.seed, cfg.nA, cfg.nB, cfg.transform, n_test)
    if cfg.dataset == 'cifar10':
        return cifar10_pair(cfg.seed, cfg.nA, cfg.nB, cfg.transform, n_test)
    if cfg.dataset == 'synth':
        return synth_pair(cfg.seed, cfg.nA, cfg.nB, cfg.transform, n_test, image=False)
    if cfg.dataset == 'synth_img':
        return synth_pair(cfg.seed, cfg.nA, cfg.nB, cfg.transform, n_test, image=True)
    raise ValueError(cfg.dataset)
