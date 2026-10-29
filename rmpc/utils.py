import json, os, time, math, random, socket, contextlib
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def pick_device(device=None):
    if device:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def configure_torch(tf32=False, threads=None):
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    if threads:
        torch.set_num_threads(int(threads))


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class Timer:
    """Accumulates wall time per key: with timer('kernel'): ..."""

    def __init__(self, device=None):
        self.t = {}
        self.n = {}
        self.device = device

    @contextlib.contextmanager
    def __call__(self, key):
        if self.device is not None:
            sync(self.device)
        t0 = time.time()
        try:
            yield
        finally:
            if self.device is not None:
                sync(self.device)
            self.t[key] = self.t.get(key, 0.0) + time.time() - t0
            self.n[key] = self.n.get(key, 0) + 1

    def as_dict(self):
        return {k: dict(sec=round(v, 3), n=self.n[k]) for k, v in self.t.items()}


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, torch.Tensor):
        return to_jsonable(x.detach().cpu().tolist())
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def write_json_atomic(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, 'w') as f:
        json.dump(to_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def savez_atomic(path, **arrays):
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def host_tag():
    return f"{socket.gethostname()}:{os.getpid()}"


def gpu_name(device):
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    return 'cpu'
