"""Download MNIST and CIFAR-10 once into data/<name>.npz (uint8 images + int labels).

    python scripts/prepare_data.py                # both
    python scripts/prepare_data.py --only mnist
Uses torchvision if installed, otherwise downloads the original archives directly.
Set RMPC_DATA=/persistent/path to keep the data outside the repository.
"""
import argparse, gzip, io, os, pickle, sys, tarfile, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmpc.data import DATA_ROOT

MNIST = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
CIFAR = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def _get(url):
    print('downloading', url, flush=True)
    with urllib.request.urlopen(url) as r:
        return r.read()


def mnist_direct():
    def idx(buf, images):
        b = gzip.decompress(buf)
        return np.frombuffer(b, np.uint8, offset=16).reshape(-1, 28, 28) if images else np.frombuffer(b, np.uint8, offset=8)
    xtr = idx(_get(MNIST + 'train-images-idx3-ubyte.gz'), True)
    ytr = idx(_get(MNIST + 'train-labels-idx1-ubyte.gz'), False)
    xte = idx(_get(MNIST + 't10k-images-idx3-ubyte.gz'), True)
    yte = idx(_get(MNIST + 't10k-labels-idx1-ubyte.gz'), False)
    return xtr, ytr, xte, yte


def cifar_direct():
    tf = tarfile.open(fileobj=io.BytesIO(_get(CIFAR)), mode='r:gz')
    def batch(name):
        d = pickle.load(tf.extractfile(f'cifar-10-batches-py/{name}'), encoding='bytes')
        return d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b'labels'])
    tr = [batch(f'data_batch_{i}') for i in range(1, 6)]
    xte, yte = batch('test_batch')
    return np.concatenate([x for x, _ in tr]), np.concatenate([y for _, y in tr]), xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['mnist', 'cifar10'], default=None)
    a = ap.parse_args()
    os.makedirs(DATA_ROOT, exist_ok=True)
    for name in (['mnist', 'cifar10'] if a.only is None else [a.only]):
        path = os.path.join(DATA_ROOT, f'{name}.npz')
        if os.path.exists(path):
            print(path, 'exists')
            continue
        try:
            from rmpc.data import _load_npz_or_torchvision
            _load_npz_or_torchvision(name)          # torchvision path writes the npz itself
        except Exception as e:
            print('torchvision path failed:', e, '-> direct download')
            xtr, ytr, xte, yte = mnist_direct() if name == 'mnist' else cifar_direct()
            np.savez_compressed(path, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
        z = np.load(path)
        print(name, {k: z[k].shape for k in z.files})


if __name__ == '__main__':
    main()
