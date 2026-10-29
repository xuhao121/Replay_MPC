"""Two extra checks for the replay-controller theory (run after sim2.py):
1) quasi-static ratio check: r*||gA|| / ((1-r)*||gB||) versus rho = ||P_N gB|| / ||gB||
2) schedule invariance: every converging schedule ends at the same joint solution."""
import numpy as np
from sim2 import make, lmax

def ratio_check(scaleB=0.2, seed=0):
    XA, yA, XB, yB = make(seed, alpha=0.5, scaleB=scaleB)
    nA, nB = len(yA), len(yB); HA = XA.T @ XA / nA; HB = XB.T @ XB / nB
    PN = np.linalg.pinv(XA) @ XA; thA = np.linalg.pinv(XA) @ yA
    print(f"ratio check, new-task curvature / old = {lmax(HB)/lmax(HA):.3f}")
    for r in [0.3, 0.5, 0.8]:
        eta = 1.9 / lmax((1 - r) * HB + r * HA); w = thA.copy(); row = []
        for t in range(1001):
            gA = XA.T @ (XA @ w - yA) / nA; gB = XB.T @ (XB @ w - yB) / nB
            if t in (10, 100, 1000):
                row.append(f"t={t}: ratio {r*np.linalg.norm(gA)/((1-r)*np.linalg.norm(gB)):.2f} vs rho {np.linalg.norm(PN@gB)/np.linalg.norm(gB):.2f}")
            w -= eta * ((1 - r) * gB + r * gA)
        print(f"  r={r}: " + " | ".join(row))

def invariance_check(seed=0, T=20000):
    XA, yA, XB, yB = make(seed)
    nA, nB = len(yA), len(yB); HA = XA.T @ XA / nA; HB = XB.T @ XB / nB
    thA = np.linalg.pinv(XA) @ yA
    X = np.vstack([XA, XB]); y = np.concatenate([yA, yB])
    th_joint = thA + np.linalg.pinv(X) @ (y - X @ thA)       # closest joint solution to theta_A
    scale = np.linalg.norm(th_joint - thA)
    def go(rule, eta):
        w = thA.copy()
        for _ in range(T):
            gA = XA.T @ (XA @ w - yA) / nA; gB = XB.T @ (XB @ w - yB) / nB
            w -= eta * rule(gA, gB)
        return np.linalg.norm(w - th_joint) / scale
    rules = {
        'fixed r=0.2': (lambda a, b: 0.8 * b + 0.2 * a, 1.9 / lmax(0.8 * HB + 0.2 * HA)),
        'fixed r=0.7': (lambda a, b: 0.3 * b + 0.7 * a, 1.9 / lmax(0.3 * HB + 0.7 * HA)),
        'normbal_lr':  (lambda a, b: b + min(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-300), 1e4) * a, 1.0 / lmax(HB)),
    }
    for name, (rule, eta) in rules.items():
        print(f"  {name:12s} final distance to closest joint solution (relative): {go(rule, eta):.2e}")

if __name__ == '__main__':
    ratio_check()
    print("schedule invariance check:")
    invariance_check()
