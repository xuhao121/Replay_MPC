"""Replay-weight controllers. A controller sets the mixing coefficients (cA, cB) of one lane:
per-sample loss weights cA / nA on task A (replay) and cB / nB on task B; a fixed replay ratio r is (r, 1 - r).

Hooks called by the trainer (all optional):
    start(ctx)                       -> initial (cA, cB)
    on_losses(t, LA, LB, ctx)        -> new (cA, cB) or None      every step, cheap
    wants_grads(t) / on_grads(...)   -> per-task gradient norms   (lanes run with split gradients)
    check_due(t) / on_check(t, view, ctx) -> new (cA, cB) or None  every M steps (MPC), may compute kernels
The decision taken at step t is applied to the update made at step t (same convention as the CPU pilot).
"""
import math
import torch
from .linsim import plan, plan2, predict_window
from .ntk import rel_change


class Controller:
    split_grads = False

    def __init__(self, name):
        self.name = name
        self.trace = []                  # (t, r_equivalent) whenever the mixing changes

    def start(self, ctx):
        raise NotImplementedError

    def on_losses(self, t, LA, LB, ctx):
        return None

    def wants_grads(self, t):
        return False

    def on_grads(self, t, gA, gB, LA, LB, ctx):
        return None

    def check_due(self, t):
        return False

    def on_check(self, t, view, ctx):
        return None

    def _log(self, t, cA, cB, every=1):
        r = cA / (cA + cB) if (cA + cB) > 0 else float('nan')
        if not self.trace or (abs(self.trace[-1][1] - r) > 1e-12 and (t - self.trace[-1][0]) >= every):
            self.trace.append((int(t), round(float(r), 5)))

    def summary(self):
        return dict(trace=self.trace)


class Fixed(Controller):
    """Fixed replay ratio r; mult > 1 scales both weights (= a larger step, used by oracle_edge)."""

    def __init__(self, name, r, mult=1.0):
        super().__init__(name)
        self.r, self.mult = float(r), float(mult)

    def start(self, ctx):
        self._log(0, self.r, 1 - self.r)
        return (self.mult * self.r, self.mult * (1.0 - self.r))


class MPC(Controller):
    """Online model-predictive replay control.
    replan_every: re-plan r from the current state at every check (otherwise only when the trigger fires)
    trigger: 'drift'   kernel drift on the fixed subset since the last linearisation > eps
             'prederr' || E_real - E_pred || / || E_real - E_prev || over the last window > eps (no Jacobian)
             'always'  re-linearise at every check;   'none' never re-linearise (stale theta_A kernel)
    """

    def __init__(self, name, replan_every, trigger, eps, M, two_phase=False):
        super().__init__(name)
        self.replan_every, self.trigger, self.eps, self.M = replan_every, trigger, float(eps), int(M)
        self.two_phase = bool(two_phase)
        self.plans2 = []
        self.n_relin = 0
        self.n_plans = 0
        self.drifts, self.deltas = [], []
        self.relin_t = []

    def _plan(self, E, remaining, ctx, t):
        with ctx.timer('plan'):
            if self.two_phase:
                p = plan2(self.K, E, ctx.r_grid, ctx.isA, ctx.nA, ctx.nB, ctx.eta, remaining, ctx.tauA, ctx.tauB,
                          self.M)
                self.plans2.append((int(t), p['r'], p['r2'], p['t_switch']))
                return p['r']
            return plan(self.K, E, ctx.r_grid, ctx.isA, ctx.nA, ctx.nB, ctx.eta, remaining, ctx.tauA, ctx.tauB)['r']

    def start(self, ctx):
        self.K = ctx.K_A
        self.Ks_ref = ctx.Ks_A
        self.r = ctx.r_lin if not self.two_phase else self._plan(ctx.E_A, ctx.H, ctx, 0)
        if self.trigger == 'prederr':
            self.E_prev = ctx.E_A
            self.E_pred = predict_window(self.K, ctx.E_A, (self.r, 1 - self.r), ctx.isA, ctx.nA, ctx.nB, ctx.eta, self.M)
        self._log(0, self.r, 1 - self.r)
        return (self.r, 1.0 - self.r)

    def check_due(self, t):
        return t > 0 and t % self.M == 0

    def on_check(self, t, view, ctx):
        fire = False
        Ks = None
        if self.trigger == 'drift':
            Ks = ctx.kernel_sub(view.params)
            d = rel_change(Ks, self.Ks_ref)
            self.drifts.append(round(d, 4))
            fire = d > self.eps
        elif self.trigger == 'prederr':
            num = float(torch.linalg.norm(view.E - self.E_pred))
            den = max(float(torch.linalg.norm(view.E - self.E_prev)), 1e-12)
            self.deltas.append(round(num / den, 4))
            fire = num / den > self.eps
        elif self.trigger == 'always':
            fire = True
        if fire:
            self.K = None                                   # free the old kernel before computing the new one
            self.K = ctx.kernel_full(view.params)
            self.n_relin += 1
            self.relin_t.append(int(t))
            if Ks is not None:
                self.Ks_ref = Ks
        new = None
        if self.replan_every or fire:
            new = self._plan(view.E, view.remaining, ctx, t)
            self.n_plans += 1
        if self.trigger == 'prederr':
            r_next = new if new is not None else self.r
            self.E_prev = view.E.clone()
            self.E_pred = predict_window(self.K, view.E, (r_next, 1 - r_next), ctx.isA, ctx.nA, ctx.nB, ctx.eta, self.M)
        if new is None:
            return None
        if new != self.r:
            self.r = new
            self._log(t, new, 1 - new)
        return (self.r, 1.0 - self.r)

    def summary(self):
        return dict(trace=self.trace, n_relin=self.n_relin, n_plans=self.n_plans, relin_t=self.relin_t,
                    drifts=self.drifts, deltas=self.deltas, M=self.M, eps=self.eps, trigger=self.trigger,
                    replan_every=self.replan_every, two_phase=self.two_phase, plans2=self.plans2)

    def release(self):
        self.K = None
        self.Ks_ref = None


class NormBal(Controller):
    """Equal-norm ratio rule: r = |g_B| / (|g_A| + |g_B|), so (1 - r)|g_B| = r |g_A| (bisector direction)."""
    split_grads = True

    def __init__(self, name, every=1, clip=(0.01, 0.99)):
        super().__init__(name)
        self.every, self.clip = int(every), clip
        self.r = 0.5

    def start(self, ctx):
        return (self.r, 1.0 - self.r)

    def wants_grads(self, t):
        return t % self.every == 0

    def on_grads(self, t, gA, gB, LA, LB, ctx):
        r = gB / (gA + gB) if (gA + gB) > 0 else 0.5
        self.r = min(max(r, self.clip[0]), self.clip[1])
        self._log(t, self.r, 1 - self.r, every=10)
        return (self.r, 1.0 - self.r)


class NormBalLR(Controller):
    """LR-compensated equal-norm rule: new-task weight fixed at s, replay weight s |g_B| / |g_A| (capped so
    that the replay step stays stable on task A). Equivalent to the ratio rule with a time-varying step
    multiplier (1 + lambda_t); in the small-step limit it slides along the old-task solution set."""
    split_grads = True

    def __init__(self, name, every=1, s=0.5):
        super().__init__(name)
        self.every, self.s = int(every), float(s)
        self.lam = 1.0

    def start(self, ctx):
        self.cap = ctx.weight_cap
        return (min(self.s * self.lam, self.cap), self.s)

    def wants_grads(self, t):
        return t % self.every == 0

    def on_grads(self, t, gA, gB, LA, LB, ctx):
        self.lam = gB / gA if gA > 0 else float('inf')
        cA = min(self.s * self.lam, self.cap)
        self._log(t, cA, self.s, every=10)
        return (cA, self.s)


class MegaI(Controller):
    """MEGA-I (Guo et al. 2020) with the new-task weight fixed at s: replay weight s L_A / L_B while
    L_B > tau_B, replay only (weight s) once the new task is inside the band."""

    def __init__(self, name, s=0.5):
        super().__init__(name)
        self.s = float(s)

    def start(self, ctx):
        self.cap = ctx.weight_cap
        return (0.0, self.s)

    def on_losses(self, t, LA, LB, ctx):
        if LB > ctx.tauB:
            cA, cB = min(self.s * LA / max(LB, 1e-30), self.cap), self.s
        else:
            cA, cB = self.s, 0.0
        self._log(t, cA, cB, every=10)
        return (cA, cB)


def build_controllers(methods, ctx, cfg):
    """Lane controllers for the online / gradient-statistics methods (oracle lanes are built by the job)."""
    from .config import parse_method
    mpc, gstat = [], []
    for name in methods:
        m = parse_method(name)
        if m['kind'] == 'mpc':
            mpc.append(MPC(name, m['replan_every'], m['trigger'], m['eps'], ctx.M, m.get('two_phase', False)))
        elif m['kind'] == 'normbal':
            gstat.append(NormBal(name, cfg.grad_every))
        elif m['kind'] == 'normbal_lr':
            gstat.append(NormBalLR(name, cfg.grad_every, cfg.gstat_new_weight))
        elif m['kind'] == 'megaI':
            gstat.append(MegaI(name, cfg.gstat_new_weight))
        elif m['kind'] == 'fixed':
            gstat.append(Fixed(name, m['r']))
    return mpc, gstat
