"""Phase A (task A alone) and phase B (mixed objective) training.

Phase B trains several lanes in lockstep: each lane is a copy of the network started at theta_A with its own
controller. Parameters of all active lanes are stacked along a leading axis and evaluated with one vmapped
functional call, so the whole fixed-r grid costs about as much as one run for small models. A lane stops when
it enters (all) tolerance bands, diverges, or reaches the horizon H; finished lanes are compacted away.

Step convention (identical to the CPU pilot): at step t the current losses are evaluated, the band is checked
(T = t if inside), controllers may change the mixing, then one GD update with the current mixing is applied.
"""
import math
import torch
from torch.func import functional_call


class LaneView:
    __slots__ = ('params', 'E', 't', 'remaining')

    def __init__(self, params, E, t, remaining):
        self.params, self.E, self.t, self.remaining = params, E, t, remaining


def _finite(x):
    return x == x and abs(x) != float('inf')


# ============================================================================ phase A
def train_A(prob, etaA, steps_max, target, max_restarts=3, momentum=0.0):
    """Full-batch GD (optionally heavy-ball momentum) on L_A only from theta0 until L_A <= target.
    Halves the step and restarts on divergence."""
    model, alpha = prob.model, prob.alpha
    XA, YA, g0A = prob.X[prob.idxA], prob.Y[prob.idxA], prob.g0[prob.idxA]
    chunk = prob.chunk or XA.shape[0]
    restarts = 0
    while True:
        th = {k: v.clone().requires_grad_(True) for k, v in prob.theta0.items()}
        vel = [torch.zeros_like(v) for v in th.values()] if momentum else None
        lr = etaA / alpha ** 2
        LA_first, ok, t, LAv = None, True, 0, float('nan')
        for t in range(steps_max):
            grads, LAv = None, 0.0
            for s in range(0, XA.shape[0], chunk):
                out = functional_call(model, th, (XA[s:s + chunk],))
                E = alpha * (out - g0A[s:s + chunk]) - YA[s:s + chunk]
                loss = 0.5 * (E * E).sum() / prob.nA
                g = torch.autograd.grad(loss, list(th.values()))
                grads = list(g) if grads is None else [a + b for a, b in zip(grads, g)]
                LAv += float(loss.detach())
            if LA_first is None:
                LA_first = LAv
            if not _finite(LAv) or LAv > 10 * LA_first:
                ok = False
                break
            if LAv <= target:
                break
            with torch.no_grad():
                if vel is None:
                    for p, g in zip(th.values(), grads):
                        p -= lr * g
                else:
                    for p, g, v in zip(th.values(), grads, vel):
                        v.mul_(momentum).add_(g)
                        p -= lr * v
        if ok:
            break
        restarts += 1
        etaA *= 0.5
        if restarts > max_restarts:
            raise RuntimeError('phase A diverged repeatedly')
    return {k: v.detach() for k, v in th.items()}, dict(steps=t, etaA=etaA, restarts=restarts, LA_end=LAv)


# ============================================================================ phase B
def _weights(prob, coef):
    """(R, 2) mixing -> (R, n) per-sample loss weights."""
    return torch.where(prob.isA[None, :], coef[:, :1] / prob.nA, coef[:, 1:] / prob.nB)


def _fwd_nograd(prob, stacked, lane_chunk):
    R = next(iter(stacked.values())).shape[0]
    lc = lane_chunk or R
    sc = prob.chunk or prob.n
    outs = []
    with torch.no_grad():
        for l0 in range(0, R, lc):
            p = {k: v[l0:l0 + lc] for k, v in stacked.items()}
            outs.append(torch.cat([prob.fwd_stacked(p, prob.X[s:s + sc]) for s in range(0, prob.n, sc)], dim=1))
    return torch.cat(outs, dim=0)


def _chunked_grads(prob, stacked, W, lane_chunk):
    R = next(iter(stacked.values())).shape[0]
    lc = lane_chunk or R
    sc = prob.chunk or prob.n
    grads = {k: torch.zeros_like(v) for k, v in stacked.items()}
    for l0 in range(0, R, lc):
        p = {k: v[l0:l0 + lc].detach().requires_grad_(True) for k, v in stacked.items()}
        for s in range(0, prob.n, sc):
            out = prob.fwd_stacked(p, prob.X[s:s + sc])
            E = prob.alpha * (out - prob.g0[s:s + sc]) - prob.Y[s:s + sc]
            loss = 0.5 * (W[l0:l0 + lc, s:s + sc, None] * E * E).sum()
            g = torch.autograd.grad(loss, list(p.values()))
            for k, gi in zip(p.keys(), g):
                grads[k][l0:l0 + lc] += gi
    return grads


def _lane_norms(g, R):
    sq = None
    for v in g:
        s = (v.reshape(R, -1).double() ** 2).sum(1)
        sq = s if sq is None else sq + s
    return sq.sqrt().tolist()


def run_lanes(prob, theta, ctrls, ctx, H, *, split_grads=False, bands=None, snap_outputs=False,
              keep_final=False, record_every=1, batch_size=0, eval_every=1, seed=0, lane_chunk=0):
    """Train len(ctrls) lanes from theta for at most H steps. Returns (list of lane results, finals dict)."""
    dev, dt = prob.device, prob.dtype
    alpha = prob.alpha
    lr = ctx.eta / alpha ** 2
    bands = bands or [(ctx.tauA, ctx.tauB)]
    nb = len(bands)
    R = len(ctrls)
    names = list(theta.keys())
    stacked = {k: theta[k].detach().unsqueeze(0).repeat(R, *([1] * theta[k].dim())).clone().requires_grad_(True)
               for k in names}
    coef = torch.tensor([list(c.start(ctx)) for c in ctrls], dtype=dt, device=dev)
    act = list(range(R))
    lanes = [dict(name=c.name, T=None, T_bands=[None] * nb, band=0, peak=0.0, LA=[], LB=[], rec_t=[],
                  diverged=False, t_end=None, snaps=[None] * nb) for c in ctrls]
    finals = {}
    sgd = batch_size > 0
    gen = torch.Generator(device=dev).manual_seed(int(seed)) if sgd else None
    memsave = (lane_chunk not in (0, None) and lane_chunk < R) or prob.chunk is not None
    if split_grads and memsave:
        raise ValueError('split gradients need the whole lane group in memory (use few lanes, no chunking)')
    LA0, LB0 = ctx.LA0, ctx.LB0

    def params_of(j):
        return {k: v[j].detach() for k, v in stacked.items()}

    for t in range(H + 1):
        if not act:
            break
        Ra = len(act)
        is_eval = (not sgd) or (t % eval_every == 0) or (t == H)
        due = [j for j, i in enumerate(act) if is_eval and ctrls[i].check_due(t)]
        use_graph = (not sgd) and (not memsave) and (split_grads or not due)
        done = set()
        E = None
        if is_eval:
            if use_graph:
                out = prob.fwd_stacked(stacked, prob.X)
            else:
                out = _fwd_nograd(prob, stacked, lane_chunk)
            E = alpha * (out - prob.g0) - prob.Y                      # (Ra, n, C)
            LA_t, LB_t = prob.losses(E.detach())
            LA_l, LB_l = LA_t.tolist(), LB_t.tolist()
            for j, i in enumerate(act):
                ln = lanes[i]
                la, lb = LA_l[j], LB_l[j]
                if t % record_every == 0:
                    ln['LA'].append(la); ln['LB'].append(lb); ln['rec_t'].append(t)
                if not (_finite(la) and _finite(lb)) or la > 1e6 or lb > 1e6:
                    ln['diverged'] = True
                    done.add(j)
                    continue
                ln['peak'] = max(ln['peak'], (la - LA0) / LB0)
                b = ln['band']
                while b < nb and la <= bands[b][0] and lb <= bands[b][1]:
                    ln['T_bands'][b] = t
                    ln['snaps'][b] = prob.test_metrics(params_of(j), keep_outputs=snap_outputs)
                    b += 1
                ln['band'] = b
                if b == nb:
                    done.add(j)
            if t == H:
                done = set(range(Ra))
            for j in done:
                i = act[j]
                lanes[i]['t_end'] = t
                if keep_final:
                    finals[i] = {k: v.clone() for k, v in params_of(j).items()}
            if t == H or len(done) == Ra:
                break
            if done:
                coef[sorted(done)] = 0.0
            for j, i in enumerate(act):
                if j in done:
                    continue
                new = ctrls[i].on_losses(t, LA_l[j], LB_l[j], ctx)
                if new is not None:
                    coef[j] = torch.tensor(new, dtype=dt, device=dev)
            for j in due:
                if j in done:
                    continue
                i = act[j]
                view = LaneView(params_of(j), E[j].detach().float(), t, H - t)
                new = ctrls[i].on_check(t, view, ctx)
                if new is not None:
                    coef[j] = torch.tensor(new, dtype=dt, device=dev)

        # ------------------------------------------------------------ gradient
        P = list(stacked.values())
        if sgd:
            cA, cB = coef[:, 0], coef[:, 1]
            r_l = torch.where(cA + cB > 0, cA / (cA + cB), torch.zeros_like(cA))
            bA = torch.round(r_l * batch_size)
            bA = torch.where(cA > 0, bA.clamp(min=1), bA)
            bA = torch.where(cB > 0, bA.clamp(max=batch_size - 1), bA)
            mask = torch.arange(batch_size, device=dev)[None, :] < bA[:, None]
            ia = prob.idxA[torch.randint(prob.nA, (Ra, batch_size), device=dev, generator=gen)]
            ib = prob.idxB[torch.randint(prob.nB, (Ra, batch_size), device=dev, generator=gen)]
            idx = torch.where(mask, ia, ib)
            out = prob.fwd_stacked(stacked, prob.X[idx], per_lane=True)
            Eb = alpha * (out - prob.g0[idx]) - prob.Y[idx]
            e2 = 0.5 * (Eb * Eb).sum(-1)                               # (Ra, b)
            mA = mask.to(dt)
            if split_grads:
                lA = ((mA * e2).sum(1) / bA.clamp(min=1)).sum()
                lB = (((1 - mA) * e2).sum(1) / (batch_size - bA).clamp(min=1)).sum()
                gA = torch.autograd.grad(lA, P, retain_graph=True)
                gB = torch.autograd.grad(lB, P)
            else:
                w = torch.where(mask, (cA / bA.clamp(min=1))[:, None], (cB / (batch_size - bA).clamp(min=1))[:, None])
                g = torch.autograd.grad((w * e2).sum(), P)
        elif split_grads:
            e2 = 0.5 * (E * E).sum(-1)                                  # (Ra, n)
            lA = e2[:, prob.isA].sum() / prob.nA
            lB = e2[:, ~prob.isA].sum() / prob.nB
            gA = torch.autograd.grad(lA, P, retain_graph=True)
            gB = torch.autograd.grad(lB, P)
        elif use_graph:
            W = _weights(prob, coef)
            g = torch.autograd.grad(0.5 * (W[:, :, None] * E * E).sum(), P)
        else:
            gd = _chunked_grads(prob, stacked, _weights(prob, coef), lane_chunk)
            g = [gd[k] for k in stacked.keys()]

        if split_grads:
            nA_, nB_ = _lane_norms(gA, Ra), _lane_norms(gB, Ra)
            for j, i in enumerate(act):
                if j in done or not ctrls[i].wants_grads(t):
                    continue
                new = ctrls[i].on_grads(t, nA_[j], nB_[j], lanes[i]['LA'][-1] if lanes[i]['LA'] else None,
                                        lanes[i]['LB'][-1] if lanes[i]['LB'] else None, ctx)
                if new is not None:
                    coef[j] = torch.tensor(new, dtype=dt, device=dev)
            if done:
                coef[sorted(done)] = 0.0
            shp = lambda v: (Ra,) + (1,) * (v.dim() - 1)
            g = [coef[:, 0].reshape(shp(a)) * a + coef[:, 1].reshape(shp(b)) * b for a, b in zip(gA, gB)]

        with torch.no_grad():
            for p, gi in zip(P, g):
                p -= lr * gi
        E = None
        if done:
            keep = [j for j in range(Ra) if j not in done]
            stacked = {k: v[keep].detach().clone().requires_grad_(True) for k, v in stacked.items()}
            coef = coef[keep]
            act = [act[j] for j in keep]

    results = []
    for i, ln in enumerate(lanes):
        ln['T'] = ln['T_bands'][0]
        ln['ctrl'] = ctrls[i].summary()
        del ln['band']
        results.append(ln)
    return results, finals
