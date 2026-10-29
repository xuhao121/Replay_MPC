"""Job configuration. One job = one (setting, seed); it runs the fixed-r oracle grid and every method.

Method names (cfg.methods):
    oracle          fixed r on the whole grid (real training; gives T_best and the T(r) curve). Always run.
    oracle_edge     fixed r with an edge-normalised step: weights mu_r (r, 1 - r), mu_r = edge_target /
                    (eta lambda_max(K_A W_r)). Comparator for controllers whose weights do not sum to 1.
    plin2           linearise both tasks at theta_A, predict T(r) for the grid, pick once (no extra training)
    probe_swap      offline probe: train probe_frac * T_hat steps at r_lin, take the kernel there, re-plan from
                    the start, pick once (negative control; pilot worst case 1.19-1.25)
    mpc_<eps>       MAIN METHOD: re-plan r from the current state at every check; re-linearise only when the
                    kernel drift on a fixed sample subset since the last linearisation exceeds eps
    mpcpe_<eps>     same, but the trigger is the linear model's one-window prediction error (no Jacobian;
                    pilot: eps = 0.2 matches the drift trigger at the same number of re-linearisations)
    mpc2_<eps>      mpc_<eps> with two-phase plans: candidates 'r1 for the next j windows (j = 1, 2, 4, 8), then
                    r2' plus every constant r; r1 is applied and re-planned at the next check (E0: captures the
                    5-18% gain of time-varying schedules that constant-r re-planning misses)
    trig_<eps>      re-linearise and re-plan only when drift > eps, otherwise keep r (pilot 'trig')
    periodic        re-linearise and re-plan at every check (most kernel computations)
    replan_only     re-plan at every check with the stale theta_A kernel (isolates feedback from re-linearising)
    normbal         equal-norm ratio rule r = |g_B| / (|g_A| + |g_B|)   (theory: stalls)
    normbal_lr      LR-compensated version: new-task weight s = gstat_new_weight (1), replay weight s |g_B| / |g_A|
                    capped at the stability limit (theory: tuning-free, not faster than the best tuned fixed r)
    megaI           MEGA-I: new-task weight s, replay weight s L_A / L_B (same cap); replay only once L_B is in the
                    band. normbal_lr and megaI do not keep the weights summing to 1, i.e. they also change the
                    step size, so they are compared with oracle_edge (best fixed r at its own stable step), not
                    with the convex oracle (fixed r at the shared eta). The pilot showed the confound: against the
                    convex oracle normbal_lr looked 2x faster when the band on task A was tight.
"""
from dataclasses import dataclass, field, asdict, fields
import hashlib, json

R_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))          # 0.05 ... 0.95
PILOT_GRID = (0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9)

DEFAULT_METHODS = ('oracle', 'oracle_edge', 'plin2', 'probe_swap', 'mpc_0.05', 'mpc_0.02', 'mpc_0.1', 'mpcpe_0.2',
                   'mpcpe_0.1', 'mpc2_0.05', 'trig_0.05', 'periodic', 'replan_only', 'normbal', 'normbal_lr', 'megaI')


@dataclass
class JobConfig:
    exp: str = 'E1'                 # E1 (MLP) | E2 (ResNet) | E1W (width threshold) | E1N (SGD noise) | ...
    tag: str = ''                   # free text, e.g. scenario name
    priority: str = 'P0'
    seed: int = 0
    # ---- data
    dataset: str = 'mnist'          # mnist | cifar10 | digits | synth | synth_img
    transform: str = 'rot90'        # rot90 | perm | neg  (applied to task B inputs)
    nA: int = 1000
    nB: int = 1000
    n_test: int = 1000              # per task (0 = no test set)
    # ---- model
    model: str = 'mlp'              # mlp | resnet18k
    width: int = 1024               # MLP hidden width / ResNet base width k
    act: str = 'gelu'               # ResNet activation after GroupNorm: gelu | relu (both centred); MLP ignores it
    alpha: float = 1.0              # Chizat output scale: f = alpha (g(theta) - g(theta0)), loss / alpha^2
    kernel: str = 'auto'            # auto | analytic | entk | pntk
    init: str = 'torch'             # torch | numpy_pilot (bit-for-bit init of the CPU pilot)
    # ---- tolerance band (relative to L_B(theta_A))
    tolA: float = 0.03
    tolB: float = 0.1
    # ---- phase A (train task A alone from theta0)
    targetA_rel: float = 2e-3       # stop when L_A <= targetA_rel * L_A(theta0)
    stepsA_max: int = 20000
    optA: str = 'gd'                # gd | momentum (heavy ball, beta 0.9) - phase A only; phase B is always plain GD
    etaA_rule: str = 'pilot'        # pilot: 0.9 m / lambda_max(K(theta0) on all samples) | taskA: 0.9 nA / lambda_max(K_AA)
    # ---- phase B
    lr_safety: float = 0.9          # eta = lr_safety * min(nA, nB) / lambda_max(K(theta_A))
    H: int = 0                      # horizon; 0 = clip(H_mult * T_hat, H_min, H_max)
    H_min: int = 1000
    H_max: int = 20000
    H_mult: float = 4.0
    n_checks: int = 10              # check interval M = max(M_min, int(T_hat / n_checks))
    M_min: int = 20
    r_grid: tuple = R_GRID
    batch_size: int = 0             # 0 = full-batch GD; >0 = minibatch SGD with replay composition r
    eval_every: int = 1             # SGD: full training-set evaluation interval (T resolution)
    # ---- methods
    methods: tuple = DEFAULT_METHODS
    drift_subset: int = 30          # samples per task in the fixed drift subset
    probe_frac: float = 0.15
    grad_every: int = 1             # normbal / normbal_lr update interval (steps)
    gstat_new_weight: float = 1.0   # new-task weight s of normbal_lr and megaI
    edge_target: float = 1.8        # eta * lambda_max(K_A W) for oracle_edge lanes and the replay-weight cap (GD limit 2)
    dest_tight: bool = False        # continue oracle lanes to a tight band (destination analysis)
    tight_tolA: float = 0.01
    tight_tolB: float = 0.03
    # ---- numerics
    dtype: str = 'float32'          # dtype of real training (simulator is always float32)
    tf32: bool = False
    lane_chunk: int = 0             # lanes per vmap call (0 = all)
    sample_chunk: int = 0           # samples per forward chunk (0 = all); >0 switches to the memory-saving step
    jac_chunk: int = 8              # samples per vmap(jacrev) call for generic eNTK
    jac_dtype: str = 'float32'      # storage dtype of the Jacobian for generic eNTK (float32 | bfloat16)
    record_every: int = 1           # loss-curve recording interval
    mpc_group: int = 0              # MPC lanes trained together (each may hold its own full kernel); 0 = all
    full_drift_max_rows: int = 40000  # compute full-kernel drift of the best run only if n*C <= this
    # ---- bookkeeping (filled by make_jobs)
    est_hours: float = 0.0
    est_mem_gb: float = 0.0
    owner: str = ''

    # ------------------------------------------------------------------
    def job_id(self):
        keys = ['exp', 'tag', 'dataset', 'transform', 'nA', 'nB', 'model', 'width', 'alpha', 'kernel',
                'tolA', 'tolB', 'batch_size', 'seed']
        short = '_'.join(str(getattr(self, k)) for k in ['exp', 'tag', 'model', 'width', 'alpha', 'batch_size', 'seed'])
        skip = {'priority', 'est_hours', 'est_mem_gb', 'owner'} | ({'act'} if self.model == 'mlp' else set())
        full = json.dumps({k: v for k, v in self.to_dict().items() if k not in skip}, sort_keys=True)
        h = hashlib.sha1(full.encode()).hexdigest()[:8]
        return f"{short}_{h}".replace('/', '-')

    def to_dict(self):
        d = asdict(self)
        d['r_grid'] = list(self.r_grid)
        d['methods'] = list(self.methods)
        return d

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        if 'r_grid' in d:
            d['r_grid'] = tuple(float(x) for x in d['r_grid'])
        if 'methods' in d:
            d['methods'] = tuple(d['methods'])
        return cls(**d)

    def resolved_kernel(self):
        if self.kernel != 'auto':
            return self.kernel
        return 'analytic' if self.model == 'mlp' else 'entk'


def parse_method(name):
    """-> dict describing the method. Raises ValueError for unknown names."""
    if name in ('oracle', 'oracle_edge', 'plin2', 'probe_swap', 'normbal', 'normbal_lr', 'megaI'):
        return dict(kind=name)
    if name == 'periodic':
        return dict(kind='mpc', replan_every=True, trigger='always', eps=0.0)
    if name == 'replan_only':
        return dict(kind='mpc', replan_every=True, trigger='none', eps=float('inf'))
    for prefix, replan, trig, two in (('mpc2pe_', True, 'prederr', True), ('mpc2_', True, 'drift', True),
                                      ('mpcpe_', True, 'prederr', False), ('mpc_', True, 'drift', False),
                                      ('trig_', False, 'drift', False)):
        if name.startswith(prefix):
            return dict(kind='mpc', replan_every=replan, trigger=trig, eps=float(name[len(prefix):]), two_phase=two)
    if name.startswith('fixed_'):
        return dict(kind='fixed', r=float(name[len('fixed_'):]))
    raise ValueError(f'unknown method {name}')
