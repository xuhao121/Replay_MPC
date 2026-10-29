"""rmpc: replay-ratio planning and online control (model-predictive) for two-task sequential training.

Modules
    config       JobConfig dataclass and method-name parsing
    data         task pairs (MNIST / CIFAR-10 / sklearn digits / synthetic), task-B transforms
    models       MLP1 (NTK parameterisation, analytic NTK) and ResNet18-k with GroupNorm
    ntk          empirical NTK (analytic, eNTK via torch.func, pseudo-NTK), drift, lambda_max
    linsim       linearised function-space simulator and planner (batched over replay ratios)
    controllers  fixed r, online MPC variants, gradient-statistics controllers
    trainer      phase A training and multi-lane phase-B training (lanes share one stacked model)
    job          one job = one (config, seed): oracle grid + all methods + mechanism measurements
    queue        file-based job queue with atomic claims (several lanes per GPU, two owners)
"""
__version__ = "1.0.0"
