# replay_mpc — predicting and controlling the replay ratio

Two tasks are learned in sequence (A, then B). During the B phase a fraction r of the loss weight goes to
replayed A data. This repository measures how fast both tasks get back inside a tolerance band, T(r), and
tests three ways of choosing r:

* **P_lin2** — linearise the network at the task switch (NTK) and predict T(r) for every r before training;
* **online MPC** — re-plan r from the current state every window, re-linearise only when the kernel has
  drifted (drift trigger) or the linear model mispredicts (prediction-error trigger), optionally with
  two-phase plans;
* **gradient-statistics rules** — the equal-norm ratio rule runs in E1 as a descriptive control; its
  LR-compensated version and MEGA-I stay in the code (tests, E0) but are no longer in the GPU grids.

Method lists per experiment (`rmpc/grids.py`): E1 runs 9 methods (oracle, plin2, mpc_0.05, mpcpe_0.2, mpc2_0.05,
trig_0.05, replan_only, periodic, normbal; the main-line rot_sym setting adds the threshold variants mpc_0.02,
mpc_0.1, mpcpe_0.1 for the appendix), E2 runs 6 (no mpc2, periodic, normbal), E1N 4, E1W only oracle and plin2.

The experiment plan and the theory are in the two project documents; this README is the operating manual.

## Layout

```
rmpc/            the library
  config.py      JobConfig (one job = one setting + seed) and method names
  data.py        MNIST / CIFAR-10 / digits / synthetic task pairs (task B = transformed inputs)
  models.py      MLP1 (analytic NTK) and pre-activation ResNet18-k (GroupNorm, centred ReLU)
  problem.py     data + model + Chizat output scaling f = alpha (g - g0)
  ntk.py         analytic NTK, eNTK (torch.func), pseudo-NTK, drift
  linsim.py      linearised function-space simulator, planners (constant r, two-phase), lambda_max(K W)
  controllers.py fixed r, MPC variants, normbal, normbal_lr, MEGA-I
  trainer.py     phase A, multi-lane phase B (lanes stacked with vmap)
  job.py         one job: oracle grid + every method + mechanism measurements -> results/<exp>/<id>.json/.npz
  grids.py       experiment grids P0-P3, method lists and the cost model
  queue.py       file-based job queue (atomic claims, heartbeats, two owners)
scripts/         prepare_data, calibrate, bench, make_jobs, worker, launch_tmux, mps, status, stop_workers
analysis/        aggregate.py (tables + pre-registered tests), figures.py, gate_check.py (gates G1-G3)
tests/           parity with the CPU pilot, kernels, simulator, end-to-end smoke tests, grid sizes
                 (test_grids.py) and the analysis on synthetic results (test_analysis.py)
e0/              CPU theory checks (linear_theory.py, two_phase.py, sim2.py) and the CPU pilot (e0/pilot)
```

## Day 1 (about 4 hours; m1 = member 1's GPU, m2 = member 2's GPU)

```bash
# 0. m1 once: put this folder in a private git repository shared with m2; calib.json, bench.json,
#    jobs/jobs.jsonl and results/ travel through it (git init && git add . && git commit && git push)
# 1. both: environment (Python >= 3.10, PyTorch >= 2.3 with CUDA); repo and data on persistent storage
pip install -r requirements.txt
export RMPC_DATA=/persistent/path/data            # optional; default ./data
python scripts/prepare_data.py                    # MNIST and CIFAR-10 -> data/*.npz

# 2. both: tests on your own GPU - parity with the CPU pilot (exact step counts), kernels, simulator, smoke
python tests/test_ntk.py && python tests/test_linsim.py
python tests/test_pilot_parity.py && python tests/test_job_pilot.py
python tests/test_smoke.py
python tests/test_grids.py && python tests/test_analysis.py     # no training, about a minute

# 3. both: MPS (several processes share the GPU); add the two exports it prints to ~/.bashrc
bash scripts/mps.sh start

# 4a. m1: speed -> bench.json (about 2 h), then commit bench.json
python scripts/bench.py single
python scripts/bench.py concurrent --procs 2 4 6 8
# 4b. m2, at the same time: laziness calibration -> calib.json (about 1.5 h), then commit calib.json
python scripts/calibrate.py --exp E1 --alphas 0.003 0.01 0.03 0.1 0.3 1 3
python scripts/calibrate.py --exp E2 --alphas 0.1 0.2 0.3 0.5 1 2 3 5 10 20 50 100   # 6 + 2 distinct alphas
python scripts/calibrate.py --exp E1W --alphas 3 10 30 --width 8      # check: width 8 stays lazy
python scripts/calibrate.py --exp E1N --alphas 1 0.03 --batch 32      # check: SGD still reaches the band

# 5. gate G1 (see the plan). m1 pulls calib.json, checks, creates the core jobs, commits jobs/jobs.jsonl; m2 pulls
python analysis/gate_check.py G1
python scripts/make_jobs.py --phases P0 P1 P2 --calib calib.json --update-estimates

# 6. both: launch on your own GPU; one budget (GB) per lane; --deadline = end of your GPU allocation
bash scripts/launch_tmux.sh m1 "90 38 13 13 13" "--deadline 2026-11-08T20:00"     # m2: same with m2
```

Gate G1: all tests pass on both GPUs; calibration drift spans both regimes (at least 2 chosen alphas on each
side of 0.2, for E1 and for E2); E2 reaches the band in < 10^4 steps (otherwise see "risks" in the plan: smaller n, looser tol_B, k = 32 + pNTK); bench within 2x of the
estimates (`analysis/gate_check.py G1` checks everything except the tests). One lane configuration serves
the whole run: the 90 GB lane first takes the jobs above 38 GB (E1 with n = 10000, ResNet k = 64), the 38 GB
lane the other ResNet jobs and the width-4096 MLP, the 13 GB lanes the remaining MLP jobs. The deadline in
the example is a placeholder: use the time your GPU allocation ends.

## Running

```bash
python scripts/status.py --lanes 5                  # progress, GPU hours used, estimate left
tmux attach -t rmpc-m1                              # watch the lanes
python analysis/gate_check.py G2                    # day 2 morning, on the merged P0 results
python scripts/make_jobs.py --phases P0 P1 P2 P3 --calib calib.json   # gate G2 passed: add the filler
python analysis/gate_check.py G3                    # day 4 morning: P0-P2 complete -> freeze the analysis code
python scripts/make_jobs.py --phases P3 --calib calib.json --seeds-p3-e2 3 6 --seeds-p3-e1 3 5   # queue dry
bash scripts/stop_workers.sh                        # graceful stop after the current jobs
python scripts/worker.py --owner m1 --retry-failed  # re-run failed jobs after a fix
git add results && git commit -m "results day N" && git push       # daily; files never conflict
```

Seeds 3 and up add no settings (the unit of analysis), so P3 first runs new settings and scale points; seeds
are added only when the queue runs dry. New jobs are picked up by running lanes within a minute; no relaunch
is needed. With `--deadline` a lane
only starts a job whose estimate x 2 still fits before the deadline, so nothing is cut off at the end and the
small jobs fill the tail.

## Analysis

```bash
python analysis/aggregate.py          # results/summary/: summary.md, tests.json (verdicts), tests_e2.json, *.csv
python analysis/figures.py            # results/figures/fig1..fig7 (.png, .pdf); fig5 = appendix, from e0/results
```

Every pre-registered hypothesis gets SUPPORT, REJECT or INCONCLUSIVE (support and rejection criteria of the
plan). E2 is also evaluated alone (descriptive). The cost table reports the wall-clock regret: T single-lane
steps plus the method's own kernels and plans, divided by the time of training with r* directly; `vs_grid`
divides the same wall time by a full fixed-r grid search.

## Outputs of one job

`results/<exp>/<job_id>.json`: config, phase-A statistics, eta, H, M, the linearised curve T_lin(r), the
oracle curve T(r) (+ edge-normalised oracle), kernel drift, and per method: T, regret, regret_edge, number
of re-linearisations / plans, r(t) trace, test metrics at T, and for online controllers `ctrl_sec` (wall time
of their own kernels and plans); `timing`: step_sec (one training step of a single lane, measured on a short
fixed-r run), KA_sec, plin2_sec (the P_lin2 decision: the grid simulated up to the first predicted arrival,
which is what the wall-clock regret charges), plin2_curve_sec (the full predicted curve, a diagnostic), KsA_sec.
`<job_id>.npz`: loss curves of every lane and the test outputs of the
oracle lanes at T (destination analysis).

## E0 and the CPU pilot (no GPU)

```bash
python e0/linear_theory.py            # exact checks in a linear model (5 min)
python e0/two_phase.py                # two-phase schedules (20 min)
python e0/sim2.py                     # replay-weighting rules, step size tuned per method -> e0/results/sim2.json
python e0/extra_checks.py             # ratio rho and schedule invariance in the sim2 problems
bash e0/pilot/run_all.sh              # the 27-configuration pilot incl. the prediction-error trigger (2 h)
```

## Conventions

* T counts GD updates; the band is checked before each update (same as the pilot).
* Fixed-r and MPC lanes use convex weights (r, 1 - r) and the same eta. normbal_lr and megaI change the
  total weight (= the step size), so they are compared with `oracle_edge`, never with the plain oracle.
* `dtype` float32 by default; `tests/test_pilot_parity.py` runs float64 for bit-level parity.
