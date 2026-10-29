"""One worker lane: repeatedly claims the next unfinished job of its owner and runs it.

    python scripts/worker.py --owner m1 --lane 0 --mem-gb 90 --prefer-mem-above 38   # the 'big' lane
    python scripts/worker.py --owner m1 --lane 2 --mem-gb 13                         # a small lane
    python scripts/worker.py --owner m1 --lane 2 --mem-gb 13 --deadline 2026-11-08T20:00
Usually started by scripts/launch_tmux.sh. Stop all lanes gracefully with scripts/stop_workers.sh.
--deadline = when the GPU allocation ends (local time): a lane only starts a job whose estimate times
--slowdown (wall time of a job that shares the GPU / exclusive estimate) still fits before it, so no job is
cut off at the end; small jobs keep filling the tail.
"""
import argparse, datetime, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rmpc import queue as Q
from rmpc.job import run_job

ap = argparse.ArgumentParser()
ap.add_argument('--owner', required=True, help="m1 | m2 | any")
ap.add_argument('--lane', type=int, default=0)
ap.add_argument('--mem-gb', type=float, default=0, help='only take jobs with est_mem_gb <= this (0 = any)')
ap.add_argument('--max-hours', type=float, default=0, help='only take jobs with est_hours <= this (0 = any)')
ap.add_argument('--priorities', nargs='*', default=['P0', 'P1', 'P2', 'P3'])
ap.add_argument('--exp', nargs='*', default=None, help='restrict to these experiments, e.g. E1 E2')
ap.add_argument('--device', default=None)
ap.add_argument('--retry-failed', action='store_true')
ap.add_argument('--stale-min', type=float, default=20)
ap.add_argument('--exit-when-empty', action='store_true')
ap.add_argument('--hard-cap', action='store_true', help='cap this process at --mem-gb of GPU memory')
ap.add_argument('--prefer-mem-above', type=float, default=0, help='within a priority, take jobs above this many GB first')
ap.add_argument('--deadline', default=None, help='end of the GPU allocation, local time, e.g. 2026-11-08T20:00')
ap.add_argument('--slowdown', type=float, default=2.0, help='assumed wall time / estimated exclusive time')
a = ap.parse_args()
deadline = datetime.datetime.fromisoformat(a.deadline).timestamp() if a.deadline else None


def hours_cap():
    """Largest job (estimated exclusive GPU hours) this lane may still start; None = no limit, <= 0 = stop."""
    caps = [a.max_hours] if a.max_hours else []
    if deadline is not None:
        caps.append((deadline - time.time()) / 3600 / a.slowdown)
    return min(caps) if caps else None


dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
if a.hard_cap and a.mem_gb and dev.startswith('cuda'):
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb / total))
tag = f'{a.owner}/lane{a.lane}'
log = lambda s: print(f'{time.strftime("%m-%d %H:%M:%S")} [{tag}] {s}', flush=True)
log(f'start device={dev} mem<={a.mem_gb or "any"} GB priorities={a.priorities} exp={a.exp or "all"}'
    + (f' first>{a.prefer_mem_above:g}GB' if a.prefer_mem_above else '')
    + (f' deadline={a.deadline} slowdown={a.slowdown}' if a.deadline else ''))
idle = 0
while True:
    if os.path.exists(Q.STOP):
        log('STOP file found, exiting')
        break
    cap = hours_cap()
    if cap is not None and cap <= 0.01:
        log('deadline reached, exiting')
        break
    jobs = Q.load_jobs()
    cands = Q.candidates(jobs, a.owner, a.priorities, a.exp, a.mem_gb, cap or 0, a.retry_failed, a.stale_min,
                         a.prefer_mem_above)
    job = None
    for c in cands:
        if Q.try_claim(c, a.stale_min):
            job = c
            break
    if job is None:
        if a.exit_when_empty:
            log('no runnable job left, exiting')
            break
        if idle % 30 == 0:
            extra = f' (only jobs <= {cap:.2f} est. h still fit before the deadline)' if cap is not None else ''
            log(f'queue empty for this lane{extra}; waiting (add jobs with make_jobs.py, or stop with '
                f'stop_workers.sh)')
        idle += 1
        time.sleep(60)
        continue
    idle = 0
    t0 = time.time()
    log(f'run {job.job_id()} ({job.priority}, est {job.est_hours:.2f} h, {job.est_mem_gb:.0f} GB)')
    try:
        with Q.Heartbeat(job):
            run_job(job, os.path.join(Q.RESULTS, job.exp), dev, log=log)
        log(f'done {job.job_id()} in {(time.time() - t0) / 3600:.2f} h')
    except Exception:
        err = traceback.format_exc()
        Q.record_failure(job, err)
        log(f'FAILED {job.job_id()}:\n{err[-2000:]}')
    finally:
        Q.release(job)
        if dev.startswith('cuda'):
            torch.cuda.empty_cache()
