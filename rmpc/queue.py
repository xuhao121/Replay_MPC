"""File-based job queue for one machine: several worker lanes (processes) share one GPU.

    jobs/jobs.jsonl                 one JobConfig per line (written by scripts/make_jobs.py; has 'owner')
    results/<exp>/<job_id>.json     result = done marker (written atomically at the very end of a job)
    results/_claims/<job_id>.claim  claimed by a running lane (created with O_EXCL; mtime = heartbeat)
    results/_failed/<job_id>.txt    traceback of a failed job (not retried unless --retry-failed)
    STOP                            if this file exists, lanes exit after their current job

Two owners (m1, m2) on two machines: each runs only its own jobs, results are merged through git
(distinct files per job, so merges never conflict).
"""
import json, os, socket, threading, time, traceback
from .config import JobConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.environ.get('RMPC_JOBS', os.path.join(ROOT, 'jobs', 'jobs.jsonl'))
RESULTS = os.environ.get('RMPC_RESULTS', os.path.join(ROOT, 'results'))
CLAIMS = os.path.join(RESULTS, '_claims')
FAILED = os.path.join(RESULTS, '_failed')
STOP = os.environ.get('RMPC_STOP', os.path.join(ROOT, 'STOP'))
PRIO = {'P0': 0, 'P1': 1, 'P2': 2, 'P3': 3}


def load_jobs(path=JOBS):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(JobConfig.from_dict(json.loads(line)))
    return out


def result_path(cfg):
    return os.path.join(RESULTS, cfg.exp, f'{cfg.job_id()}.json')


def is_done(cfg):
    return os.path.exists(result_path(cfg))


def is_failed(cfg):
    return os.path.exists(os.path.join(FAILED, f'{cfg.job_id()}.txt'))


def claim_path(cfg):
    return os.path.join(CLAIMS, f'{cfg.job_id()}.claim')


def claim_live(cfg, stale_min):
    p = claim_path(cfg)
    if not os.path.exists(p):
        return False
    try:
        return (time.time() - os.path.getmtime(p)) < stale_min * 60
    except FileNotFoundError:
        return False


def try_claim(cfg, stale_min=20):
    os.makedirs(CLAIMS, exist_ok=True)
    p = claim_path(cfg)
    if os.path.exists(p) and not claim_live(cfg, stale_min):
        try:
            os.remove(p)                       # stale claim of a killed lane
        except FileNotFoundError:
            pass
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        f.write(f'{socket.gethostname()} pid={os.getpid()} t={time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    return True


def release(cfg):
    try:
        os.remove(claim_path(cfg))
    except FileNotFoundError:
        pass


class Heartbeat:
    def __init__(self, cfg, every=60):
        self.p, self.every = claim_path(cfg), every
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            try:
                os.utime(self.p, None)
            except FileNotFoundError:
                pass

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()


def candidates(jobs, owner, priorities, exps, mem_gb, max_hours, retry_failed, stale_min, prefer_above=0):
    """Runnable jobs in queue order: priority, then (optionally) jobs needing more than `prefer_above` GB
    first (the big lane takes the big jobs early), then file order."""
    out = []
    for i, j in enumerate(jobs):
        if owner != 'any' and j.owner != owner:
            continue
        if priorities and j.priority not in priorities:
            continue
        if exps and j.exp not in exps:
            continue
        if mem_gb and j.est_mem_gb > mem_gb:
            continue
        if max_hours and j.est_hours > max_hours:
            continue
        if is_done(j) or (is_failed(j) and not retry_failed) or claim_live(j, stale_min):
            continue
        big = 0 if (prefer_above and j.est_mem_gb > prefer_above) else 1
        out.append((PRIO.get(j.priority, 9), big, i, j))
    out.sort(key=lambda x: x[:3])
    return [x[3] for x in out]


def record_failure(cfg, err):
    os.makedirs(FAILED, exist_ok=True)
    with open(os.path.join(FAILED, f'{cfg.job_id()}.txt'), 'w') as f:
        f.write(f'{socket.gethostname()} {time.strftime("%Y-%m-%d %H:%M:%S")}\n{json.dumps(cfg.to_dict())}\n{err}\n')
