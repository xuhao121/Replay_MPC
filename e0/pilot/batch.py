"""Run exp.run_config over a list of configs with a small process pool (1 BLAS thread each)."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool

def work(cfg):
    import exp, scan
    name, alpha, seed = cfg
    try:
        out = exp.run_config(scan.SCEN[name], alpha, seed)
        out['scen'] = name
    except Exception as e:
        import traceback
        out = dict(scen=name, alpha=alpha, seed=seed, error=traceback.format_exc())
    return out

if __name__ == '__main__':
    cfgs = [tuple(json.loads(x)) for x in sys.argv[2:]]
    outpath = sys.argv[1]
    with Pool(2) as pool, open(outpath, 'a') as f:
        for out in pool.imap_unordered(work, cfgs):
            f.write(json.dumps(out) + '\n'); f.flush()
            short = {k: out.get(k) for k in ['scen', 'alpha', 'seed', 'r_best', 'T_best', 'sec', 'error']}
            for k in ['P_lin2', 'probe_swap', 'probe_piece', 'probe_cont', 'trig_0.05', 'trig_0.2', 'periodic']:
                if k in out: short[k] = (out[k].get('r'), out[k].get('regret'), out[k].get('n_relin'))
            print(json.dumps(short), flush=True)
    print('BATCH DONE', flush=True)
