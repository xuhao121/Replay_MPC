"""Second scan: narrower networks (more kernel drift) and a very rich regime."""
import json, time
import scan
scan.SCEN.update({
    'rot_h64':  dict(transform='rot90', nA=150, nB=150, tolA=0.03, tolB=0.1, h=64),
    'perm_h64': dict(transform='perm',  nA=150, nB=150, tolA=0.03, tolB=0.1, h=64),
})
if __name__ == '__main__':
    out = []
    for name, alphas in [('rot_h64', [1.0, 0.05, 0.02]), ('perm_h64', [1.0, 0.05, 0.02]), ('rot_sym', [0.02])]:
        for alpha in alphas:
            t = time.time(); res = scan.one(name, alpha); res['sec'] = round(time.time() - t)
            print(json.dumps({k: res[k] for k in ['scen', 'alpha', 'r_lin', 'r_true', 'T_true_at_rlin', 'T_best', 'regret', 'sec']}), flush=True)
            out.append(res)
    json.dump(out, open('scan2.json', 'w'), indent=1)
