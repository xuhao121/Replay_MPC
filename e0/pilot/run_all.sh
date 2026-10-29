#!/bin/bash
# Reproduce everything on CPU (about 2 h on 2 cores). Needs numpy, scikit-learn, matplotlib.
set -e
cd "$(dirname "$0")"
python3 scan.py                      # scenario scan, seed 0 (about 12 min)
python3 scan2.py                     # narrow networks / very rich regime (about 7 min)
ARGS=()
for s in 0 1 2; do
  for sa in "rot_h64 0.05" "rot_h64 0.02" "rot_tightA 0.05" "perm_h64 0.02" "rot_sym 0.05" \
            "rot_h64 1.0" "rot_tightA 1.0" "perm_h64 1.0" "rot_sym 1.0"; do
    set -- $sa; ARGS+=("[\"$1\", $2, $s]")
  done
done
rm -f results.jsonl
python3 batch.py results.jsonl "${ARGS[@]}"   # main comparison, 27 configs (about 35 min)
python3 exp_extra.py                         # re-plan-every-check variants (about 15 min)
python3 exp_prederr.py                       # Jacobian-free prediction-error trigger (about 40 min)
python3 aggregate.py                         # table by regime
python3 fig_curves.py                        # figure
