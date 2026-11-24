#!/bin/bash
# Round-3 CPU queue, resumable after a power cut: every fold (and every external-test unit)
# is saved atomically as soon as it is finished; a restarted queue skips everything already saved.
#   1) external test (anxiety/depression x mixed/twitter, ~8-10 min per unit)
#   2) repeated training runs: fold assignment fixed by --seed 42, training seed 43/44,
#      objectives composite (1,1,1), CE only, focal only; ordered by category so that complete
#      category blocks become available first (anxiety, depression, npd, panic, anger).
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D=../data/multi_disorders
done_job() { local tag=$1 d=$2; for k in 1 2 3 4; do [ -f out/$tag/folds/${d}_f$k.pkl ] || return 1; done; return 0; }
run() { local tag=$1 d=$2; shift 2
        if done_job $tag $d; then echo "skip $tag $d"; return; fi
        echo "=== START $tag $d $(date '+%F %T')"
        python -I run_experiments.py --tag $tag --disorders $d --data-dir $D "$@" >> out/$tag.stdout 2>&1
        echo "=== END $tag $d $(date '+%F %T')"; }
mkdir -p out
n_units=$(ls out/external/units/*.pkl 2>/dev/null | wc -l)
if [ "$n_units" -lt 4 ]; then
  echo "=== START external $(date '+%F %T')"; python -I ext_test.py --data-dir $D --ext-dir ../ext >> out/external.stdout 2>&1; echo "=== END external $(date '+%F %T')"
fi
for d in anxiety depression npd panic anger; do
  run seed43_111   $d --train-seed 43
  run seed43_ce    $d --train-seed 43 --ce 1 --focal 0 --dice 0
  run seed43_focal $d --train-seed 43 --ce 0 --focal 1 --dice 0
  run seed44_111   $d --train-seed 44
  run seed44_ce    $d --train-seed 44 --ce 1 --focal 0 --dice 0
  run seed44_focal $d --train-seed 44 --ce 0 --focal 1 --dice 0
done
echo "QUEUE_R3_DONE $(date '+%F %T')"
