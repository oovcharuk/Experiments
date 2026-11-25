#!/bin/bash
# Round 3, after review (24.11.2026): minimal runs on the Anxiety subset rebuilt with the intended sampling rule
# (all 145 positives, NPD share restored). About 30-40 min on the CPU; per-fold / per-unit checkpoints, so a rerun
# after a power cut continues where it stopped.
cd /c/Code_/ovcharuk_resubmission/round3_20261123/exp || exit 1
D=../data/multi_disorders_anxiety_rebuilt
[ -f $D/AnxietyDisorder.csv ] || python -I rebuild_anxiety_subset.py
echo "=== START anx_rebuilt_111 $(date '+%F %T')"
python -I run_experiments.py --tag anx_rebuilt_111 --disorders anxiety --data-dir $D >> out/anx_rebuilt.stdout 2>&1
echo "=== END anx_rebuilt_111 $(date '+%F %T')"
echo "=== START audit rebuilt $(date '+%F %T')"
python -I audit_r3.py --data-dir $D --out out/audit_anxiety_rebuilt.json >> out/anx_rebuilt.stdout 2>&1
echo "=== END audit rebuilt $(date '+%F %T')"
echo "=== START external anxiety mixed rebuilt $(date '+%F %T')"
python -I ext_test.py --data-dir $D --ext-dir ../ext --targets anxiety --arms mixed --out out/external_anx_rebuilt >> out/anx_rebuilt.stdout 2>&1
echo "=== END external anxiety mixed rebuilt $(date '+%F %T')"
echo "ANX_REBUILT_DONE $(date '+%F %T')"
