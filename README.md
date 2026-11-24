# Reproducibility package: IEEE Access manuscript Access-2026-50336 (round 3)

"Identifying Linguistic Patterns Associated with Mental Health Risk Categories in Social Media Using a
Transformer-Based Ensemble of Binary Classifiers" (O. Ovcharuk et al.).

This folder reproduces every number reported in the revised manuscript. It is tagged `v3.0-ieee-access-r3`
in the GitHub repository and archived on Zenodo.

## Data
* Source corpora (already in the repository, `datasets/multi_disorders/`): `train.csv` (Kaggle "Text Classification",
  785 tweets, five categories) and `depression_dataset_reddit_cleaned.csv` (Kaggle "Depression: Reddit Dataset (Cleaned)").
  The per-category files `AngerIntermittentExplosiveDisorder.csv`, `AnxietyDisorder.csv`, `Depression.csv`,
  `NarcissisticDisorder.csv`, `PanicDisorder.csv` are the balanced subsets of the first version (mixed negatives).
* `subsets/<arm>/<category>.csv`: the cleaned subsets actually used (exact duplicates and label-conflicting texts removed),
  with `row_id`, `text`, `text_normalized`, label `y` (1 = target category), and the source of each text.
  Arms: `mixed` (main experiment), `twitter` (Twitter-only negatives), `reddit` (Reddit-only negatives).
* External test: 2018 posts of r/anxiety, r/depression, r/teaching, r/divorce, r/guns, r/fitness from the Reddit Mental
  Health Dataset (Low et al., JMIR 2020; Zenodo 10.5281/zenodo.3941387, ODC-PDDL). The 1,200 sampled posts and all scores
  are in `results/external/external_scores.csv`.

## Seeds and splits
* Fold assignment: `StratifiedKFold(n_splits=4, shuffle=True, random_state=42)` on the cleaned subset.
* Inner validation split of fold k: `train_test_split(test_size=0.2, stratify=y, random_state=42+k)`.
* Training seed of fold k: 42+k (main experiment and all ablations); 43+k and 44+k (repeated runs, tags `seed43_*`, `seed44_*`).
  The training seed sets the classification-head initialization, dropout masks, and mini-batch order; folds and inner splits
  are the same in every run.
* `splits/<arm>/<category>.csv` lists, for every fold, the role (`train`, `val`, `test`) of every `row_id`.

## Software and determinism
See `requirements.txt`. The main configuration, the repeated runs, the external test, and the cost measurements were run on
a CPU (Intel Core i5-10210U, 4 threads), where the computation is deterministic: re-running fold 1 of Anger/IED reproduces the
stored out-of-fold scores exactly (maximum absolute difference 0.0). The remaining round-2 ablations were run on a GPU
(NVIDIA GeForce MX250, torch 2.0.1+cu117), where small floating-point differences are possible.

## Commands (from this folder; the data folder is `../datasets/multi_disorders`)
```
python code/check_invariants.py --data-dir ../datasets/multi_disorders                                          # Section II-C: subset invariants I1-I7 (exit code 1 on failure)
python code/run_experiments.py --tag main_mixed_111_cpu --data-dir ../datasets/multi_disorders                # Table II, Figs. 4-9
python code/run_experiments.py --tag arm_twitter_111 --negatives twitter --data-dir ../datasets/multi_disorders   # Table III
python code/run_experiments.py --tag arm_reddit_111  --negatives reddit  --data-dir ../datasets/multi_disorders   # Table III
python code/run_experiments.py --tag loss_ce_only --ce 1 --focal 0 --dice 0 --data-dir ../datasets/multi_disorders # Table IV (one row per weighting)
python code/run_experiments.py --tag seed43_111 --train-seed 43 --data-dir ../datasets/multi_disorders            # Table V (also _ce/_focal, seed 44)
python code/run_experiments.py --tag npd_shap --disorders npd --shap --data-dir ../datasets/multi_disorders        # Fig. 10
python code/run_experiments.py --tag lc_anx_mixed_025 --disorders anxiety --train-frac 0.25 --data-dir ../datasets/multi_disorders  # Fig. 11
python code/audit_r3.py --data-dir ../datasets/multi_disorders                                                    # Table VI, Section III-E
python code/ext_test.py --data-dir ../datasets/multi_disorders --ext-dir <folder with the six Reddit files>         # Table VII, Table VIII
python code/seeds_r3.py                                                                                           # Table V statistics
```
`code/queue_r3.sh` runs the round-3 jobs; every fold is checkpointed, so an interrupted run resumes where it stopped.

## Results
`results/<tag>/` holds, for every run: `summary.csv` (pooled out-of-fold metrics with bootstrap CI), `fold_metrics.csv`
(per-fold metrics, thresholds, timing), `oof_scores.csv` (out-of-fold score, threshold, and prediction of every text),
`loss_history.csv`, `subset_info.json`, and `log.txt`. `results/round2_superseded/` keeps the round-2 audit files, which were
computed before the removal of label-conflicting texts and are replaced by `results/audit_r3.json`.

## Correspondence with the manuscript
| Manuscript item | Files |
|---|---|
| Section II-C, Section III-A (subset construction) | results/invariants_r3.json: invariants I1-I7 for the authors' subset files as delivered (violations in Anger/IED 5, Anxiety 30, Depression 8 texts; keep-first de-duplication would keep 3 / 13 / 4 of them as negatives) and for the final subsets of all three negative-sampling arms (all pass) |
| Table II, Figs. 4-9 | results/main_mixed_111_cpu |
| Table III | results/main_mixed_111_cpu, results/arm_twitter_111, results/arm_reddit_111 |
| Table IV | results/main_mixed_111_cpu and results/loss_* |
| Table V | results/seeds_r3.json (from main/loss_ce_only/loss_focal_only and seed43_*/seed44_*) |
| Table VI, Section III-E | results/audit_r3.json |
| Table VII | results/external/external_metrics.csv |
| Table VIII | results/main_mixed_111_cpu/log.txt, results/external/cost.csv, results/audit_r3.json |
| Fig. 10 | results/npd_shap |
| Fig. 11 | results/lc_anx_* and results/main_mixed_111_cpu, results/arm_twitter_111 |
