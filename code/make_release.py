#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the reproducibility package (round 3) for the GitHub tag and the Zenodo archive.

Layout (to be copied into the repository as folder  experiments_r3/ ):
  README.md                      how to reproduce every number; seeds; software versions
  requirements.txt               exact versions used
  code/                          run_experiments.py, check_invariants.py, ext_test.py, audit_r3.py, seeds_r3.py, queue_r3.sh, make_release.py,
                                 fill_numbers_r2.py (round-2 table generator, kept for provenance)
  subsets/<arm>/<category>.csv   cleaned subsets with row_id, text, normalized text, label y, source
  splits/<arm>/<category>.csv    fold (test fold) and inner role (train/val/test) of every row_id in every fold
  results/<run tag>/             summary.csv, fold_metrics.csv, oof_scores.csv, loss_history.csv, subset_info.json, log.txt
  results/external/              external_metrics.csv, external_scores.csv (no post texts), cost.csv, log.txt
  results/invariants_r3.json, results/audit_r3.json, results/seeds_r3.json
"""
import io, json, shutil, sys, zipfile
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as rx  # noqa: E402

REL = HERE.parent / "release" / "experiments_r3"
R2ZIP = HERE.parent.parent / "experiments_R2.zip"
R2_TAGS = ["main_mixed_111_cpu", "arm_twitter_111", "arm_reddit_111", "loss_ce_only", "loss_focal_only", "loss_dice_only",
           "loss_ce_focal", "loss_ce_dice", "loss_focal_dice", "loss_211", "loss_122", "npd_shap",
           "lc_anx_mixed_025", "lc_anx_mixed_050", "lc_anx_mixed_075", "lc_anx_tw_025", "lc_anx_tw_050", "lc_anx_tw_075"]
R3_TAGS = ["seed43_111", "seed43_ce", "seed43_focal", "seed44_111", "seed44_ce", "seed44_focal"]
KINDS = ["summary.csv", "fold_metrics.csv", "oof_scores.csv", "loss_history.csv", "subset_info.json", "log.txt", "splits.csv", "shap_tokens.csv", "shap_summary.json"]


def strip_external_texts(src_csv, dst_csv, ext_dir):
    """Replace the post text of external_scores.csv by (source_file, source_row) in the public Reddit Mental Health
    Dataset; the texts and user names of that dataset are not redistributed."""
    sc = pd.read_csv(src_csv)
    idx = {}
    for sub in sc.subreddit.unique():
        f = f"{sub}_2018_features_tfidf_256.csv"
        d = pd.read_csv(Path(ext_dir) / f, usecols=["subreddit", "post"]).dropna(subset=["post"])
        t = d.post.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().map(rx.normalize)
        idx[sub] = (f, {n: i for n, i in zip(t[::-1], d.index[::-1])})  # first occurrence wins
    norm = sc.Text.map(rx.normalize)
    sc["source_file"] = [idx[s][0] for s in sc.subreddit]
    sc["source_row"] = [idx[s][1].get(n, -1) for s, n in zip(sc.subreddit, norm)]
    assert (sc.source_row >= 0).all(), "external post not found in the source files"
    sc.drop(columns=["Text"]).to_csv(dst_csv, index=False)


def main():
    rx.DATA = HERE.parent / "data" / "multi_disorders"
    if REL.exists():
        shutil.rmtree(REL)
    (REL / "code").mkdir(parents=True)
    for f in ["run_experiments.py", "check_invariants.py", "ext_test.py", "audit_r3.py", "seeds_r3.py", "queue_r3.sh", "make_release.py",
              "rebuild_anxiety_subset.py", "ext_same75.py", "queue_anx_rebuilt.sh"]:
        shutil.copy(HERE / f, REL / "code" / f)
    shutil.copy(HERE / "fill_numbers.py", REL / "code" / "fill_numbers_r2.py")
    # subsets and splits for the three negative arms (deterministic, seed 42)
    from sklearn.model_selection import StratifiedKFold, train_test_split
    for arm in ["mixed", "twitter", "reddit"]:
        for key in rx.SUBSETS:
            df, info = rx.load_subset(key, arm, 42)
            (REL / "subsets" / arm).mkdir(parents=True, exist_ok=True); (REL / "splits" / arm).mkdir(parents=True, exist_ok=True)
            df.rename(columns={"Text": "text", "n": "text_normalized"}).to_csv(REL / "subsets" / arm / f"{key}.csv", index=False)
            rows = []
            cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            for k, (tv_idx, te_idx) in enumerate(cv.split(df.Text, df.y), 1):
                tv = df.iloc[tv_idx]; te = df.iloc[te_idx]
                tr, va = train_test_split(tv, test_size=0.2, random_state=42 + k, stratify=tv.y)
                for role, fr in (("train", tr), ("val", va), ("test", te)):
                    rows += [dict(fold=k, row_id=int(r), role=role) for r in fr.row_id]
            pd.DataFrame(rows).to_csv(REL / "splits" / arm / f"{key}.csv", index=False)
    # round-2 runs (from the stored archive) and round-3 runs
    z = zipfile.ZipFile(R2ZIP)
    names = set(z.namelist())
    for tag in R2_TAGS:
        for k in KINDS:
            n = f"out/{tag}/{k}"
            if n in names:
                (REL / "results" / tag).mkdir(parents=True, exist_ok=True)
                (REL / "results" / tag / k).write_bytes(z.read(n))
    for f in ["dup_audit.json", "lexical_baseline.json"]:
        (REL / "results" / "round2_superseded").mkdir(parents=True, exist_ok=True)
        (REL / "results" / "round2_superseded" / f).write_bytes(z.read(f))
    for tag in R3_TAGS:
        src = HERE / "out" / tag
        if not src.exists():
            continue
        (REL / "results" / tag).mkdir(parents=True, exist_ok=True)
        for k in KINDS:
            if (src / k).exists():
                shutil.copy(src / k, REL / "results" / tag / k)
    ext = HERE / "out" / "external"
    if ext.exists():
        (REL / "results" / "external").mkdir(parents=True, exist_ok=True)
        for f in ["external_metrics.csv", "external_metrics_same75.csv", "cost.csv", "external_sample.json", "log.txt"]:
            if (ext / f).exists():
                shutil.copy(ext / f, REL / "results" / "external" / f)
        strip_external_texts(ext / "external_scores.csv", REL / "results" / "external" / "external_scores.csv", HERE.parent / "ext")
    # Anxiety subset rebuilt with the intended sampling rule (after review, 24.11.2026)
    rbd = HERE.parent / "data" / "multi_disorders_anxiety_rebuilt"
    if rbd.exists():
        rx.DATA = rbd
        df, info = rx.load_subset("anxiety", "mixed", 42)
        (REL / "subsets" / "mixed_anxiety_rebuilt").mkdir(parents=True, exist_ok=True)
        df.rename(columns={"Text": "text", "n": "text_normalized"}).to_csv(REL / "subsets" / "mixed_anxiety_rebuilt" / "anxiety.csv", index=False)
        rx.DATA = HERE.parent / "data" / "multi_disorders"
    for tag in ["anx_rebuilt_111"]:
        src = HERE / "out" / tag
        if src.exists():
            (REL / "results" / tag).mkdir(parents=True, exist_ok=True)
            for k in KINDS:
                if (src / k).exists():
                    shutil.copy(src / k, REL / "results" / tag / k)
    erb = HERE / "out" / "external_anx_rebuilt"
    if erb.exists():
        (REL / "results" / "external_anx_rebuilt").mkdir(parents=True, exist_ok=True)
        for f in ["external_metrics.csv", "external_metrics_same75.csv", "log.txt"]:
            if (erb / f).exists():
                shutil.copy(erb / f, REL / "results" / "external_anx_rebuilt" / f)
        strip_external_texts(erb / "external_scores.csv", REL / "results" / "external_anx_rebuilt" / "external_scores.csv", HERE.parent / "ext")
    for f in ["invariants_r3.json", "invariants_anxiety_rebuilt.json", "audit_r3.json", "audit_anxiety_rebuilt.json", "seeds_r3.json"]:
        if (HERE / "out" / f).exists():
            shutil.copy(HERE / "out" / f, REL / "results" / f)
    shutil.copy(HERE / "RELEASE_README.md", REL / "README.md")
    import torch, transformers, sklearn, numpy, scipy
    (REL / "requirements.txt").write_text(
        f"python==3.12.0\ntorch=={torch.__version__.split('+')[0]}  # CPU build (+cpu); GPU ablations of round 2: torch 2.0.1+cu117\n"
        f"transformers=={transformers.__version__}\nscikit-learn=={sklearn.__version__}\nnumpy=={numpy.__version__}\n"
        f"pandas=={pd.__version__}\nscipy=={scipy.__version__}\nshap  # only for the SHAP run (npd_shap)\n", encoding="utf8")
    n = sum(1 for _ in REL.rglob("*") if _.is_file())
    print("release files:", n, "->", REL)


if __name__ == "__main__":
    main()
