#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariants of subset construction (round 3, after the Anxiety sampling error).

The Anxiety subset of the authors' Algorithm 1 (method/GenerateBalancedDataset.py, copied from the NPD script)
listed 'Anxiety Disorder' instead of 'Narcissistic Disorder' among its non-target classes, so 30 Anxiety texts were
sampled as negatives; under keep-first de-duplication 13 of them stayed in the data labelled as negatives.
Nothing in the old pipeline checked for this. This script makes the rules explicit and checks them for every
category and every negative-sampling arm, both on the authors' subset files as delivered (to show what the checks
catch) and on the final subsets used for all experiments (which must pass):

  I1  no normalized text occurs twice in a subset
  I2  no normalized text occurs in both classes
  I3  no negative carries the target label anywhere in the source corpus (Twitter train.csv)
  I4  every positive carries the target label in the source corpus
  I5  every negative comes from a non-target source (another Twitter category or the Reddit no-depression pool)
  I6  in every fold, the training, validation, and test texts are disjoint and together cover the subset
  I7  the classes are balanced (positive share 0.45-0.55)
Reported, not asserted: the source composition of the negatives against Algorithm 1 (four other Twitter categories
plus Reddit), so that a missing share (NPD in the cleaned Anxiety subset) stays visible.

Usage: python check_invariants.py [--data-dir DIR]   -> out/invariants_r3.json; exit code 1 if a final subset fails.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as R  # noqa: E402  (same normalisation and subset builder as the experiments)

INV = ["I1_unique", "I2_no_cross_class", "I3_neg_not_target", "I4_pos_is_target", "I5_neg_source_ok", "I6_folds_disjoint", "I7_balanced"]


def source_labels(tw):
    return tw.groupby("n").label.agg(lambda s: set(s)).to_dict()


def check_frame(dd, pos_label, labsets, rdset, folds=True, seed=42, n_folds=4, inner_val=0.2):
    """dd: columns n, y (1 = target). Returns {invariant: number of violations} and the negative composition."""
    v = {}
    v["I1_unique"] = int(dd.n.duplicated().sum())
    cls = dd.groupby("n").y.nunique()
    v["I2_no_cross_class"] = int((cls > 1).sum())
    neg, pos = dd[dd.y == 0], dd[dd.y == 1]
    v["I3_neg_not_target"] = int(neg.n.map(lambda t: pos_label in labsets.get(t, set())).sum())
    v["I4_pos_is_target"] = int((~pos.n.map(lambda t: pos_label in labsets.get(t, set()))).sum())

    def src(t):
        if t in labsets:
            return "twitter:" + "/".join(sorted(labsets[t]))
        return "reddit" if t in rdset else "unknown"
    nsrc = neg.n.map(src)
    v["I5_neg_source_ok"] = int(sum(1 for s in nsrc if s == "unknown" or (s.startswith("twitter:") and pos_label in s)))
    if folds:
        bad = 0
        d = dd.reset_index(drop=True)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (tv_idx, te_idx) in enumerate(cv.split(d.n, d.y), 1):
            tv, te = d.iloc[tv_idx], d.iloc[te_idx]
            tr, va = train_test_split(tv, test_size=inner_val, random_state=seed + k, stratify=tv.y)
            a, b, c = set(tr.n), set(va.n), set(te.n)
            bad += len(a & b) + len(a & c) + len(b & c) + (len(a | b | c) != d.n.nunique())
        v["I6_folds_disjoint"] = bad
    share = pos.shape[0] / max(len(dd), 1)
    v["I7_balanced"] = int(not 0.45 <= share <= 0.55)
    return v, nsrc.value_counts().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE.parent / "data" / "multi_disorders"))
    ap.add_argument("--out", default=str(HERE / "out" / "invariants_r3.json"))
    a = ap.parse_args()
    R.DATA = Path(a.data_dir)
    tw = pd.read_csv(R.DATA / "train.csv"); tw["n"] = tw.Text.map(R.normalize)
    labsets = source_labels(tw)
    rd = pd.read_csv(R.DATA / "depression_dataset_reddit_cleaned.csv")
    rdset = set(rd[rd.is_depression == 0].clean_text.map(R.normalize))
    others = sorted(tw.label.unique())
    res, failed = {"invariants": INV, "raw_author_files": {}, "final_subsets": {}}, []
    print("%-11s %-8s %5s  %s" % ("category", "arm", "n", "  ".join(i.split("_")[0] for i in INV)))
    for key, (fname, pos_label, pretty) in R.SUBSETS.items():
        # authors' subset file as delivered (normalized text, original labels, all rows)
        raw = pd.read_csv(R.DATA / fname); raw["n"] = raw.Text.map(R.normalize); raw["y"] = (raw.label == pos_label).astype(int)
        v, comp = check_frame(raw, pos_label, labsets, rdset, folds=False)
        # what keep-first de-duplication (the earlier protocol) would have kept: target-category texts labelled negative
        kf = raw.drop_duplicates("n", keep="first")
        kept_neg = int(kf[(kf.y == 0) & kf.n.map(lambda t: pos_label in labsets.get(t, set()))].shape[0])
        res["raw_author_files"][key] = dict(n=len(raw), violations=v, negative_sources=comp,
                                            target_texts_kept_as_negatives_by_keep_first_dedup=kept_neg)
        print("%-11s %-8s %5d  %s   <- authors' file as delivered; keep-first de-dup keeps %d target texts as negatives"
              % (key, "raw", len(raw), "  ".join("%2d" % v.get(i, 0) for i in INV), kept_neg))
        for arm in ("mixed", "twitter", "reddit"):
            dd, info = R.load_subset(key, arm, 42)
            v, comp = check_frame(dd, pos_label, labsets, rdset)
            expected = ["twitter:" + l for l in others if l != pos_label] + (["reddit"] if arm != "twitter" else [])
            if arm == "reddit":
                expected = ["reddit"]
            missing = [e for e in expected if not any(p == e or p.startswith(e) for p in comp)]
            res["final_subsets"]["%s/%s" % (key, arm)] = dict(info=info, violations=v, negative_sources=comp,
                                                              missing_negative_shares=missing)
            ok = all(x == 0 for x in v.values())
            if not ok:
                failed.append("%s/%s" % (key, arm))
            print("%-11s %-8s %5d  %s   %s%s" % (key, arm, len(dd), "  ".join("%2d" % v.get(i, 0) for i in INV),
                                                "ok" if ok else "FAIL", ("  missing share: " + ", ".join(missing)) if missing else ""))
    res["all_final_subsets_pass"] = not failed
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("final subsets:", "ALL PASS" if not failed else "FAILED: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
