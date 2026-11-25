#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External test, round 3 after review: every threshold-dependent metric on the SAME 75% evaluation part.

ext_test.py reported F1/sensitivity/specificity at the transferred threshold on the whole external sample and F1 at
the re-selected threshold on the 75% evaluation part (25% used to re-select it). This script recomputes, from the
stored scores, the transferred-threshold metrics on exactly that 75% part (same split: train_test_split of the row
positions, test_size=0.75, seed 42, stratified by y), so that both F1 values refer to the same texts. AUC stays on
the whole sample (no threshold, no selection). It first reproduces the stored f1_recal and f1_transfer_same75.

Usage: python ext_same75.py [--dir out/external]  -> <dir>/external_metrics_same75.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as rx  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=str(HERE / 'out' / 'external')); ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()
D = Path(a.dir)
sc = pd.read_csv(D / 'external_scores.csv')
m0 = pd.read_csv(D / 'external_metrics.csv')
rows = []
for (t, arm), g in sc.groupby(['target', 'arm'], sort=False):
    other = 'depression' if t == 'anxiety' else 'anxiety'
    for neg in ('control', other):
        s = g[g.group.isin([t, neg])]
        y = s.y.values
        for model, col, tc in (('distilbert', 'bert_score', 'bert_tau'), ('tfidf_lr', 'tfidf_score', 'tfidf_tau')):
            x = s[col].values; tau = float(s[tc].iloc[0])
            cal, ev = train_test_split(np.arange(len(y)), test_size=0.75, random_state=a.seed, stratify=y)
            t2 = rx.youden(y[cal], x[cal])
            p = (x[ev] >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[ev], p, labels=[0, 1]).ravel()
            r = dict(target=t, arm=arm, negatives=neg, model=model, n_eval=int(len(ev)),
                     f1_transfer=float(f1_score(y[ev], p, average='weighted')), sens_transfer=tp / max(tp + fn, 1),
                     spec_transfer=tn / max(tn + fp, 1), f1_recal=float(f1_score(y[ev], (x[ev] >= t2).astype(int), average='weighted')))
            old = m0[(m0.target == t) & (m0.arm == arm) & (m0.negatives == neg) & (m0.model == model)].iloc[0]
            assert abs(old.f1_recal - r['f1_recal']) < 1e-9 and abs(old.f1_transfer_same75 - r['f1_transfer']) < 1e-9, (t, arm, neg, model)
            r.update(auc_all=float(old.auc), f1_transfer_all=float(old.f1), sens_all=float(old.sens), spec_all=float(old.spec))
            rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(D / 'external_metrics_same75.csv', index=False)
print(out.round(3).to_string(index=False))
