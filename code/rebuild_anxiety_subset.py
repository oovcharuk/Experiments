#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the Anxiety subset with the sampling rule that Algorithm 1 intended (round 3, after review).

The authors' AnxietyDisorder.csv took one of its five equal non-target shares (30 texts) from the Anxiety category
itself instead of from NPD (script copied from the NPD one). The conflict-free subset used for the main results
removes those 30 texts from both classes (115 positives, no NPD negatives). This script instead repairs the file:
the 30 Anxiety texts in the non-target class are replaced by 30 NPD texts drawn at random (seed 42) from the source
corpus, excluding any text that carries the Anxiety label or already occurs in the subset; every other row is kept.
The rebuilt file then goes through the normal pipeline (exact de-duplication, label-conflict removal) unchanged.

Usage: python rebuild_anxiety_subset.py [--src DIR] [--dst DIR]   (dst: a copy of src with AnxietyDisorder.csv rebuilt)
"""
import argparse, shutil, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=str(HERE.parent / 'data' / 'multi_disorders'))
ap.add_argument('--dst', default=str(HERE.parent / 'data' / 'multi_disorders_anxiety_rebuilt'))
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()
src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.name != 'AnxietyDisorder.csv':
        shutil.copy2(f, dst / f.name)

tw = pd.read_csv(src / 'train.csv'); tw['n'] = tw.Text.map(R.normalize)
labsets = tw.groupby('n').label.agg(lambda s: set(s)).to_dict()
df = pd.read_csv(src / 'AnxietyDisorder.csv'); df['n'] = df.Text.map(R.normalize)
POS = 'Anxiety Disorder'
bad = (df.label != POS) & df.n.map(lambda t: POS in labsets.get(t, set()))
print('non-target rows that carry the Anxiety label:', int(bad.sum()))
pool = tw[(tw.label == 'Narcissistic Disorder') & ~tw.n.map(lambda t: POS in labsets[t]) & ~tw.n.isin(set(df.n))]
pool = pool.drop_duplicates('n')
take = pool.sample(n=int(bad.sum()), random_state=a.seed)
new = pd.concat([df[~bad][['Text', 'label']], take[['Text']].assign(label='non-target')], ignore_index=True)
new.to_csv(dst / 'AnxietyDisorder.csv', index=False)
print('rebuilt AnxietyDisorder.csv:', new.label.value_counts().to_dict(), '->', dst / 'AnxietyDisorder.csv')
R.DATA = dst
dd, info = R.load_subset('anxiety', 'mixed', a.seed)
print('after the normal pipeline:', info)
print('negative sources:', dd[dd.y == 0].src.value_counts().to_dict())
