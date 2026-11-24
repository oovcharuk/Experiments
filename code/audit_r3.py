#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 3: leakage / lexical-artifact audit and TF-IDF + LR baseline on the FINAL subsets.

Round 2 computed the bag-of-words baseline and the audit on the de-duplicated subsets before the
removal of label-conflicting texts (n = 297, 263, 404, 309, 221). Here everything is recomputed on
the final subsets (n = 292, 233, 396, 309, 221) with exactly the fold assignment, inner splits and
threshold protocol of the transformer (StratifiedKFold(4, shuffle, seed 42); inner 80/20 split with
seed 42+k; Youden threshold on the inner split; pooled out-of-fold weighted F1).
Also: length-only AUC, keyword-giveaway rates, near-duplicate pairs, timing of the baseline, and
TF-IDF centroid similarity between the five source categories (for the anxiety discussion).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as rx  # noqa: E402

STEMS = {"anger": ["anger"], "anxiety": ["anxi", "worr"], "depression": ["depress"], "npd": ["narciss", "grandi"], "panic": ["panic"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--out", default=str(HERE / "out" / "audit_r3.json"))
    a = ap.parse_args(); rx.DATA = Path(a.data_dir)
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.metrics.pairwise import cosine_similarity
    res = {}
    for key in rx.SUBSETS:
        df, info = rx.load_subset(key, "mixed", 42)
        y = df.y.values; n_words = df.n.str.split().str.len().values
        pos, neg = df[df.y == 1].n, df[df.y == 0].n
        kw = {"pos": float(pos.map(lambda t: any(s in t for s in STEMS[key])).mean()),
              "neg": float(neg.map(lambda t: any(s in t for s in STEMS[key])).mean())}
        X = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(df.n); S = cosine_similarity(X); np.fill_diagonal(S, 0)
        iu = np.triu_indices(len(df), 1); nd = int((S[iu] >= 0.9).sum())
        # TF-IDF + LR with the transformer's folds and thresholding
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        oof_s, oof_p, fit_t, pred_t, nfeat = np.zeros(len(df)), np.zeros(len(df), int), [], [], []
        for k, (tv_idx, te_idx) in enumerate(cv.split(df.Text, df.y), 1):
            tv = df.iloc[tv_idx]; te = df.iloc[te_idx]
            tr, va = train_test_split(tv, test_size=0.2, random_state=42 + k, stratify=tv.y)
            t0 = time.perf_counter()
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); Xtr = vec.fit_transform(tr.n)
            lr = LogisticRegression(max_iter=2000).fit(Xtr, tr.y); fit_t.append(time.perf_counter() - t0); nfeat.append(Xtr.shape[1])
            t0 = time.perf_counter(); sv = lr.predict_proba(vec.transform(va.n))[:, 1]; st = lr.predict_proba(vec.transform(te.n))[:, 1]
            pred_t.append((time.perf_counter() - t0) / (len(va) + len(te)))
            tau = rx.youden(va.y.values, sv); oof_s[te_idx] = st; oof_p[te_idx] = (st >= tau).astype(int)
        res[key] = dict(n=len(df), n_pos=int(y.sum()), length_only_auc=float(roc_auc_score(y, n_words)),
                        mean_words_pos=float(n_words[y == 1].mean()), mean_words_neg=float(n_words[y == 0].mean()),
                        keyword_stems=STEMS[key], keyword_rate=kw, near_dup_pairs_ge_0_9=nd,
                        tfidf_lr_oof_f1=float(f1_score(y, oof_p, average="weighted")), tfidf_lr_oof_auc=float(roc_auc_score(y, oof_s)),
                        tfidf_fit_s_per_fold=float(np.mean(fit_t)), tfidf_predict_ms_per_text=float(1000 * np.mean(pred_t)),
                        tfidf_n_features=int(np.mean(nfeat)))
        print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[key].items()}))
    # TF-IDF centroid similarity between the five source categories (distinct single-label texts)
    tw = pd.read_csv(rx.DATA / "train.csv"); tw["n"] = tw.Text.map(rx.normalize)
    lab = tw.groupby("n").label.agg(lambda s: set(s)); single = lab[lab.map(len) == 1].map(lambda s: next(iter(s)))
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); X = vec.fit_transform(single.index)
    cats = sorted(set(single)); C = np.vstack([np.asarray(X[(single == c).values].mean(axis=0)) for c in cats]); C = cosine_similarity(C)
    res["centroid_similarity"] = {f"{cats[i]} | {cats[j]}": float(C[i, j]) for i in range(len(cats)) for j in range(i + 1, len(cats))}
    for k, v in sorted(res["centroid_similarity"].items(), key=lambda x: -x[1]): print(f"  {v:.3f}  {k}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
