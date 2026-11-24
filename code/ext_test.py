#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 3: cross-platform external test (Twitter-trained detectors -> Reddit posts).

External data: Reddit Mental Health Dataset (Low et al., 2020; Zenodo 3941387, ODC-PDDL),
2018 files of r/anxiety, r/depression and four non-mental-health control subreddits
(r/teaching, r/divorce, r/guns, r/fitness).

For each target (anxiety, depression) and each training-negative arm (mixed = reported
configuration, twitter = same-source negatives):
  * the cleaned subset is split once 80/20 (stratified, seed 42); the detector is fine-tuned
    with the reported settings (composite loss, 4 epochs, seed 42) and the Youden threshold
    is chosen on the 20% split, exactly as inside each cross-validation fold;
  * the detector is applied unchanged to an external sample:
      positives  = posts of the target subreddit,
      negatives  = (a) control posts, (b) posts of the other mental-health subreddit;
  * reported: AUC, weighted F1 / sensitivity / specificity at the transferred threshold,
    and F1 after re-selecting the threshold on 25% of the external sample (evaluated on
    the remaining 75%), which separates ranking loss from threshold (calibration) shift;
  * a TF-IDF + logistic-regression baseline is trained on the same 80% split and evaluated
    on the same external sample; training and inference times of both models are logged.
External posts that duplicate (exactly or TF-IDF cosine >= 0.9) any text of the training
corpora are removed before sampling.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as rx  # noqa: E402

CONTROLS = ["teaching", "divorce", "guns", "fitness"]


def load_external(ext_dir, n_per, seed, train_norm):
    from sklearn.feature_extraction.text import TfidfVectorizer
    frames = {}
    for sub in ["anxiety", "depression"] + CONTROLS:
        d = pd.read_csv(Path(ext_dir) / f"{sub}_2018_features_tfidf_256.csv", usecols=["subreddit", "post"])
        d = d.dropna(subset=["post"])
        d["Text"] = d.post.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        d = d[~d.Text.str.lower().isin(["[removed]", "[deleted]", ""])]
        d["n"] = d.Text.map(rx.normalize)
        d = d[d.n.str.split().str.len() >= 5].drop_duplicates("n")
        frames[sub] = d
    rng = np.random.default_rng(seed)
    # candidate pool: 2x the needed number per subreddit, in a seeded random order
    cand = []
    for sub, d in frames.items():
        need = n_per if sub in ("anxiety", "depression") else n_per // len(CONTROLS)
        cand.append(d.iloc[rng.permutation(len(d))[:2 * need]])
    allx = pd.concat(cand)
    n_before = len(allx)
    exact = allx.n.isin(train_norm).values
    # near-duplicates against the training corpora (TF-IDF fitted on training texts), in chunks
    tn = sorted(train_norm)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(tn)
    Xt = vec.transform(tn).T.tocsr(); Xe = vec.transform(allx.n.tolist())
    sim = np.concatenate([(Xe[i:i + 200] @ Xt).max(axis=1).toarray().ravel() for i in range(0, Xe.shape[0], 200)])
    keep = (~exact) & (sim < 0.9)
    dedup = dict(candidates=int(n_before), exact_overlap=int(exact.sum()), near_dup_ge_0_9=int(((sim >= 0.9) & ~exact).sum()),
                 max_similarity=float(sim.max()), posts_available={k: int(len(v)) for k, v in frames.items()})
    allx = allx[keep]
    out = {}
    for sub in ["anxiety", "depression"]:
        out[sub] = allx[allx.subreddit == sub].iloc[:n_per]
    ctrl = []
    for sub in CONTROLS:
        ctrl.append(allx[allx.subreddit == sub].iloc[:n_per // len(CONTROLS)])
    out["control"] = pd.concat(ctrl)
    dedup.update({f"n_{k}": int(len(v)) for k, v in out.items()})
    return out, dedup


def evaluate(y, s, tau, seed):
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    y = np.asarray(y); s = np.asarray(s); p = (s >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    res = dict(n=int(len(y)), auc=float(roc_auc_score(y, s)), f1=float(f1_score(y, p, average="weighted")),
               sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), tau=float(tau))
    # threshold re-selected on 25% of the external sample, evaluated on the other 75%
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    cal, ev = train_test_split(idx, test_size=0.75, random_state=seed, stratify=y)
    t2 = rx.youden(y[cal], s[cal])
    res.update(f1_recal=float(f1_score(y[ev], (s[ev] >= t2).astype(int), average="weighted")), tau_recal=float(t2),
               f1_transfer_same75=float(f1_score(y[ev], (s[ev] >= tau).astype(int), average="weighted")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--ext-dir", required=True)
    ap.add_argument("--out", default=str(HERE / "out" / "external"))
    ap.add_argument("--n-per", type=int, default=400); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", nargs="*", default=["anxiety", "depression"])
    ap.add_argument("--arms", nargs="*", default=["mixed", "twitter"])
    a = ap.parse_args()
    rx.DATA = Path(a.data_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf8")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()
    import torch, transformers, sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    log(f"=== external test torch={torch.__version__} transformers={transformers.__version__} sklearn={sklearn.__version__} threads={torch.get_num_threads()} {json.dumps(vars(a))}")
    tw = pd.read_csv(rx.DATA / "train.csv"); rd = pd.read_csv(rx.DATA / "depression_dataset_reddit_cleaned.csv")
    train_norm = set(tw.Text.map(rx.normalize)) | set(rd.clean_text.map(rx.normalize))
    ext, dedup = load_external(a.ext_dir, a.n_per, a.seed, train_norm)
    log(f"external sample: {json.dumps(dedup)}")
    json.dump(dedup, open(out / "external_sample.json", "w"), indent=1)
    targs = argparse.Namespace(model_name="distilbert/distilbert-base-uncased", dropout=0.3, ce=1.0, focal=1.0, dice=1.0, focal_alpha=0.25,
                               focal_gamma=2.0, max_length=128, weight_decay=0.1, lr=2e-5, batch_size=4, epochs=4, warmup_steps=24, shap=False)
    for target in a.targets:
        other = "depression" if target == "anxiety" else "anxiety"
        pos = ext[target].assign(y=1, group=target)
        ext_frame = pd.concat([pos, ext["control"].assign(y=0, group="control"), ext[other].assign(y=0, group=other)]).reset_index(drop=True)
        ext_frame["row_id"] = np.arange(len(ext_frame)); ext_frame["src"] = "reddit_ext:" + ext_frame.subreddit
        for arm in a.arms:
            ck = out / "units" / f"{target}_{arm}.pkl"; ck.parent.mkdir(exist_ok=True)
            if ck.exists():
                log(f">>> {target} / {arm}: checkpoint found, skipped"); continue
            df, info = rx.load_subset(target, arm, a.seed)
            train, val = train_test_split(df, test_size=0.2, random_state=a.seed, stratify=df.y)
            train = train.reset_index(drop=True); val = val.reset_index(drop=True)
            log(f">>> {target} / {arm}: {json.dumps(info)} n_train={len(train)} n_val={len(val)} n_ext={len(ext_frame)}")
            vs, es, tau, lh, _, tt = rx.train_eval_fold(train, val, ext_frame, targs, a.seed, "cpu", log)
            # TF-IDF + LR on the same split
            t0 = time.time()
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            Xtr = vec.fit_transform(train.n); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, train.y)
            t_fit = time.time() - t0
            t0 = time.time(); bv = lr.predict_proba(vec.transform(val.n))[:, 1]; be = lr.predict_proba(vec.transform(ext_frame.n))[:, 1]; t_pred = time.time() - t0
            btau = rx.youden(val.y.values, bv)
            unit_rows = []
            for neg in ["control", other, "both"]:
                m = ext_frame.group.isin([target, neg] if neg != "both" else [target, "control", other]).values
                r1 = evaluate(ext_frame.y.values[m], es[m], tau, a.seed)
                r2 = evaluate(ext_frame.y.values[m], be[m], btau, a.seed)
                unit_rows.append(dict(target=target, arm=arm, negatives=neg, model="distilbert", **r1))
                unit_rows.append(dict(target=target, arm=arm, negatives=neg, model="tfidf_lr", **r2))
                log(f"   ext {target}/{arm} vs {neg}: DistilBERT AUC={r1['auc']:.3f} F1={r1['f1']:.3f} (sens {r1['sens']:.2f} spec {r1['spec']:.2f}) F1recal={r1['f1_recal']:.3f} | TF-IDF AUC={r2['auc']:.3f} F1={r2['f1']:.3f} F1recal={r2['f1_recal']:.3f}")
            log(f"   cost {target}/{arm}: DistilBERT train {tt['t_train_s']:.0f}s on {len(train)} texts, inference {tt['t_infer_s']:.1f}s on {tt['n_infer']} texts | TF-IDF fit {t_fit:.2f}s, predict {t_pred:.3f}s on {len(val)+len(ext_frame)} texts")
            cost = dict(target=target, arm=arm, n_train=len(train), bert_train_s=tt["t_train_s"], bert_infer_s=tt["t_infer_s"], bert_n_infer=tt["n_infer"],
                        tfidf_fit_s=t_fit, tfidf_predict_s=t_pred, tfidf_n_predict=len(val) + len(ext_frame), tfidf_n_features=int(Xtr.shape[1]))
            scores = ext_frame.assign(target=target, arm=arm, bert_score=es, bert_tau=tau, tfidf_score=be, tfidf_tau=btau)[
                ["target", "arm", "row_id", "group", "subreddit", "y", "bert_score", "bert_tau", "tfidf_score", "tfidf_tau", "Text"]]
            tmp = ck.with_suffix(".tmp"); pd.to_pickle(dict(rows=unit_rows, cost=cost, scores=scores), tmp); os.replace(tmp, ck)
            units = [pd.read_pickle(u) for u in sorted((out / "units").glob("*.pkl"))]
            pd.DataFrame([r for u in units for r in u["rows"]]).to_csv(out / "external_metrics.csv", index=False)
            pd.DataFrame([u["cost"] for u in units]).to_csv(out / "cost.csv", index=False)
            pd.concat([u["scores"] for u in units]).to_csv(out / "external_scores.csv", index=False)
    log("DONE external")


if __name__ == "__main__":
    main()
