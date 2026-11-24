#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resubmission experiments (IEEE Access, Access-2026-33542 / Access-2026-50336).

Protocol (identical to the authors' corrected 4-fold protocol, with one addition):
  normalise -> remove exact duplicates -> remove label-conflicting texts (texts that occur
  under both the target and the non-target label inside a subset) -> StratifiedKFold(4, seed 42)
  -> inner 80/20 stratified split -> fine-tune DistilBERT with composite loss
  -> Youden tau on the inner validation split -> single evaluation on the fold test set
  -> pooled out-of-fold metrics + 2000-resample bootstrap CI.

Arms / ablations selected via CLI:
  --negatives mixed|twitter|reddit   negative-sampling arm (mixed = authors' subset files)
  --ce/--focal/--dice weights        composite-loss ablation
  --train-frac f                     learning-curve run (subsample D_train only)
  --shap                             SHAP token attributions on each fold test set
  --train-seed s                     training seed (classifier-head init, dropout, batch order);
                                     fold assignment and inner split stay fixed by --seed
                                     (round 3: separates run-to-run from fold-to-fold variability)
  --data-dir d                       folder with the authors' CSV files (datasets/multi_disorders)
  --max-folds k                      stop after k folds (quick reproduction check)
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault("TRANSFORMERS_NO_TF", "1"); os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "multi_disorders"
SUBSETS = {
    "anger": ("AngerIntermittentExplosiveDisorder.csv", "Anger/ Intermittent Explosive Disorder", "Anger/IED (d1)"),
    "anxiety": ("AnxietyDisorder.csv", "Anxiety Disorder", "Anxiety (d2)"),
    "depression": ("Depression.csv", "Depression", "Depression (d3)"),
    "npd": ("NarcissisticDisorder.csv", "Narcissistic Disorder", "NPD (d4)"),
    "panic": ("PanicDisorder.csv", "Panic Disorder", "Panic (d5)"),
}


def normalize(text):
    s = str(text).lower().strip()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[#@]\w+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_subset(key, negatives, seed):
    """Return de-duplicated, conflict-free frame with columns Text, n, y, src."""
    fname, pos_label, _ = SUBSETS[key]
    tw = pd.read_csv(DATA / "train.csv"); tw["n"] = tw.Text.map(normalize)
    labsets = tw.groupby("n").label.agg(lambda s: set(s)).to_dict()
    rd = pd.read_csv(DATA / "depression_dataset_reddit_cleaned.csv"); rd["n"] = rd.clean_text.map(normalize)
    rd0 = rd[rd.is_depression == 0].drop_duplicates("n").reset_index(drop=True)
    info = {}
    if negatives == "mixed":
        df = pd.read_csv(DATA / fname); df["n"] = df.Text.map(normalize)
        raw = len(df)
        nlab = df.groupby("n").label.nunique(); conflicts = set(nlab[nlab > 1].index)
        dd = df.drop_duplicates("n", keep="first")
        exact_removed = raw - len(dd)
        dd = dd[~dd.n.isin(conflicts)].reset_index(drop=True)
        dd["y"] = (dd.label == pos_label).astype(int)
        rdset = set(rd0.n)
        dd["src"] = ["twitter:" + "/".join(sorted(labsets[t])) if t in labsets else ("reddit" if t in rdset else "unknown") for t in dd.n]
        info = dict(raw=raw, exact_removed=exact_removed, conflict_removed=len(conflicts))
    else:
        pos = tw[tw.n.map(lambda t: pos_label in labsets[t])].drop_duplicates("n")[["Text", "n"]].copy()
        pos["y"] = 1; pos["src"] = "twitter:" + pos_label
        rng = np.random.default_rng(seed)
        n_pos = len(pos)
        if negatives == "twitter":
            others = [l for l in tw.label.unique() if l != pos_label]
            per = int(math.ceil(n_pos / len(others)))
            parts = []
            for l in others:
                pool = tw[(tw.label == l) & tw.n.map(lambda t: pos_label not in labsets[t])].drop_duplicates("n")
                take = pool.sample(n=min(per, len(pool)), random_state=int(rng.integers(1e9)))[["Text", "n"]].copy()
                take["src"] = "twitter:" + l; parts.append(take)
            neg = pd.concat(parts).drop_duplicates("n")
        elif negatives == "reddit":
            neg = rd0.sample(n=n_pos, random_state=int(rng.integers(1e9)))[["clean_text", "n"]].rename(columns={"clean_text": "Text"})
            neg["src"] = "reddit"
        else:
            raise ValueError(negatives)
        neg = neg[~neg.n.isin(set(pos.n))].copy(); neg["y"] = 0
        dd = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        info = dict(raw=len(dd), exact_removed=0, conflict_removed=0)
    dd["row_id"] = np.arange(len(dd))
    info.update(n=len(dd), n_pos=int(dd.y.sum()), n_neg=int((dd.y == 0).sum()),
                neg_reddit=int((dd.src == "reddit").sum()))
    return dd[["row_id", "Text", "n", "y", "src"]], info


def youden(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    i = int(np.argmax(j))
    t = float(thr[i])
    return 1.0 if not np.isfinite(t) else t


def metrics_at(y, s, tau):
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
    y = np.asarray(y); s = np.asarray(s); p = (s >= tau).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    f1pos = precision_recall_fscore_support(y, p, average="binary", zero_division=0)[2]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    try: auc = roc_auc_score(y, s)
    except ValueError: auc = float("nan")
    return dict(f1=float(f1), f1_pos=float(f1pos), auc=float(auc), acc=float(accuracy_score(y, p)),
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), ppv=tp / max(tp + fp, 1),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def bootstrap_ci(y, p, n_boot, seed):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); n = len(y); st = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        st.append(f1_score(y[idx], p[idx], average="weighted", zero_division=0))
    lo, hi = np.percentile(st, [2.5, 97.5]); return float(lo), float(hi)


def composite_loss(logits, labels, w):
    import torch, torch.nn.functional as F
    ce = F.cross_entropy(logits, labels)
    probs = torch.softmax(logits, -1)
    pt = probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp(1e-7, 1 - 1e-7)
    focal = (-w["alpha"] * (1 - pt) ** w["gamma"] * torch.log(pt)).mean()
    pp = probs[:, 1]; yp = labels.float(); sm = 1.0
    dice = 1 - (2 * (pp * yp).sum() + sm) / (pp.sum() + yp.sum() + sm)
    return w["ce"] * ce + w["focal"] * focal + w["dice"] * dice, (ce.item(), focal.item(), dice.item())


def train_eval_fold(train, val, test, args, fold_seed, device, log):
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertConfig, get_linear_schedule_with_warmup
    torch.manual_seed(fold_seed); np.random.seed(fold_seed)
    tok = DistilBertTokenizerFast.from_pretrained(args.model_name)
    cfg = DistilBertConfig.from_pretrained(args.model_name, num_labels=2, dropout=args.dropout, attention_dropout=args.dropout)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, config=cfg).to(device)
    w = dict(ce=args.ce, focal=args.focal, dice=args.dice, alpha=args.focal_alpha, gamma=args.focal_gamma)

    def enc(texts):
        return tok(list(texts), truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")

    nd = ["bias", "LayerNorm.weight"]
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in nd)], "weight_decay": args.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in nd)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = math.ceil(len(train) / args.batch_size); total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, total)
    g = torch.Generator().manual_seed(fold_seed)
    texts = train.Text.tolist(); ys = torch.tensor(train.y.values, dtype=torch.long)
    loss_hist = []
    t_train0 = time.time()
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(train), generator=g); tot = np.zeros(4); nb = 0
        for b in range(0, len(train), args.batch_size):
            idx = perm[b:b + args.batch_size]
            batch = enc([texts[i] for i in idx]).to(device); yb = ys[idx].to(device)
            out = model(**batch)
            loss, comps = composite_loss(out.logits, yb, w)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += np.array([loss.item(), *comps]); nb += 1
        m = tot / nb; loss_hist.append(dict(epoch=ep + 1, loss=m[0], ce=m[1], focal=m[2], dice=m[3]))
        log(f"      epoch {ep+1}: loss={m[0]:.4f} (ce {m[1]:.3f} focal {m[2]:.3f} dice {m[3]:.3f})")

    @torch.no_grad()
    def scores(frame):
        model.eval(); out = []
        for b in range(0, len(frame), 32):
            batch = enc(frame.Text.iloc[b:b + 32]).to(device)
            out.append(torch.softmax(model(**batch).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    t_train = time.time() - t_train0
    t_inf0 = time.time(); vs, ts = scores(val), scores(test); t_infer = time.time() - t_inf0
    tau = youden(val.y.values, vs)
    shap_rows = []
    if args.shap:
        shap_rows = run_shap(model, tok, test, ts, device, log)
    import gc
    model.to("cpu"); del model, opt, sched; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return vs, ts, tau, loss_hist, shap_rows, dict(t_train_s=t_train, t_infer_s=t_infer, n_infer=len(val) + len(test))


def run_shap(model, tok, test, ts, device, log):
    """SHAP partition explainer on the fold test set; returns per-token attribution rows."""
    import shap, torch
    def f(x):
        out = []
        for b in range(0, len(x), 32):
            batch = tok(list(x[b:b + 32]), truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out.append(torch.softmax(model(**batch).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)
    masker = shap.maskers.Text(tok, mask_token="[MASK]", collapse_mask_token=True)
    explainer = shap.Explainer(f, masker, silent=True)
    rows = []
    t0 = time.time()
    ex = explainer(test.Text.tolist(), max_evals=300, batch_size=128)
    for i in range(len(test)):
        toks = [str(t).strip() for t in ex.data[i]]; vals = np.asarray(ex.values[i], dtype=float)
        for t, v in zip(toks, vals):
            if t: rows.append(dict(row_id=int(test.row_id.iloc[i]), y=int(test.y.iloc[i]), score=float(ts[i]), token=t, shap=float(v), base=float(ex.base_values[i])))
    log(f"      shap: {len(test)} texts in {time.time()-t0:.0f}s")
    import gc; del ex, explainer, masker; gc.collect()
    return rows


def aggregate(out, args, log):
    """Rebuild all output tables from the per-fold checkpoints (any number of earlier processes)."""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, accuracy_score
    nf = args.max_folds or args.folds
    summary, fms, scs, losses, shaps, splits, infos = [], [], [], [], [], [], {}
    for key in SUBSETS:
        cks = [out / "folds" / f"{key}_f{k}.pkl" for k in range(1, nf + 1)]
        if not all(c.exists() for c in cks): continue
        recs = [pd.read_pickle(c) for c in cks]
        info, pretty = recs[0]["info"], recs[0]["pretty"]; infos[key] = info
        fm = pd.DataFrame([r["fm"] for r in recs]); sf = pd.concat([r["scores"] for r in recs])
        fms.append(fm); scs.append(sf)
        for r in recs: losses += r["loss"]; shaps += r["shap"]; splits += r["splits"]
        y, s, p = sf.y_true.values, sf.score.values, sf.pred.values
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel(); lo, hi = bootstrap_ci(y, p, args.bootstrap, args.seed)
        row = dict(tag=args.tag, disorder=key, pretty=pretty, negatives=args.negatives, ce=args.ce, focal=args.focal, dice=args.dice, train_frac=args.train_frac,
                   train_seed=(args.seed if args.train_seed is None else args.train_seed),
                   n=recs[0]["n"], n_pos=info["n_pos"], n_neg=info["n_neg"], neg_reddit=info["neg_reddit"],
                   oof_f1=float(f1), f1_lo=lo, f1_hi=hi, oof_auc=float(roc_auc_score(y, s)), oof_acc=float(accuracy_score(y, p)),
                   sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / max(tp + fp, 1), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                   f1_fold_mean=float(fm.f1.mean()), f1_fold_std=float(fm.f1.std(ddof=1)), auc_fold_mean=float(fm.auc.mean()), auc_fold_std=float(fm.auc.std(ddof=1)),
                   taus=";".join(f"{t:.4f}" for t in fm.tau))
        summary.append(row)
        log(f"<<< {pretty}: OOF F1={f1:.4f} [{lo:.3f},{hi:.3f}] AUC={row['oof_auc']:.4f} TP={tp} FP={fp} FN={fn} TN={tn} fold F1 {row['f1_fold_mean']:.3f}+-{row['f1_fold_std']:.3f}")
    if not summary: return
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.concat(fms).to_csv(out / "fold_metrics.csv", index=False)
    pd.concat(scs).to_csv(out / "oof_scores.csv", index=False)
    pd.DataFrame(losses).to_csv(out / "loss_history.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "splits.csv", index=False)
    if shaps: pd.DataFrame(shaps).to_csv(out / "shap_tokens.csv", index=False)
    json.dump(infos, open(out / "subset_info.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--disorders", nargs="*", default=list(SUBSETS))
    ap.add_argument("--negatives", default="mixed", choices=["mixed", "twitter", "reddit"])
    ap.add_argument("--ce", type=float, default=1.0); ap.add_argument("--focal", type=float, default=1.0); ap.add_argument("--dice", type=float, default=1.0)
    ap.add_argument("--focal-alpha", type=float, default=0.25); ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=4); ap.add_argument("--inner-val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--warmup-steps", type=int, default=24); ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--model-name", default="distilbert/distilbert-base-uncased"); ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--shap", action="store_true"); ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--train-seed", type=int, default=None); ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    global DATA
    if args.data_dir: DATA = Path(args.data_dir)
    train_seed = args.seed if args.train_seed is None else args.train_seed
    import torch
    from sklearn.model_selection import StratifiedKFold, train_test_split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) / args.tag; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf8")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()
    import transformers, sklearn
    log(f"=== {args.tag} device={device} torch={torch.__version__} transformers={transformers.__version__} sklearn={sklearn.__version__} threads={torch.get_num_threads()} {json.dumps(vars(args))}")
    split_rows = []
    ckdir = out / "folds"; ckdir.mkdir(exist_ok=True)
    for key in args.disorders:
        fname, pos_label, pretty = SUBSETS[key]
        df, info = load_subset(key, args.negatives, args.seed)
        log(f">>> {pretty}: {json.dumps(info)}")
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for k, (tv_idx, te_idx) in enumerate(cv.split(df.Text, df.y), 1):
            if args.max_folds and k > args.max_folds: break
            ck = ckdir / f"{key}_f{k}.pkl"
            if ck.exists():
                log(f"   fold {k}: checkpoint found, skipped"); continue
            tv = df.iloc[tv_idx]; test = df.iloc[te_idx].reset_index(drop=True)
            train, val = train_test_split(tv, test_size=args.inner_val, random_state=args.seed + k, stratify=tv.y)
            if args.train_frac < 1.0:
                train, _ = train_test_split(train, train_size=args.train_frac, random_state=args.seed + k, stratify=train.y)
            sp = []
            for role, fr in (("train", train), ("val", val), ("test", test)):
                sp += [dict(disorder=key, fold=k, row_id=int(r), role=role) for r in fr.row_id]
            train = train.reset_index(drop=True); val = val.reset_index(drop=True)
            log(f"   fold {k}: n_train={len(train)} n_val={len(val)} n_test={len(test)}")
            t0 = time.time()
            vs, ts, tau, lh, sh, tt = train_eval_fold(train, val, test, args, train_seed + k, device, log)
            m = metrics_at(test.y.values, ts, tau)
            log(f"   fold {k}: tau={tau:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']} [{time.time()-t0:.0f}s]")
            rec = dict(info=info, pretty=pretty, n=len(df),
                       fm=dict(disorder=key, fold=k, tau=tau, n_train=len(train), n_val=len(val), n_test=len(test), **m, **tt),
                       scores=pd.DataFrame(dict(disorder=key, fold=k, row_id=test.row_id, y_true=test.y, src=test.src, score=ts, tau=tau, pred=(ts >= tau).astype(int), text=test.n)),
                       loss=[dict(disorder=key, fold=k, **r) for r in lh], shap=[dict(disorder=key, fold=k, **r) for r in sh], splits=sp)
            tmp = ck.with_suffix(".tmp"); pd.to_pickle(rec, tmp); os.replace(tmp, ck)   # atomic: safe against power loss
        aggregate(out, args, log)
    log("DONE " + args.tag + " " + " ".join(args.disorders))


if __name__ == "__main__":
    main()
