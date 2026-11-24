#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 3: run-to-run versus fold-to-fold variability and paired differences between objectives.

Seed 42 (stored round-2 runs): composite = main_mixed_111_cpu (CPU), CE only = loss_ce_only and
focal only = loss_focal_only (GPU). Seeds 43 and 44 (this round, CPU): seed43_111, seed43_ce, ...
All runs share the fold assignment and inner splits (seed 42), so out-of-fold rows are paired by row_id.
Works with partial results: a category enters only when all its runs are complete.
Output: out/seeds_r3.json
"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
R2 = zipfile.ZipFile(HERE.parent.parent / "experiments_R2.zip")
OBJ = {"composite": ("main_mixed_111_cpu", "seed43_111", "seed44_111"),
       "ce": ("loss_ce_only", "seed43_ce", "seed44_ce"),
       "focal": ("loss_focal_only", "seed43_focal", "seed44_focal")}
CATS = ["anger", "anxiety", "depression", "npd", "panic"]


def load(tag, kind):
    if tag.startswith(("main_", "loss_")):
        return pd.read_csv(io.BytesIO(R2.read(f"out/{tag}/{kind}.csv")))
    p = OUT / tag / f"{kind}.csv"
    return pd.read_csv(p) if p.exists() else None


def wf1(y, p):
    return f1_score(y, p, average="weighted")


def main():
    res = {"categories": {}, "note": "seed 42 = round-2 runs (composite on CPU, CE/focal on GPU); seeds 43/44 on CPU"}
    rng = np.random.default_rng(42)
    for c in CATS:
        runs = {}
        for o, tags in OBJ.items():
            for s, tag in zip((42, 43, 44), tags):
                sc, fm = load(tag, "oof_scores"), load(tag, "fold_metrics")
                if sc is None or fm is None or (sc.disorder == c).sum() == 0 or (fm.disorder == c).sum() < 4:
                    continue
                runs[(o, s)] = (sc[sc.disorder == c].set_index("row_id").sort_index(), fm[fm.disorder == c])
        if len(runs) < 9:
            res["categories"][c] = {"complete": False, "runs_available": len(runs)}; continue
        y = runs[("composite", 42)][0].y_true.values; idx = runs[("composite", 42)][0].index
        d = {"complete": True}
        for o in OBJ:
            f = [wf1(y, runs[(o, s)][0].loc[idx].pred.values) for s in (42, 43, 44)]
            a = [roc_auc_score(y, runs[(o, s)][0].loc[idx].score.values) for s in (42, 43, 44)]
            fms = [runs[(o, s)][1] for s in (42, 43, 44)]
            d[o] = {"f1_by_seed": f, "f1_mean": float(np.mean(f)), "f1_sd_seeds": float(np.std(f, ddof=1)),
                    "fold_sd_mean": float(np.mean([x.f1.std(ddof=1) for x in fms])),
                    # further stability measures (round 3, review note): none of them is a test, all are descriptive
                    "auc_by_seed": [float(v) for v in a], "auc_sd_seeds": float(np.std(a, ddof=1)),
                    "fold_sd_auc_mean": float(np.mean([x.auc.std(ddof=1) for x in fms])),
                    "tau_sd_mean": float(np.mean([x.tau.std(ddof=1) for x in fms])),
                    "tau_range_mean": float(np.mean([x.tau.max() - x.tau.min() for x in fms]))}
        # paired bootstrap of the seed-averaged difference (same texts, same folds)
        P = {o: np.stack([runs[(o, s)][0].loc[idx].pred.values for s in (42, 43, 44)]) for o in OBJ}
        n = len(y); B = 2000
        for alt in ("ce", "focal"):
            obs = np.mean([wf1(y, P[alt][k]) - wf1(y, P["composite"][k]) for k in range(3)])
            boot = []
            for _ in range(B):
                i = rng.integers(0, n, n)
                boot.append(np.mean([wf1(y[i], P[alt][k][i]) - wf1(y[i], P["composite"][k][i]) for k in range(3)]))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            d[f"delta_{alt}_vs_composite"] = {"mean": float(obs), "ci": [float(lo), float(hi)],
                                              "by_seed": [float(wf1(y, P[alt][k]) - wf1(y, P["composite"][k])) for k in range(3)]}
        res["categories"][c] = d
        print(c, {o: round(d[o]["f1_mean"], 3) for o in OBJ}, {o: round(d[o]["f1_sd_seeds"], 3) for o in OBJ},
              "fold SD", round(d["composite"]["fold_sd_mean"], 3),
              "dFocal", round(d["delta_focal_vs_composite"]["mean"], 3), np.round(d["delta_focal_vs_composite"]["ci"], 3),
              "dCE", round(d["delta_ce_vs_composite"]["mean"], 3), np.round(d["delta_ce_vs_composite"]["ci"], 3))
    done = [c for c in CATS if res["categories"][c].get("complete")]
    if len(done) == len(CATS):
        res["mean_over_categories"] = {o: {"by_seed": list(np.mean([res["categories"][c][o]["f1_by_seed"] for c in CATS], axis=0))} for o in OBJ}
        for o in OBJ:
            v = res["mean_over_categories"][o]["by_seed"]; res["mean_over_categories"][o].update(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
        print("mean over categories", {o: (round(res["mean_over_categories"][o]["mean"], 3), round(res["mean_over_categories"][o]["sd"], 3)) for o in OBJ})
    # which objective has the smallest variability, per measure and category (descriptive; three seeds, four folds)
    meas = {"f1_sd_seeds": "F1 across seeds", "auc_sd_seeds": "AUC across seeds", "fold_sd_mean": "F1 across folds",
            "fold_sd_auc_mean": "AUC across folds", "tau_sd_mean": "Youden threshold across folds"}
    st = {}
    for m, label in meas.items():
        win = {c: min(OBJ, key=lambda o: res["categories"][c][o][m]) for c in done}
        vals = [res["categories"][c][o][m] for c in done for o in OBJ]
        spread = [max(res["categories"][c][o][m] for o in OBJ) - min(res["categories"][c][o][m] for o in OBJ) for c in done]
        st[m] = {"label": label, "smallest_by_category": win,
                 "count_smallest": {o: sum(1 for w in win.values() if w == o) for o in OBJ},
                 "range_of_values": [float(min(vals)), float(max(vals))] if vals else None,
                 "max_difference_between_objectives": float(max(spread)) if spread else None}
    res["stability"] = st
    for m, x in st.items():
        print("stability", m, x["count_smallest"], "max diff", None if x["max_difference_between_objectives"] is None else round(x["max_difference_between_objectives"], 3))
    json.dump(res, open(OUT / "seeds_r3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
