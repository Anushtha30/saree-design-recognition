"""Evaluation: identification (Recall@K, mAP) and verification (AUC, EER, TAR@FAR) on held-out designs.

Protocol (documented for the report):
  * Split by design -> test designs never seen in training.
  * Identification: for every test design with >=2 images, one image is the QUERY (seeded);
    the rest go to the GALLERY. Single-image designs go to the gallery as distractors.
  * Verification: positives = same design (different image/colorway); negatives are of two kinds:
      - random: different design
      - hard / same-palette: different design chosen to have the most similar colour histogram
    The 'hard' set directly tests the requirement "different motifs must not match even in identical palettes".
  * Threshold is picked on VAL pairs (Youden J) and applied unchanged to TEST.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from data import EvalDataset, make_eval_tf
from model import Embedder, load_model


@torch.no_grad()
def embed(model, paths, size=224, bs=128, workers=2, device=None, neck=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    dl = DataLoader(EvalDataset(paths, make_eval_tf(size)), batch_size=bs, num_workers=workers)
    out = [model(x.to(device), neck=neck).float().cpu() for x in dl]
    return torch.cat(out).numpy()


def make_query_gallery(df, seed=0):
    rng = np.random.RandomState(seed)
    q_idx, g_idx = [], []
    for _, g in df.groupby("design"):
        idx = g.index.to_numpy()
        if len(idx) >= 2:
            q = rng.choice(idx)
            q_idx.append(q)
            g_idx += [i for i in idx if i != q]
        else:
            g_idx += list(idx)
    return np.array(q_idx), np.array(g_idx)


def identification_metrics(F, y, q, g):
    S = F[q] @ F[g].T
    yq, yg = y[q], y[g]
    r1 = r5 = ap = 0.0
    for i in range(len(q)):
        m = yg[np.argsort(-S[i])] == yq[i]
        r1 += m[0]
        r5 += m[:5].any()
        hits = np.where(m)[0]
        ap += np.mean((np.arange(len(hits)) + 1) / (hits + 1)) if len(hits) else 0.0
    n = len(q)
    return {"R@1": r1 / n, "R@5": r5 / n, "mAP": ap / n, "n_query": n, "n_gallery": len(g)}


def color_hist(path, bins=4):
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 64))).reshape(-1, 3) // (256 // bins)
    idx = a[:, 0] * bins * bins + a[:, 1] * bins + a[:, 2]
    h = np.bincount(idx, minlength=bins ** 3).astype(float)
    return h / h.sum()


def make_pairs(df, n_pos=4000, seed=0):
    rng = np.random.RandomState(seed)
    y = pd.factorize(df.design)[0]
    by = {}
    for i, c in enumerate(y):
        by.setdefault(c, []).append(i)
    multi = [v for v in by.values() if len(v) >= 2]
    pos = []
    for _ in range(n_pos):
        v = multi[rng.randint(len(multi))]
        a, b = rng.choice(v, 2, replace=False)
        pos.append((a, b))
    H = np.stack([color_hist(p) for p in df.path])
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    rand_neg, hard_neg = [], []
    for a, _ in pos:
        b = rng.randint(len(df))
        while y[b] == y[a]:
            b = rng.randint(len(df))
        rand_neg.append((a, b))
        s = Hn @ Hn[a]
        s[y == y[a]] = -1
        hard_neg.append((a, int(rng.choice(np.argsort(-s)[:5]))))
    return pos, rand_neg, hard_neg


def pair_scores(F, pairs):
    p = np.array(pairs)
    return (F[p[:, 0]] * F[p[:, 1]]).sum(1)


def verif_metrics(pos_s, neg_s, thr=None):
    scores = np.concatenate([pos_s, neg_s])
    labels = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    fpr, tpr, th = roc_curve(labels, scores)
    out = {"AUC": roc_auc_score(labels, scores),
           "EER": float(fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]),
           "TAR@FAR1%": float(np.interp(0.01, fpr, tpr))}
    if thr is not None:
        out["acc@thr"] = float(((scores >= thr) == labels).mean())
    return out, float(th[np.argmax(tpr - fpr)])


def run_eval(model, df, size=224, neck=True, seed=0):
    df = df.reset_index(drop=True)
    F = embed(model, df.path.tolist(), size=size, neck=neck)
    y = pd.factorize(df.design)[0]
    q, g = make_query_gallery(df, seed)
    res = {"identification": identification_metrics(F, y, q, g)}
    pos, rn, hn = make_pairs(df, seed=seed)
    ps = pair_scores(F, pos)
    res["verification_random_neg"], thr_r = verif_metrics(ps, pair_scores(F, rn))
    res["verification_hard_neg"], thr_h = verif_metrics(ps, pair_scores(F, hn))
    return res, {"random": thr_r, "hard": thr_h}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing best.pt and splits.csv")
    ap.add_argument("--baseline", default="", help="backbone name -> evaluate frozen ImageNet features (no training)")
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    sp = pd.read_csv(os.path.join(a.run, "splits.csv"))
    val, test = sp[sp.split == "val"].reset_index(drop=True), sp[sp.split == "test"].reset_index(drop=True)
    if a.baseline:
        model, neck, tag = Embedder(a.baseline, pretrained=True), False, f"baseline_{a.baseline}"
    else:
        model, _ = load_model(os.path.join(a.run, "best.pt"))
        neck, tag = True, "trained"
    _, thr = run_eval(model, val, a.size, neck)          # threshold chosen on VAL
    res, _ = run_eval(model, test, a.size, neck)
    # re-score test verification at the val threshold
    F = embed(model, test.path.tolist(), a.size, neck=neck)
    pos, rn, hn = make_pairs(test)
    for k, negs in (("verification_random_neg", rn), ("verification_hard_neg", hn)):
        res[k], _ = verif_metrics(pair_scores(F, pos), pair_scores(F, negs), thr["hard"])
    res["threshold_from_val(hard)"] = thr["hard"]
    print(json.dumps(res, indent=2, default=float))
    json.dump(res, open(os.path.join(a.run, f"results_{tag}.json"), "w"), indent=2, default=float)
