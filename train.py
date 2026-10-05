"""Train: pretrained backbone + ArcFace on design ids + colour-consistency loss between two colour-randomised views."""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TwoViewDataset, build_index, make_train_tf, split_by_design
from evaluate import embed, identification_metrics, make_query_gallery
from model import ArcFace, Embedder


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="folders laid out as <root>/<design>/<img>")
    ap.add_argument("--backbone", default="mobilenetv3_large_100")
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of colour-consistency loss")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_designs", type=int, default=0, help="smoke test: use only N designs")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_all(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"

    df = build_index(a.roots)
    if a.max_designs:
        df = df[df.design.isin(sorted(df.design.unique())[: a.max_designs])].reset_index(drop=True)
    tr, va, te = split_by_design(df, seed=a.seed)
    pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]).to_csv(
        os.path.join(a.out, "splits.csv"), index=False)
    print(f"images: train {len(tr)} | val {len(va)} | test {len(te)}  "
          f"designs: {tr.design.nunique()}/{va.design.nunique()}/{te.design.nunique()}")

    classes = {d: i for i, d in enumerate(sorted(tr.design.unique()))}
    dl = DataLoader(TwoViewDataset(tr, classes, make_train_tf(a.img)), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=amp)

    model = Embedder(a.backbone, a.emb).to(dev)
    head = ArcFace(a.emb, len(classes)).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.body.parameters(), "lr": a.lr * 0.3},
        {"params": list(model.neck.parameters()) + list(head.parameters()), "lr": a.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr * 0.3, a.lr], total_steps=a.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)

    best = -1.0
    for ep in range(a.epochs):
        model.train(); head.train()
        run = 0.0
        for x1, x2, y in tqdm(dl, desc=f"epoch {ep + 1}/{a.epochs}"):
            x1, x2, y = x1.to(dev), x2.to(dev), y.to(dev)
            with torch.autocast(device_type=dev, enabled=amp):
                e1, e2 = model(torch.cat([x1, x2])).chunk(2)
            e1, e2 = e1.float(), e2.float()
            loss = head(e1, y) + head(e2, y) + a.lam * (1 - (e1 * e2).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        msg = f"epoch {ep + 1}: loss {run / len(dl):.3f}"

        score = -run  # fallback if no val set
        if len(va) and va.groupby("design").size().max() >= 2:
            F = embed(model, va.path.tolist(), a.img, workers=a.workers, device=dev)
            yv = pd.factorize(va.design)[0]
            q, g = make_query_gallery(va)
            m = identification_metrics(F, yv, q, g)
            score = m["R@1"]
            msg += f" | val R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} mAP {m['mAP']:.3f}"
        print(msg)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "emb": a.emb, "img": a.img},
                       os.path.join(a.out, "best.pt"))
    print("done. best checkpoint:", os.path.join(a.out, "best.pt"))


if __name__ == "__main__":
    main()
