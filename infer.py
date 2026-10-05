"""Inference.
  python infer.py search --ckpt runs/exp1/best.pt --query q.jpg --gallery path/to/gallery_folder --topk 5
  python infer.py verify --ckpt runs/exp1/best.pt --a a.jpg --b b.jpg --thr 0.6
"""
import argparse
import os

import numpy as np
import torch

from evaluate import embed
from model import load_model

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "verify"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--query"); ap.add_argument("--gallery"); ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--thr", type=float, default=0.6)
    a = ap.parse_args()
    model, ck = load_model(a.ckpt)

    if a.cmd == "search":
        paths = [os.path.join(r, f) for r, _, fs in os.walk(a.gallery) for f in fs if f.lower().endswith(EXT)]
        G = embed(model, paths, ck["img"])
        q = embed(model, [a.query], ck["img"])[0]
        s = G @ q
        for i in np.argsort(-s)[: a.topk]:
            print(f"{s[i]:.3f}  {paths[i]}")
    else:
        e = embed(model, [a.a, a.b], ck["img"])
        s = float(e[0] @ e[1])
        print(f"cosine similarity = {s:.3f} -> {'SAME design' if s >= a.thr else 'DIFFERENT design'} (thr {a.thr})")


if __name__ == "__main__":
    main()
