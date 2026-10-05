"""Efficiency report: params, FLOPs, latency, embedding size."""
import argparse
import json
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    model, ck = load_model(a.ckpt)
    s = ck["img"]
    x = torch.randn(1, 3, s, s)
    params = sum(p.numel() for p in model.parameters())
    fc = FlopCounterMode(display=False)
    with fc:
        model(x)
    rep = {"backbone": ck["backbone"], "embedding_dim": ck["emb"], "input": f"{s}x{s}",
           "params_M": round(params / 1e6, 2), "GFLOPs": round(fc.get_total_flops() / 1e9, 3),
           "fp32_MB": round(params * 4 / 1e6, 1)}

    def bench(dev, n=50):
        m = model.to(dev); xx = x.to(dev)
        with torch.no_grad():
            for _ in range(5): m(xx)
            if dev == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n): m(xx)
            if dev == "cuda": torch.cuda.synchronize()
        return round((time.perf_counter() - t) / n * 1000, 2)

    rep["latency_ms_cpu_bs1"] = bench("cpu")
    if torch.cuda.is_available():
        rep["latency_ms_gpu_bs1"] = bench("cuda")
    print(json.dumps(rep, indent=2))
    json.dump(rep, open(a.ckpt.replace("best.pt", "efficiency.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
