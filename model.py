"""Embedding network (pretrained timm backbone + 256-d neck) and ArcFace loss."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, backbone="mobilenetv3_large_100", emb=256, pretrained=True):
        super().__init__()
        self.body = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.neck = nn.Sequential(nn.Linear(self.body.num_features, emb), nn.BatchNorm1d(emb))

    def forward(self, x, neck=True):
        f = self.body(x)
        if neck:
            f = self.neck(f)
        return F.normalize(f, dim=-1)


class ArcFace(nn.Module):
    """Additive angular margin softmax over design ids (training only; dropped at inference)."""

    def __init__(self, emb, n_cls, s=30.0, m=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, emb) * 0.01)
        self.s, self.m = s, m

    def forward(self, x, y):
        with torch.autocast(device_type=x.device.type, enabled=False):
            cos = F.linear(x.float(), F.normalize(self.W)).clamp(-1 + 1e-6, 1 - 1e-6)
            target = torch.cos(torch.acos(cos) + self.m)
            onehot = F.one_hot(y, cos.size(1)).bool()
            logits = torch.where(onehot, target, cos) * self.s
            return F.cross_entropy(logits, y)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = Embedder(ck["backbone"], ck["emb"], pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck
