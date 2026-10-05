"""Data indexing, design-level splits, and colour-destroying augmentation."""
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_index(roots):
    """Index folders laid out as <root>/<design_id>/<image files>.
    Each sub-folder = one design (all colorways inside). If your data is laid out
    differently (e.g. a CSV with design ids), replace this function only."""
    rows = []
    for root in roots:
        tag = os.path.basename(os.path.normpath(root))
        for design in sorted(os.listdir(root)):
            d = os.path.join(root, design)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    rows.append((os.path.join(d, f), f"{tag}/{design}"))
    return pd.DataFrame(rows, columns=["path", "design"])


def split_by_design(df, val_frac=0.10, test_frac=0.15, seed=0):
    """Split by DESIGN (not by image) so val/test designs are never seen in training.
    This is what makes the evaluation honest: the model must generalise to new motifs."""
    designs = np.array(sorted(df.design.unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(designs)
    n_te, n_va = int(len(designs) * test_frac), int(len(designs) * val_frac)
    te, va = set(designs[:n_te]), set(designs[n_te:n_te + n_va])
    tr_df = df[~df.design.isin(te | va)].reset_index(drop=True)
    va_df = df[df.design.isin(va)].reset_index(drop=True)
    te_df = df[df.design.isin(te)].reset_index(drop=True)
    return tr_df, va_df, te_df


class ColorChaos:
    """Randomly destroys palette information so the network cannot rely on colour:
    channel permutation, full hue rotation, saturation 0-2x, brightness/contrast, grayscale."""

    def __call__(self, img):
        if random.random() < 0.5:
            img = TF.to_pil_image(TF.to_tensor(img)[torch.randperm(3)])
        img = TF.adjust_hue(img, random.uniform(-0.5, 0.5))
        img = TF.adjust_saturation(img, random.uniform(0.0, 2.0))
        img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
        img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
        if random.random() < 0.3:
            img = TF.rgb_to_grayscale(img, 3)
        return img


def make_train_tf(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        ColorChaos(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def make_eval_tf(size=224):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class TwoViewDataset(Dataset):
    """Returns two independently crop+colour-augmented views of the same image + design label."""

    def __init__(self, df, class_map, tf):
        self.paths = df.path.tolist()
        self.labels = [class_map[d] for d in df.design]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.tf(img), self.labels[i]


class EvalDataset(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))
