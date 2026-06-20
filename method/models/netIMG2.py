import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import torch.nn.functional as F


# ============================================================
#                ★★★ Dataset: ShapeImageDataset ★★★
# ============================================================

class ShapeImageDataset(Dataset):
    def __init__(self, data_path, pairs_path):
        print("Loading images:", data_path)
        self.data = torch.load(data_path)  # 479*12*1*128*128
        print("Loading pairs:", pairs_path)
        pairs_raw = torch.load(pairs_path)

        # 自动适配 dict {i:[j, margin, ...]} 格式
        if isinstance(pairs_raw, dict):
            print("Detected pair format: dict {i: [j, margin, ...]}")
            self.pairs = []
            for i in pairs_raw:
                raw = pairs_raw[i]
                if len(raw) < 2:
                    raise ValueError(f"Pairs[{i}] has fewer than 2 elements: {raw}")
                idx1 = i
                idx2 = raw[0]
                weight = raw[1]      # margin / weight
                self.pairs.append([idx1, idx2, weight])
        else:
            raise ValueError("Unsupported pairs format, should be dict")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, weight = self.pairs[idx]
        shape1 = self.data[idx1]
        shape2 = self.data[idx2]
        return shape1, shape2, torch.tensor([weight], dtype=torch.float32)




# ============================================================
#                ★★★ CLIP Aesthetic Model ★★★
# ============================================================

class CLIP_Aesthetic_Model(nn.Module):
    """
    输入：B × 12 × 1 × 128 × 128
    处理：repeat → resize → CLIP
    输出：B × 1 aesthetic score（tanh）
    """
    def __init__(self, clip_model_name="ViT-B/32", clip_root="~/CLIP"):
        super().__init__()

        # ------------------ Load CLIP ------------------
        clip_root = os.path.expanduser(clip_root)
        import sys
        sys.path.append(clip_root)

        import clip
        self.clip_model, _ = clip.load(clip_model_name, device="cuda")
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.clip_dim = self.clip_model.visual.output_dim  # 512

        # ------------------ Head ------------------
        self.score_head = nn.Sequential(
            nn.Linear(self.clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        B, V, C, W, H = x.shape

        # B*V × 1 × 128 ×128
        x = x.view(B * V, 1, W, H)

        # CLIP 需要 3-channel
        x = x.repeat(1, 3, 1, 1)

        # resize → 224×224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        feat = self.clip_model.encode_image(x)   # (B*V, 512)

        feat = feat.view(B, V, self.clip_dim)

        # 多视角聚合
        feat = feat.max(dim=1)[0]  # (B,512)

        score = self.score_head(feat)
        return score


# 保持兼容名称
ModelImg1 = CLIP_Aesthetic_Model
ModelImg2 = CLIP_Aesthetic_Model
ModelImg3 = CLIP_Aesthetic_Model
ModelImg = CLIP_Aesthetic_Model
