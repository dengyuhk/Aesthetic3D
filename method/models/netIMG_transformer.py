# netIMG.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os

# -----------------------
# Dataset (unchanged interface)
# -----------------------
class DatasetPair(Dataset):
    def __init__(self, pairs, data):
        self.data = data  # expected tensor-like: N x V x 1 x H x W
        self.pairs = pairs

    def __getitem__(self, index):
        idx1 = int(self.pairs[index][0])
        idx2 = int(self.pairs[index][1])
        shape1 = self.data[idx1]
        shape2 = self.data[idx2]
        margin_w = torch.tensor([float(self.pairs[index][2])], dtype=torch.float32)
        return shape1, shape2, margin_w

    def __len__(self):
        return len(self.pairs)


class DatasetSingle(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# -----------------------
# Weighted ranking loss (kept compatible with original code)
# pairs format from README: [idx1, idx2, weight] where idx1 is better than idx2
# margin: scalar hyperparameter (provided by config)
# loss = mean( clamp( margin_w * margin - (s1 - s2), min=0 ) )
# -----------------------
def weighted_loss(s1, s2, margin_w, margin):
    # s1, s2: (B,1) or (B,) ; margin_w: (B,1) or (B,)
    temp = margin_w * margin - (s1 - s2)
    loss = torch.clamp(temp, min=0.0)
    return torch.mean(loss)


# -----------------------
# Small convolutional per-view encoder (shared across views)
# -----------------------
class SmallViewEncoder(nn.Module):
    """
    Input per view: (B, 1, H, W)
    Output per view: vector (B, C)
    We'll keep H/W -> fairly small (e.g., 8x8) then global avg to vector.
    """
    def __init__(self, out_channels=128):
        super().__init__()
        # keep simple and efficient
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128->64

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64->32

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32->16

            nn.Conv2d(128, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # -> (B, out_channels, 1, 1)
        )

    def forward(self, x):
        """
        x: (B, 1, H, W)
        returns (B, out_channels)
        """
        out = self.net(x)         # (B, C, 1, 1)
        out = out.view(out.size(0), out.size(1))
        return out


# -----------------------
# View Transformer Fusion
# - treat V views as V tokens
# - prepend a learnable CLS token
# - use nn.TransformerEncoder to fuse
# -----------------------
class ViewTransformerFusion(nn.Module):
    def __init__(self, in_dim, embed_dim=256, n_heads=8, num_layers=2, dropout=0.1, max_views=12):
        """
        in_dim: per-view input dim (from SmallViewEncoder)
        embed_dim: transformer embedding dim
        n_heads: number of attention heads
        num_layers: transformer encoder layers
        """
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.max_views = max_views

        # project per-view vector -> embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # cls token and positional embeddings (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)  # (1,1,E)
        self.pos_embed = nn.Parameter(torch.randn(1, max_views + 1, embed_dim) * 0.02)  # (1, V+1, E)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   dropout=dropout, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final layernorm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, V, in_dim)
        returns: cls embedding (B, embed_dim)
        """
        B, V, C = x.shape
        assert V <= self.max_views, f"V ({V}) > max_views ({self.max_views}). Increase max_views."

        # project
        x = self.input_proj(x)             # (B, V, E)

        # prepare cls tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,E)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, V+1, E)

        # add positional embeddings (slice if V < max_views)
        pos = self.pos_embed[:, : (V + 1), :].to(x.dtype).to(x.device)  # (1, V+1, E)
        x = x + pos

        # transformer (batch_first=True)
        x = self.transformer(x)  # (B, V+1, E)
        x = self.norm(x)

        # output cls token
        cls_out = x[:, 0, :]  # (B, E)
        return cls_out


# -----------------------
# Final upgraded models
# - ModelImg1 / ModelImg2 / ModelImg3 all use Transformer fusion but with different encoder widths
# - Output stays (B,1) to be compatible with training script
# -----------------------

class BaseTransformerModel(nn.Module):
    def __init__(self, view_encoder_out=128, transformer_embed=256, transformer_heads=8, transformer_layers=2,
                 dropout=0.1, n_views=12):
        super().__init__()
        self.n_views = n_views
        self.view_encoder = SmallViewEncoder(out_channels=view_encoder_out)
        self.view_fusion = ViewTransformerFusion(in_dim=view_encoder_out,
                                                 embed_dim=transformer_embed,
                                                 n_heads=transformer_heads,
                                                 num_layers=transformer_layers,
                                                 dropout=dropout,
                                                 max_views=n_views)
        # small MLP head from embed -> scalar
        self.head = nn.Sequential(
            nn.Linear(transformer_embed, transformer_embed // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(transformer_embed // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: (B, V, 1, H, W)
        returns (B,1)
        """
        B, V, Cc, H, W = x.shape
        assert V == self.n_views, f"Expect {self.n_views} views, got {V}"

        # reshape to (B*V, 1, H, W) and encode each view
        x = x.view(B * V, Cc, H, W)
        vfeat = self.view_encoder(x)    # (B*V, C)
        vfeat = vfeat.view(B, V, -1)    # (B, V, C)

        # fusion with transformer -> cls vector
        cls = self.view_fusion(vfeat)   # (B, E)

        out = self.head(cls)            # (B,1)
        return out


# ModelImg1 / 2 / 3 with different capacities (user can choose by opt.architecture)
class ModelImg1(BaseTransformerModel):
    def __init__(self):
        # stronger encoder + transformer
        super().__init__(view_encoder_out=128, transformer_embed=256, transformer_heads=8, transformer_layers=3,
                         dropout=0.1, n_views=12)


class ModelImg2(BaseTransformerModel):
    def __init__(self):
        # smaller model
        super().__init__(view_encoder_out=64, transformer_embed=128, transformer_heads=4, transformer_layers=2,
                         dropout=0.1, n_views=12)


class ModelImg3(BaseTransformerModel):
    def __init__(self, cfg=None):
        # configurable larger model (cfg ignored except for compatibility)
        super().__init__(view_encoder_out=128, transformer_embed=384, transformer_heads=12, transformer_layers=4,
                         dropout=0.15, n_views=12)

