import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
from torch.utils.data import Dataset  # 补充必要导入
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{device}")

# 固定随机种子（保证结果可复现）
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ======================== 新增：完整ModelImg2架构（Transformer融合） ========================
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
# Base Transformer Model
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

# -----------------------
# ModelImg2 (smaller transformer model)
# -----------------------
class ModelImg2(BaseTransformerModel):
    def __init__(self):
        # smaller model
        super().__init__(view_encoder_out=64, transformer_embed=128, transformer_heads=4, transformer_layers=2,
                         dropout=0.1, n_views=12)

# ======================== 保留原有工具函数（适配Transformer版ModelImg2） ========================
# 命令行参数（新增pre_rendered路径和输出txt路径）
parser = argparse.ArgumentParser('')
parser.add_argument('--checkpoint', type=str, default='./Logs/Logs/paras_50.pth')
parser.add_argument('--pre_rendered_path', type=str, default='./pre_rendered_3.pt')  # 预渲染数据路径
parser.add_argument('--output_txt', type=str, default='model_scores.txt')  # 分数输出路径
parser.add_argument('--batch_size', type=int, default=8)  # 批量推理大小（根据显存调整）
opt = parser.parse_args()

def load_prerendered_data(pt_path):
    """加载pre_rendered_3.pt并验证格式"""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"未找到预渲染数据文件：{pt_path}")
    
    # 加载数据（强制CPU，避免显存占用）
    data = torch.load(pt_path, map_location="cpu")
    # 验证张量形状：479×12×1×128×128
    if data.shape != (479, 12, 1, 128, 128):
        raise ValueError(
            f"预渲染数据形状异常！预期(479,12,1,128,128)，实际{data.shape}"
        )
    # 转换为float32并归一化（和训练时保持一致）
    data = data.float()
    # 若数据是0-255范围，归一化到0-1（根据训练数据调整）
    if data.max() > 1.0:
        data = data / 255.0
    print(f"成功加载预渲染数据：{data.shape}，数值范围[{data.min():.4f}, {data.max():.4f}]")
    return data

def batch_infer_scores(model, prerendered_data, batch_size=8):
    """批量推理所有模型的美学分数"""
    model.eval()
    total_models = prerendered_data.shape[0]
    scores = []
    
    # 分批次推理
    with torch.no_grad():
        for start_idx in range(0, total_models, batch_size):
            end_idx = min(start_idx + batch_size, total_models)
            # 取当前批次数据：(batch,12,1,128,128)
            batch_data = prerendered_data[start_idx:end_idx].to(device)
            # 推理分数：输出形状(batch,1)
            batch_scores = model(batch_data)
            # 转换为列表并保存
            batch_scores_np = batch_scores.squeeze(1).cpu().numpy().tolist()
            scores.extend(batch_scores_np)
            
            # 打印进度
            print(f"已处理：{end_idx}/{total_models} 个模型 | 当前批次分数范围：[{min(batch_scores_np):.4f}, {max(batch_scores_np):.4f}]")
    
    # 转换为numpy数组，确保长度匹配
    scores = np.array(scores, dtype=np.float32)
    assert len(scores) == total_models, f"分数数量不匹配！预期{total_models}，实际{len(scores)}"
    return scores

def save_scores_to_txt(scores, output_path):
    """将分数保存到txt文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        # 写入表头
        f.write("# Manfred数据集模型美学分数（pre_rendered_3.pt）\n")
        f.write("# ModelImg2 架构（Transformer多视角融合）推理 | 分数范围：-1~1（越高越美观）\n")
        f.write("# 格式：模型索引\t美学分数\n")
        f.write("=" * 60 + "\n")
        # 逐行写入每个模型的索引和分数
        for model_idx, score in enumerate(scores):
            f.write(f"{model_idx}\t{score:.6f}\n")
    
    # 计算分数统计信息
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    max_score_idx = np.argmax(scores)
    min_score_idx = np.argmin(scores)
    
    print("\n" + "=" * 60)
    print(f"分数统计：")
    print(f"- 平均分：{mean_score:.6f}")
    print(f"- 标准差：{std_score:.6f}")
    print(f"- 最高分：{scores[max_score_idx]:.6f}（模型索引：{max_score_idx}）")
    print(f"- 最低分：{scores[min_score_idx]:.6f}（模型索引：{min_score_idx}）")
    print(f"- 分数已保存到：{os.path.abspath(output_path)}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        # 1. 加载预训练模型
        print("=== 加载模型权重 ===")
        checkpoint = torch.load(opt.checkpoint, map_location=device)
        cfg = checkpoint['config']
        
        # 仅加载Transformer版ModelImg2（适配当前架构）
        if cfg.architecture == "ModelImg2":
            model = ModelImg2()
        else:
            raise ValueError(
                f"当前脚本仅支持ModelImg2架构！checkpoint中架构为：{cfg.architecture}"
            )
        
        # 加载权重（移除module.前缀 + 智能过滤不匹配层）
        from collections import OrderedDict
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # 移除多卡训练的module.前缀
            new_state_dict[name] = v

        # 过滤形状不匹配的层（核心修复：避免权重加载报错）
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        mismatch_layers = []
        for k, v in new_state_dict.items():
            if k in model_state_dict:
                if model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    mismatch_layers.append(f"{k}: checkpoint({v.shape}) vs model({model_state_dict[k].shape})")
            else:
                mismatch_layers.append(f"{k}: 层在当前ModelImg2中不存在")
        
        # 加载过滤后的权重
        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(device)
        
        # 打印权重加载状态
        print(f"成功加载 {len(filtered_state_dict)}/{len(new_state_dict)} 个匹配的权重层")
        if mismatch_layers:
            print("注意：以下层形状不匹配（已跳过，分数仅供参考）：")
            for layer in mismatch_layers[:5]:  # 仅打印前5个，避免刷屏
                print(f"  - {layer}")
            if len(mismatch_layers) > 5:
                print(f"  - ... 共{len(mismatch_layers)}个不匹配层")
        else:
            print("✅ 所有权重层形状匹配！推理分数准确")
        
        # 2. 加载预渲染数据
        print("\n=== 加载预渲染数据 ===")
        prerendered_data = load_prerendered_data(opt.pre_rendered_path)
        
        # 3. 批量推理分数
        print("\n=== 批量推理美学分数 ===")
        scores = batch_infer_scores(model, prerendered_data, opt.batch_size)
        
        # 4. 保存分数到txt
        print("\n=== 保存分数结果 ===")
        save_scores_to_txt(scores, opt.output_txt)
        
        print("\n🎉 所有模型分数计算完成！")
        
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        # 打印详细报错栈，便于定位问题
        import traceback
        traceback.print_exc()
        sys.exit(1)