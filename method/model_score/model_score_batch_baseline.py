import sys
import os
import torch
import numpy as np
import argparse
from utils import *
from netIMG import *
from netIMG_large import *
import random
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
        f.write("# 格式：模型索引\t美学分数（-1~1，越高越美观）\n")
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
        
        # 根据配置选择模型
        if cfg.architecture == "ModelImg1":
            model = ModelImg1()
        elif cfg.architecture == "ModelImg2":
            model = ModelImg2()
        elif cfg.architecture == "ModelImg3":
            model = ModelImg3(cfg)
        elif cfg.architecture == "ModelImgLarge1":
            model = ModelImgLarge1()
        elif cfg.architecture == "ModelImgLarge2":
            model = ModelImgLarge2()
        elif cfg.architecture == "ModelImgHybrid":
            model = ModelImgHybrid()
        else:
            raise ValueError(f"未知模型架构：{cfg.architecture}")
        
        # 加载权重（移除module.前缀）
        from collections import OrderedDict
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        print(f"成功加载模型：{cfg.architecture}")
        
        # 2. 加载预渲染数据
        print("\n=== 加载预渲染数据 ===")
        prerendered_data = load_prerendered_data(opt.pre_rendered_path)
        
        # 3. 批量推理分数
        print("\n=== 批量推理美学分数 ===")
        scores = batch_infer_scores(model, prerendered_data, opt.batch_size)
        
        # 4. 保存分数到txt
        print("\n=== 保存分数结果 ===")
        save_scores_to_txt(scores, opt.output_txt)
        
        print("\n所有模型分数计算完成！")
        
    except Exception as e:
        print(f"\n执行失败：{str(e)}")
        sys.exit(1)