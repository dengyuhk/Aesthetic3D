import sys
import os
import torch
import numpy as np
import argparse
import random
import trimesh
import matplotlib.pyplot as plt
from netIMG import *
from renderer import render_single_mesh_3
from utils import read_obj, normalization

# 设置环境变量，避免在某些环境下出现问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置设备和随机种子
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_model(checkpoint_path, device="cuda"):
    """
    加载模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint['config']

    if cfg.architecture == "ModelImg1":
        model = ModelImg1()
    elif cfg.architecture == "ModelImg2":
        model = ModelImg2()
    elif cfg.architecture == "ModelImg3":
        model = ModelImg3(cfg)
    else:
        raise ValueError("Unsupported architecture")

    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def image_grid(imgs, rows=3, cols=4, rgb=True):
    """
    将渲染结果拼接成网格并显示
    imgs: numpy 数组 [N, H, W, C]
    """
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i < imgs.shape[0]:
            if rgb:
                ax.imshow(imgs[i])
            else:
                ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def run_inference(mesh_path, checkpoint_path='./Logs/paras_50.pth', device="cuda"):
    """
    运行推理流程：加载模型、读取网格、渲染图像并推理分数
    """
    # 加载模型
    model = load_model(checkpoint_path, device)

    # 读取 mesh 文件
    v, f = read_obj(mesh_path)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    faces = np.concatenate((mesh.faces, mesh.faces[:, [0, 2, 1]]))  # 双面渲染
    verts = normalization(mesh.vertices)

    # 渲染图像
    imgs = render_single_mesh_3(verts, faces, 12)

    # 显示渲染结果
    print("Displaying rendered image grid...")
    image_grid(imgs, rows=3, cols=4, rgb=True)

    # 将渲染的图像转换为张量
    imgs_tensor = torch.tensor(
        imgs[:, :, :, 0], dtype=torch.float
    ).unsqueeze(1).unsqueeze(0).to(device)

    # 打印张量信息
    print("Tensor shape:", imgs_tensor.shape)
    print("Tensor sample values:", imgs_tensor.flatten()[:20])

    # 推理并返回结果
    with torch.no_grad():
        score = model(imgs_tensor)[0, 0]

    return score, imgs_tensor


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run shape score inference")
    parser.add_argument('--checkpoint', type=str, default='./Logs/paras_50.pth', help='Path to checkpoint file')
    parser.add_argument('--mesh_path', type=str, default='./test.obj', help='Path to mesh file')
    opt = parser.parse_args()

    # 调用推理函数并打印结果
    score, imgs_tensor = run_inference(opt.mesh_path, checkpoint_path=opt.checkpoint, device=device)
    print("Shape score:", score)


if __name__ == "__main__":
    main()
