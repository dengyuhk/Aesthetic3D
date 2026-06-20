import sys
import os
import torch
import numpy as np
import argparse
from utils import *
from netIMG import *
import random
import trimesh
from renderer import render_single_mesh_3

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
    :param checkpoint_path: 模型检查点文件路径
    :param device: 设备类型（'cuda' 或 'cpu'）
    :return: 加载好的模型
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


def run_inference(mesh_path, checkpoint_path='./Logs/paras_50.pth', device="cuda"):
    """
    运行推理流程：加载模型、读取网格、渲染图像并推理分数
    :param mesh_path: 输入网格路径
    :param checkpoint_path: 模型检查点路径
    :param device: 设备类型（'cuda' 或 'cpu'）
    :return: 推理得到的分数
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

    # 可视化渲染结果
    image_grid(imgs, rows=3, cols=4, rgb=True)

    # 将渲染的图像转换为张量
    imgs = torch.tensor(imgs[:, :, :, 0], dtype=torch.float).unsqueeze(1).unsqueeze(0).to(device)

    # 推理并返回结果
    with torch.no_grad():
        score = model(imgs)[0, 0]

    return score


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run shape score inference")
    parser.add_argument('--checkpoint', type=str, default='./Logs/Logs/paras_50.pth', help='Path to checkpoint file')
    parser.add_argument('--mesh_path', type=str, default='./test.obj', help='Path to mesh file')
    opt = parser.parse_args()

    # 调用推理函数并打印结果
    score = run_inference(opt.mesh_path, checkpoint_path=opt.checkpoint, device=device)
    print("Shape score:", score)


if __name__ == "__main__":
    main()
