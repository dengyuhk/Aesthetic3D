import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from netIMG import ModelImg2
from utils import read_obj, normalization
from renderer import render_single_mesh_3

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 注册 hook 容器
feature_maps = {}

def get_feature_map(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

def visualize_feature_map(feat, save_path="viewCNN_features.png", nrow=8):
    """
    feat: torch.Tensor [C, H, W]
    使用 torchvision.utils.make_grid 拼接通道
    每个通道使用 viridis 伪彩色显示
    """
    # 归一化到 [0,1]
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-5)
    C, H, W = feat.shape

    # 将每个通道单独显示并拼接
    n_cols = nrow
    n_rows = int(np.ceil(C / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]
        ax.axis("off")
        if i < C:
            ax.imshow(feat[i].numpy(), cmap="viridis")
        else:
            ax.imshow(np.zeros((H, W)), cmap="viridis")  # 空白填充

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature map saved to {save_path}")

def main():
    # 加载模型
    model = ModelImg2().to(device)
    model.eval()

    # 给 viewCNN 注册 hook
    model.viewCNN.register_forward_hook(get_feature_map("viewCNN_out"))

    # 准备输入 mesh -> 渲染图像
    v, f = read_obj("./0098.obj")
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    faces = np.concatenate((mesh.faces, mesh.faces[:, [0, 2, 1]]))  # 双面渲染
    verts = normalization(mesh.vertices)
    imgs = render_single_mesh_3(verts, faces, 12)

    imgs_tensor = torch.tensor(
        imgs[:, :, :, 0], dtype=torch.float
    ).unsqueeze(1).unsqueeze(0).to(device)

    # 前向传播
    with torch.no_grad():
        _ = model(imgs_tensor)

    # 取出捕获的 feature map
    feat = feature_maps["viewCNN_out"]  # shape: [B*12, C, H, W]
    print("Captured viewCNN feature map shape:", feat.shape)

    # 可视化 batch=0, view=0 的特征图
    feat_one = feat[0]  # [C,H,W]
    visualize_feature_map(feat_one, save_path="viewCNN_features2.png", nrow=8)

if __name__ == "__main__":
    main()
