import sys
import os
import torch
import numpy as np
import argparse
from utils import *
from netIMG import *
from netIMG_large import *
import random
import trimesh
from renderer import render_single_mesh_3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["PYOPENGL_PLATFORM"] = "egl"


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('')
parser.add_argument('--checkpoint', type=str, default='./Logs/Logs/paras_50.pth')
parser.add_argument('--mesh_path', type=str, default='./test.obj')
opt = parser.parse_args()

if __name__ == "__main__":
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    cfg = checkpoint['config']
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
        print(f"error architecture! Unknown architecture: {cfg.architecture}")
        sys.exit(1)
    model.to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    from collections import OrderedDict
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # 去掉 "module."
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    
    model.eval()

    v, f = read_obj(opt.mesh_path)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    faces = np.concatenate((mesh.faces, mesh.faces[:, [0, 2, 1]]))  # double-face rendering
    verts = normalization(mesh.vertices)
    imgs = render_single_mesh_3(verts, faces, 12)
    # vis_images
    image_grid(imgs, rows=3, cols=4, rgb=True)
    imgs = torch.tensor(imgs[:, :, :, 0], dtype=torch.float).unsqueeze(1).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(imgs)[0, 0]
    print("shape score: ", score)
