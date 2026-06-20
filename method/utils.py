import numpy as np 
import matplotlib.pyplot as plt


def read_obj(filename):
    V = []
    F = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for l in lines:
            l = l.strip('\n').split(' ')
            if l[0] == 'v':
                V.append([float(l[1]), float(l[2]), float(l[3])])
            elif l[0] == 'f':
                F.append([int(l[i]) - 1 for i in range(1, len(l))])
    return V, F


def write_obj(V, F, filename):
    # F idx starts with 0
    F = F + 1
    with open(filename, 'w+') as file:
        for v in V:
            file.write('v %.6f %.6f %.6f\n' % (v[0], v[1], v[2]))
        for f in F:
            file.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def normalization(V):
    pos_max = np.amax(V, axis=0, keepdims=True)
    pos_min = np.amin(V, axis=0, keepdims=True)
    pos_center = (pos_max + pos_min) / 2
    factor = np.linalg.norm(pos_max - pos_min)
    return (V - pos_center) / factor


def image_grid(images, rows=None, cols=None, fill: bool = True, show_axes: bool = False, rgb: bool = True):
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")
    if rows is None:
        rows = len(images)
        cols = 1
    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))
    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    #plt.show()
