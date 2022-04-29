# %% Imports
import matplotlib.pyplot as plt
from typing import Tuple
import torch
import numpy as np
import math
from PyQt5 import QtWidgets


# %%
def drawnow():
    plt.show(block=False)
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()


def arrange_figs(cols=1, min_rows=3, toolbar=False, x0=1400, y0=28, x1=1920, y1=1200):
    try:
        current_fig_num = plt.gcf().number
        extra = 37
        w = x1 - x0
        h = y1 - y0
        fignums = plt.get_fignums()
        n = len(fignums)
        rows = np.maximum(math.ceil(n / cols), min_rows)
        height = int(h / rows - extra)
        width = int(w / cols)
        for i, fn in enumerate(fignums):
            r = i % rows
            c = int(i / rows)
            plt.figure(fn)
            win = plt.get_current_fig_manager().window
            win.findChild(QtWidgets.QToolBar).setVisible(toolbar)
            win.setGeometry(x0 + width * c, y0 + int(h / rows * r) + extra, width, height)
        plt.figure(current_fig_num)
    except:
        pass


def unravel_indices(indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')
    coord = torch.stack(coord[::-1], dim=-1)
    return coord


def discrete_rand(v: torch.Tensor, n: int = 1):
    idx = torch.sum(torch.rand(n)[:, None].to(v.device) > torch.cumsum(v.flatten(), 0)[None, :] / torch.sum(v), dim=1)
    return unravel_indices(idx, v.shape)


def local_scramble_2d(dist: float, dim: tuple):
    grid = torch.meshgrid(*[torch.arange(d) for d in dim])
    n = [torch.argsort(m + torch.randn(dim) * dist, dim=i) for i, m in enumerate(grid)]
    idx = torch.reshape(torch.arange(torch.tensor(dim).prod()), dim)
    return idx[n[0], grid[1]][grid[0], n[1]].flatten()

def bsgen(v, v0):
    while v > v0:
        yield v
        v = math.ceil(v / 2)
