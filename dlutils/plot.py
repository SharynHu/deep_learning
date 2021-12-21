import numpy
import matplotlib
from matplotlib import pyplot as plt

def show_heatmap_2d(matrices, xlabel, ylabel,titles=None, colormap="Reds"):
    """
    Show 2d heatmaps of matrices.
    """
    num_rows, num_cols = matrices.shape[:2]
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,  sharex=True, sharey=True, squeeze=False)
    
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (axis, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = axis.imshow(matrix.detach().numpy(), cmap=colormap)
            if i == num_rows - 1:
                axis.set_xlabel(xlabel)
            if j == 0:
                axis.set_ylabel(ylabel)
            if titles:
                axis.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);