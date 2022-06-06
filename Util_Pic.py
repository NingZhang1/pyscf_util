from urllib import robotparser
import seaborn
import pandas
import matplotlib.pyplot as plt
import numpy


def draw_heatmap(mat, column, indx, vmax=100, vmin=0, x_label="orbital", y_label="atom",
                 annot=False,max_size=16):
    max_len = max(mat.shape[1],mat.shape[0])
    row_len = mat.shape[1]
    col_len = mat.shape[0]
    print(max_len,row_len,col_len)
    if max_len>16:
        row_len = (row_len/max_len)*16
        col_len = (col_len/max_len)*16
    fig, ax = plt.subplots(figsize=(row_len, col_len))
    seaborn.heatmap(pandas.DataFrame(numpy.round(mat, 2), columns=column, index=indx),
                    annot=annot,
                    robust=True,
                    vmax=vmax, vmin=vmin,
                    xticklabels=True, yticklabels=True,
                    square=True, cmap="Blues")
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel(x_label, fontsize=18)
    plt.show()
