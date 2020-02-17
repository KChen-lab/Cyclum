import matplotlib as mpl

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

from cyclum import evaluation


class FigureWriter:
    """
    keep and write figures into a pdf file.
    """
    def __init__(self, pdf_name: str):
        self.figures = []
        if not pdf_name.endswith('.pdf'):
            pdf_name += '.pdf'
        self.pdf_name = pdf_name

    def add_figure(self, figure, title=None):
        """
        add a figure, but not write to file
        :param figure:
        :param title:
        :return:
        """
        if figure != None:
            self.figures.append(figure)
            if title is not None:
                self.figures[-1].suptitle(title)

    def write(self):
        with matplotlib.backends.backend_pdf.PdfPages(self.pdf_name) as pdf:
            for figure in self.figures:
                pdf.savefig(figure, bbox_inches='tight')

    def add_figure_and_write(self, figure, title=None):
        self.add_figure(figure, title)
        self.write()

    def __call__(self, figure, title=None, wait=False):
        """
        add a figure. write to file if not "wait"
        :param figure:
        :param title:
        :param wait: if set to False, write to file.
        :return:
        """
        if wait:
            self.add_figure(figure, title)
        else:
            self.add_figure_and_write(figure, title)


def plot_gene_sparsity(linear_data, use_ratio=True):
    """
    Return a figure of #{cell, none_zero_genes(cell) > x}
    :param linear_data: data
    :param use_ratio: plot as ratio or
    :return:
    """
    figure = plt.figure()
    axes = figure.subplots()

    nonzero_cells_per_gene = np.sum(linear_data > 1e-3, axis=0)

    if use_ratio:
        nonzero_genes_ratio = nonzero_cells_per_gene / linear_data.shape[0]
        axes.plot(np.sort(nonzero_genes_ratio) * 100)
        axes.set_ylabel('nonzero cells %')
    else:
        axes.plot(np.sort(nonzero_cells_per_gene))
        axes.set_ylabel('nonzero cells #')

    axes.set_xlabel("gene # sorted by nonzero genes")
    return figure


def plot_cell_sparsity(linear_data, use_ratio=True):
    """
    Return a figure of #{cell, none_zero_genes(cell) > x}
    :param linear_data: data
    :param use_ratio: plot as ratio or
    :return:
    """
    figure = plt.figure()
    axes = figure.subplots()

    nonzero_genes_per_cell = np.sum(linear_data > 1e-3, axis=1)

    if use_ratio:
        nonzero_genes_ratio = nonzero_genes_per_cell / linear_data.shape[1]
        axes.plot(np.sort(nonzero_genes_ratio) * 100)
        axes.set_ylabel('nonzero genes %')
    else:
        axes.plot(np.sort(nonzero_genes_per_cell))
        axes.set_ylabel('nonzero genes #')

    axes.set_xlabel("cell # sorted by nonzero genes")
    return figure


def plot_pair_color(a, b, color):
    """
    either plot an embedding, two dimensions at a time
    or compare two embeddings
    :param a:
    :param b:
    :param color:
    :return:
    """
    n_col = a.shape[1]
    n_row = b.shape[1]

    figure = plt.figure(figsize=(n_row * 2 + 0.1, n_col * 2 + 0.1))
    ax_list = [figure.add_subplot(n_col, n_row, i + 1) for i in range(n_col * n_row)]

    n = 0
    for i in range(n_col):
        for j in range(n_row):
            ax_list[n].scatter(b[:, j], a[:, i], s=6, c=color)
            n = n + 1

    return figure


def plot_round_color(flat, color):
    figure = plt.figure(figsize=(8, 8))
    axes = figure.subplots()
    xx = np.array([[0.5], [1]]) @ np.cos(flat).T
    yy = np.array([[0.5], [1]]) @ np.sin(flat).T

    for i in range(len(color)):
        axes.plot(xx[:, i], yy[:, i], c=color[i], lw=1.)
    axes.set_xlim([-1.1, 1.1])
    axes.set_ylim([-1.1, 1.1])
    return figure


def plot_round_distr_color(flat, label, color_dict):
    figure = plt.figure()
    ax = figure.subplots(subplot_kw={'projection': 'polar'})
    color = [color_dict[l] for l in label]

    for x, color in zip(flat, color):
        ax.plot([x, x], [1.5, 2], color=color, linewidth=0.5)

    xx = []
    pp = []
    max_p = 0
    for l in color_dict:
        _ = evaluation.periodic_parzen_estimate(flat[label == l], 2 * np.pi)
        xx.append(_[0])
        pp.append(_[1])
        max_p = np.max([np.max(pp[-1]), max_p])
    for x, p, l in zip(xx, pp, color_dict):
        ax.fill_between(x, p / max_p + 2, 2, color=color_dict[l], alpha=0.5, linewidth=0.0)

    ax.set_yticks([])
    return figure


def plot_round_distr_color2(flat, label1, label2, color_dict1, color_dict2):
    figure = plt.figure()
    ax = figure.subplots(subplot_kw={'projection': 'polar'})
    color = [color_dict1[l] for l in label1]

    for x, color in zip(flat, color):
        ax.plot([x, x], [2, 2.5], color=color, linewidth=0.5)

    color = [color_dict2[l] for l in label2]

    for x, color in zip(flat, color):
        ax.plot([x, x], [1.5, 2.0], color=color, linewidth=0.5)

    xx = []
    pp = []
    max_p = 0
    for l in color_dict1:
        _ = evaluation.periodic_parzen_estimate(flat[label1 == l], 2 * np.pi)
        xx.append(_[0])
        pp.append(_[1])
        max_p = np.max([np.max(pp[-1]), max_p])
    for x, p, l in zip(xx, pp, color_dict1):
        ax.fill_between(x, p / max_p + 2.5, 2.5, color=color_dict1[l], alpha=0.6, linewidth=0.0)

    xx = []
    pp = []
    max_p = 0
    for l in color_dict2:
        _ = evaluation.periodic_parzen_estimate(flat[label2 == l], 2 * np.pi)
        xx.append(_[0])
        pp.append(_[1])
        max_p = np.max([np.max(pp[-1]), max_p])
    for x, p, l in zip(xx, pp, color_dict2):
        ax.fill_between(x, 1.5 - p / max_p, 1.5, color=color_dict2[l], alpha=0.6, linewidth=0.0)

    ax.set_yticks([])
    return figure


def plot_multi_distr(xs, ys, colors, labels):
    figure = plt.figure()
    axes = figure.subplots()
    if type(xs) is list:
        for x, y, color, label in zip(xs, ys, colors, labels):
            axes.fill_between(x, y, alpha=0.4, linewidth=0.0, color=color, label=label)
    else:
        x = xs
        for y, color, label in zip(ys, colors, labels):
            axes.fill_between(x, y, alpha=0.4, linewidth=0.0, color=color, label=color)
    axes.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    return figure
