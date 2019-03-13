"""
Provides filters to filter out genes and cells.
"""

import numpy as np


def cell_sparsity(data, ratio=None, count=None, threshold=0.5, return_mask=False):
    """
    filter cells by how many 0 genes they have
    :param data:
    :param ratio: ratio of non-zero entries. Either a number or a pair of numbers. Treat as [ ).
    :param count: number of non-zero entries. Treat as [ ).
    :param threshold: threshold to decide what is considered 0
    :param return_mask: if set to True, output mask instead of matrix
    You must supply ratio XOR number, i.e. one of them and only one of them.
    :return:
    """
    if (ratio is None) == (count is None):
        raise Exception("Must supply ratio XOR number!")

    nonzero_genes_per_cell = np.sum(data > threshold, axis=1)

    if ratio is not None:
        if type(ratio) in (list, tuple):
            if len(ratio) != 2:
                raise Exception("If ratio is a tuple/list, its length must be 2!")
        else:
            ratio = (ratio, np.inf)
            nonzero_genes_ratio = nonzero_genes_per_cell / data.shape[1]
            mask = (nonzero_genes_ratio >= ratio[0]) & (nonzero_genes_ratio < ratio[1])
    else:
        if type(count) in (list, tuple):
            if len(count) != 2:
                raise Exception("If count is a tuple/list, its length must be 2!")
        else:
            count = (count, np.inf)
            mask = (nonzero_genes_per_cell >= count[0]) & (nonzero_genes_per_cell < count[1])

    if return_mask:
        return mask
    else:
        return data.loc[mask, :]



def gene_sparsity(data, ratio=None, count=None, threshold=0.5, return_mask=False):
    """
    filter cells by how many 0 genes they have
    :param linear_data: data
    :param ratio: ratio of non-zero entries. Either a number or a pair of numbers. Treat as [ ).
    :param count: number of non-zero entries. Treat as [ ).
    :param threshold: threshold to decide what is considered 0
    :param return_mask: if set to True, output mask instead of matrix
    You must supply ratio XOR number, i.e. one of them and only one of them.
    :return:
    """
    if (ratio is None) == (count is None):
        raise Exception("Must supply ratio XOR number!")

    nonzero_cells_per_gene = np.sum(data > threshold, axis=0)

    if ratio is not None:
        if type(ratio) in (list, tuple):
            if len(ratio) != 2:
                raise Exception("If ratio is a tuple/list, its length must be 2!")
        else:
            ratio = (ratio, np.inf)
            nonzero_cells_ratio = nonzero_cells_per_gene / data.shape[0]
            mask = (nonzero_cells_ratio >= ratio[0]) & (nonzero_cells_ratio < ratio[1])
    else:
        if type(count) in (list, tuple):
            if len(count) != 2:
                raise Exception("If count is a tuple/list, its length must be 2!")
        else:
            count = (count, np.inf)
            mask = (nonzero_cells_per_gene >= count[0]) & (nonzero_cells_per_gene < count[1])

    if return_mask:
        return mask
    else:
        return data.loc[:, mask]


# Todo: probably fruitfulness (exclude 0s or not?). A lot of things we can do. Not at this time...
