"""
Provide transformation from count matrix to TPM/PKM.
It also supports transforming data frames
"""

def calc_tpm(count_matrix, gene_length_vector, is_cell_row = True):
    """
    Transformation from count matrix to TPM matrix.

    :param count_matrix: count matrix
    :param gene_length_vector: lengths of genes
    :param is_cell_row: if true, cells should be rows
    :return:
    """
    axis = 1 if is_cell_row else 0
    count = count_matrix.copy()
    # count -> reads per kilobase
    rpk = count / gene_length_vector.reshape([-1, 1]) * 1_000
    scailing_factor = rpk.sum(axis=axis, keepdims=True) / 1_000_000
    return rpk / scailing_factor


def calc_pkm(count_matrix, gene_length_vector, is_cell_row = True):
    """
    Transformation from count matrix to PKM matrix.

    :param count_matrix: count matrix
    :param gene_length_vector: lengths of genes
    :param is_cell_row: if true, cells should be rows
    :return:
    """

    axis = 1 if is_cell_row else 0
    count = count_matrix.copy()
    # count -> reads per million
    scailing_factor = count.sum(axis=axis, keepdims=True) / 1_000_000
    rpm = count / scailing_factor
    return rpm / gene_length_vector.reshape([-1, 1]) * 1_000


def for_df(func):
    def new_func(df, gene_length_vector, is_cell_row=True):
        df = df.astype('float')
        df.values[:, :] = func(df.values, gene_length_vector, is_cell_row)
        return df
    return new_func


calc_tpm_for_df = for_df(calc_tpm)
calc_pkm_for_df = for_df(calc_pkm)
