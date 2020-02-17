"""
Writer gives a very fast way of saving and loading float value matrices.
It saves matrices in binary and in very rigid format. This avoids overheads in csv reading functions.
The R counterpart is also available.
"""

import numpy
import pandas

def int32_to_bytes(x):
    """
    Convert an 32 bit int number to little endian 4 byte binary format.
    This helps writing a integer to a binary file.

    :type x: int32
    :param x: number to be converted
    :return: 4 byte binary
    """
    return x.to_bytes(4, 'little')

def write_matrix_to_binary(file_name, val):
    """
    Write an (unnamed) matrix to a file. The matrix should contain only float values, or at least convertible to float.

    :type file_name: str
    :param file_name: name of file
    :param val: The matrix to write
    :return: None
    """
    with open(file_name, 'wb') as file:
        nrow = val.shape[0]
        ncol = val.shape[1]
        file.write(int32_to_bytes(nrow) + int32_to_bytes(ncol) + val.astype(float).tobytes(order='C'))

def write_df_to_binary(file_name_mask, df):
    """
    Write a data frame to a file. Compared with matrix, it has column and row names
    Besides the row names and column names, the data frame must contain only float values.

    Two files will be saved. For exmaple, a call write_df_to_binary("example", df) will output an "example-value.bin"
    and "example-name.txt". They store the matrix and the column and row names separately.

    :type file_name_mask: str
    :param file_name_mask: the stem of the file name
    :param df: the data frame to write
    :return: None
    """
    write_matrix_to_binary(file_name_mask + '-value.bin', df.values)
    with open(file_name_mask + '-name.txt', 'w') as f:
        f.write("\t".join(df.index))
        f.write("\n")
        f.write("\t".join(df.columns))
        f.write("\n")

def read_matrix_from_binary(file_name):
    """
    Read a matrix from a binary file defined by this module.

    :type file_name: str
    :param file_name: the file to read
    :return: the matrix
    """
    with open(file_name, 'rb') as file:
        buffer = file.read()
    n_row = int.from_bytes(buffer[0:4], 'little')
    n_col = int.from_bytes(buffer[4:8], 'little')
    matrix = numpy.frombuffer(buffer[8:], dtype=float).reshape([n_row, n_col])
    return matrix

def read_df_from_binary(file_name_mask):
    """
    Read a data frame from a binary file defined by this module

    :param file_name_mask:
    :return: the data frame
    """
    data = read_matrix_from_binary(file_name_mask + '-value.bin')
    with open(file_name_mask + '-name.txt', 'r') as f:
        index = f.readline().strip().split('\t')
        columns = f.readline().strip().split('\t')
    return pandas.DataFrame(data=data, index=index, columns=columns)
