import argparse
import cyclum
import numpy as np
import pandas as pd
import sklearn as skl
import cyclum.writer as writer

# What is needed:
# Necessary: input name
# Can be substituted: output name (use the prefix of input)
# options: --remove
# options: --binary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recover and remove cell cycle.')
    parser.add_argument('input_file_name', metavar='input', type=str,
                        help='input file name')
    parser.add_argument('output_file_mask', metavar='output', type=str, nargs='?',
                        help='output name mask; the prefix for all output names')
    parser.add_argument('--cell-is-column', action='store_true',
                        help='indicating that each column represents a cells')
    parser.add_argument('--no-transform', action='store_true',
                        help='indicating that no further transform is needed')
    #parser.add_argument('--remove', action='store_true',
    #                    help='also output the cell cycle removed expression matrix')
    parser.add_argument('--binary-output', action='store_true',
                        help='output using binary file; faster to read by python or R')
    parser.add_argument('--binary-output-only', action='store_true',
                        help='output using binary file only without csv')
    parser.add_argument('--linear', action='store', metavar='q', type=int, nargs=1, default=1,
                        help='integer number of linear dimensions')

    args = parser.parse_args()

    if not args.cell_is_column:
        print("The rows are presumed to represent cells")
        tpm = pd.read_csv(args.input_file_name, sep='\t', index_col=0)
    else:
        print("The columns represent cells")
        tpm = pd.read_csv(args.input_file_name, sep='\t', index_col=0).T
    print(f"you have {tpm.shape[0]} cells and {tpm.shape[1]} genes.")

    if args.output_file_mask:
        print(f"The output mask is assigned to {args.output_file_mask}")
    else:
        args.output_file_mask = "".join(args.input_file_name.split('.')[:-1])
        print(f"The output mask is not assigned and is automatically set to {args.output_file_mask}")

    if not args.no_transform:
        print("The data will be log-transformed, centered, and scaled.")
        sttpm = pd.DataFrame(data=skl.preprocessing.scale(np.log2(tpm.values + 1)), index=tpm.index, columns=tpm.columns)
    else:
        print("The data will be centered, and scaled. Log-transform will not be used.")
        sttpm = pd.DataFrame(data=skl.preprocessing.scale(tpm.values), index=tpm.index, columns=tpm.columns)

    model = cyclum.PreloadCyclum(sttpm.values, q_circular=3, q_linear=1)

    pseudotime, rotation = model.fit()

    pseudotime_df = pd.DataFrame(data=pseudotime, columns=['pseudotime'], index=tpm.index)

    rotation_df = pd.DataFrame(data=rotation, index=['rotation' + str(i + 1) for i in range(rotation.shape[0])], columns=tpm.columns)

    if not args.binary_output_only:
        pseudotime_df.to_csv(args.output_file_mask + '-pseudotime.csv', sep='\t')
        rotation_df.to_csv(args.output_file_mask + '-rotation.csv', sep='\t')

    if args.binary_output:
        writer.write_df_to_binary(args.output_file_mask + '-pseudotime', pseudotime_df)
        writer.write_df_to_binary(args.output_file_mask + '-pseudotime', rotation_df)