import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np


def take_median(df, index):
    num_trial = df['Number of Trials'].unique()[0]
    time_array = []
    # for each row in dataframe df
    for i in range(num_trial):
        t1 = df.iloc[index]['Trial' + str(i) + ' Subregion0 Executor']
        time_array.append(t1)
    return np.median(time_array)


def plot_for_gemm(path):
    df = pd.read_csv(path)

    gemm_methods_tl20 = {}
    dimensions = sorted(df["Dim1"].unique())

    for index, row in df.iterrows():
        if str(row["Implementation Name"]) + "-" + str(row["NumThreads"]) not in gemm_methods_tl20:
            gemm_methods_tl20[str(row["Implementation Name"]) + "-" + str(row["NumThreads"])] = {}
        gemm_methods_tl20[str(row["Implementation Name"]) + "-" + str(row["NumThreads"])][row["Dim1"]] = index

    my_dpi = 96
    plt.figure(figsize=(1200 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    for key, value in gemm_methods_tl20.items():
        y_values = []
        for method_key in sorted(value.keys()):
            index = value[method_key]
            y_values.append(take_median(df, index))

        plt.plot(dimensions, y_values, label=key, marker='o', linestyle='-')

    plt.yscale('log')
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution time(log)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)
    plt.savefig("s.SVG")


# first argument is csv file address for gemm
if __name__ == '__main__':
    plot_for_gemm(sys.argv[1])
