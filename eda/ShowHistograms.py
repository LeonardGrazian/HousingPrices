

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from FillMissing import custom_fillna, create_ordinals


ROOT_DIR = '/home/leonard/Desktop/Workspace/Kaggle/HousingPrices'

def make_hist(data, target, col_name, show=False):
    fig, ax1 = plt.subplots()
    ax1.set_title(col_name)
    ax1.hist(data)

    hist, bin_edges = np.histogram(data)
    binned_target = []
    bin_middles = []
    for i in range(len(bin_edges) - 1):
        bin_low = bin_edges[i]
        bin_high = bin_edges[i + 1]
        bin_middles.append((bin_low + bin_high) / 2)
        bin_mask = np.logical_and(data > bin_low, data < bin_high)
        binned_target.append( np.mean(target[bin_mask]) )

    ax2 = ax1.twinx()
    ax2.scatter(bin_middles, binned_target, marker='s', s=128, color='red')
    plt.savefig('{}/plots/histograms/{}_hist.png'.format(ROOT_DIR, col_name))
    if show:
        plt.show()
    plt.close()


def make_categorical_hist(data, target, col_name, show=False):
    unique_vals = np.unique(data)
    count_dict = {uval: (col_data == uval).sum() for uval in unique_vals}
    keys = []
    counts = []
    target_vals = []
    for key in count_dict:
        keys.append(key)
        counts.append(count_dict[key])

        key_mask = (data == key)
        target_vals.append(np.mean(target[key_mask]))

    fig, ax1 = plt.subplots()
    ax1.set_title(col_name)
    ax1.bar(keys, counts)
    ax2 = ax1.twinx()
    ax2.scatter(keys, target_vals, marker='s', s=128, color='red')
    plt.savefig('{}/plots/histograms/{}_hist.png'.format(ROOT_DIR, col_name))
    if show:
        plt.show()
    plt.close()


train_data = pd.read_csv('{}/data/train.csv'.format(ROOT_DIR), index_col=0)
train_data = custom_fillna(train_data)
train_data = create_ordinals(train_data)
target = train_data['SalePrice']

show = False
for col in train_data.select_dtypes(exclude='object'):
    col_data = train_data[col]
    make_hist(col_data, target, col, show=show)

for col in train_data.select_dtypes(include='object'):
    col_data = train_data[col]
    make_categorical_hist(col_data, target, col, show=show)
