import random

import numpy as np
import pandas as pd
np.random.seed(100)
def range_set(train_data):
    min_record = []
    max_record = []
    for i in range(7):
        min = np.min(train_data[:, i])
        min_record.append(min)
        max = np.max(train_data[:, i])
        max_record.append(max)
    dim_range = [[0 for j in range(2)] for i in range(7)]
    for i in range(7):
        dim_range[i][0] = min_record[i]
        dim_range[i][1] = max_record[i]
    return dim_range

def get_data():
    data = pd.read_excel("dataset.xlsx")
    data = data.values
    train_data = data[:, 0:7]
    return train_data

def compute(number):
    train_data = get_data()
    dim_range = range_set(train_data)
    a = []
    for i in range(7):
        d1 = np.random.uniform(dim_range[i][0], dim_range[i][1], (1, number))
        a.append(d1)
    qq = np.r_[a[0], a[1], a[2], a[3], a[4], a[5], a[6]]
    # qq = np.r_[(a[i] for i in range(7))]
    qq = qq.T
    return qq


