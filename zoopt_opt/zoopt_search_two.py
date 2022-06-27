import argparse
import pandas as pd
import numpy as np
from zoopt import Dimension, Objective, Parameter, ExpOpt
from sklearn.ensemble import GradientBoostingRegressor

def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("zoopt search")
    parser.add_argument("--data_path", default="dataset.xlsx", type=str)
    parser.add_argument("--train_data_len", default=7, type=int)
    parser.add_argument("--target_len", default=1, type=int)
    parser.add_argument("--init_samples", default=True, type=bool)
    parser.add_argument("--repeat_times", default=10, type=int, help="repeat times in every experiment")
    parser.add_argument("--mean_or_all", default=True, type=bool, help="Whether to calculate the mean value")
    return parser.parse_known_args()[0].__dict__

def myobjective(config):
    param_list = config.get_x()
    param_list = np.expand_dims(param_list, axis=0)
    target_hat = rf.predict(param_list)
    return -target_hat

def get_data(config):
    if config["data_path"] == None or config["data_path"] == "":
        raise Exception("data path cannot be empty")
    data = pd.read_excel(config["data_path"])
    data = data.values
    train_data = data[:, 0:config["train_data_len"]]
    targets = data[:, config["train_data_len"]-1+config["target_len"]]
    return train_data, targets

# Range settings, based on the maximum and minimum values in the original data
def range_set(config, train_data):
    min_record = []
    max_record = []
    for i in range(config["train_data_len"]):
        min = np.min(train_data[:, i])
        min_record.append(min)
        max = np.max(train_data[:, i])
        max_record.append(max)
    dim_range = [[0 for j in range(2)] for i in range(config["train_data_len"])]
    for i in range(config["train_data_len"]):
        dim_range[i][0] = min_record[i]
        dim_range[i][1] = max_record[i]
    return dim_range

def seach_zoopt(config):
    dim = Dimension(config["train_data_len"], dim_range, [True] * config["train_data_len"])
    objective = Objective(myobjective, dim)
    budget = 200 * config["train_data_len"]
    if config["init_samples"] == True:
        sample_data, _ = get_data(config)
        parameter = Parameter(budget=budget, init_samples=sample_data, sequential=True, ponss=False)
    else:
        parameter = Parameter(budget=budget, sequential=True, ponss=False)
    # searching progress
    solution_list = ExpOpt.min(objective, parameter, repeat=config["repeat_times"])
    x_record = []
    for solution in solution_list:
        print(solution.get_x(), solution.get_value())
        x_record.append(solution.get_x())

    x_record = np.asarray(x_record)
    result = rf.predict(x_record)
    x_record = np.column_stack((x_record, result))
    if config["mean_or_all"] == True:
        x_record = np.expand_dims(np.mean(x_record, axis=0), axis=0)
    x_record = np.round(x_record.astype('float'), 2)
    return x_record

def write_excel(data):
    dataframe = pd.DataFrame(data)
    data = pd.read_excel('dataset.xlsx')
    col_name = data.columns.tolist()  # 将数据框的列名全部提取出来存放在列表里
    print(col_name)
    dataframe.to_excel("实验数据.xls", sheet_name="data", header=0)
    df = pd.read_excel('实验数据.xls', header=None, names=col_name[:-1])
    df.to_csv('zoopt_output.csv', index=None)

def fitmodel():
    clf = GradientBoostingRegressor()
    rf = clf.fit(train_data, targets)
    return rf

if __name__ == '__main__':
    config = argsparser()
    train_data, targets = get_data(config)
    rf = fitmodel()
    dim_range = range_set(config, train_data)
    data = seach_zoopt(config)
    write_excel(data)