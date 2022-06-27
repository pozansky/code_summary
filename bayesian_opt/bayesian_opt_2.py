import argparse
import pandas as pd
import numpy as np
from zoopt import Dimension, Objective, Parameter, ExpOpt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from scipy.stats import norm

def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("zoopt search")
    parser.add_argument("--data_path", default="dataset.xlsx", type=str)
    parser.add_argument("--train_data_len", default=7, type=int)
    parser.add_argument("--target_len", default=1, type=int)
    return parser.parse_known_args()[0].__dict__

def get_data(config):
    if config["data_path"] == None or config["data_path"] == "":
        raise Exception("data path cannot be empty")
    data = pd.read_excel(config["data_path"])
    data = data.values
    train_data = data[:, 0:config["train_data_len"]]
    targets = data[:, config["train_data_len"]-1+config["target_len"]]
    return data, train_data, targets


# data processing
def data_processing(config):
    data, train_data, targets = get_data(config)
    sca = MinMaxScaler()
    data = sca.fit_transform(data[:, 0:8])
    train_data = data[:, 0:7]
    targets = data[:, 7]
    return sca, train_data, targets

def get_param(new_pt, data, target, xi, best_pt):
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(data, target)
    gp = GaussianProcessRegressor().fit(data, target)
    new_pt = np.expand_dims(new_pt, axis=0)
    mean, std = gp.predict(new_pt, return_std=True)
    Z = 0
    if best_pt.ndim == 1:
        best_pt = np.expand_dims(best_pt, axis=0)
    if std != 0:
        best_val = xgb_model.predict(best_pt)
        Z = (mean - best_val - xi) / std
    return [mean, std, Z]

def expect_imp(data, best_val, test_pts, test_val, best_pt):
    xi = 0.01
    best_ei = None
    for pt in data:
        pt_mean, pt_std, Z = get_param(pt, test_pts, test_val, xi, best_pt)
        pt_ei = 0
        if pt_std != 0:
            st_norm = norm()
            cdf_Z = st_norm.cdf(Z)
            pdf_Z = st_norm.pdf(Z)
            pt_ei = (pt_mean - best_val - xi) * cdf_Z + (pt_std * pdf_Z)
        if best_ei == None:
            best_ei = pt_ei
            best_pt = pt
        elif pt_ei > best_ei :
            best_ei = pt_ei
            best_pt = pt

    return best_pt

def UCB(x, gp, kappa):
    mean, std = gp.predict(x, return_std=True)
    return mean + kappa * std

def PI(x, y_min, gp):
    xi = 0.01
    best_ei = None
    for pt in x:
        pt_mean, pt_std, Z = get_param(pt,  xi, y_min, gp)
        pt_pi = 0
        if pt_std != 0:
            st_norm = norm()
            cdf_Z = st_norm.cdf(Z)
        if best_ei == None:
            best_pi = cdf_Z
            best_pt = pt
        elif pt_pi > best_pi:
            best_pi = pt_pi
            best_pt = pt
    return best_pt


def main(config):
    sca, train_data, targets = data_processing(config)
    best_val = -1
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(train_data, targets)
    next_val = -1
    test_pts = np.asarray([[7.4, 0, 0.2, 0, 0.1, 150, 226]], dtype=object)
    test_vals = np.asarray([next_val])
    best_pt = np.asarray([[7.4, 0, 0.2, 0, 0.1, 150, 226]], dtype=object)
    for i in range(300):
        y_min = np.min(targets)
        print("kk")
        x_suggestion = expect_imp(train_data, y_min, test_pts, test_vals, best_pt)
        print("qq")
        x_suggestion = np.expand_dims(x_suggestion, axis=0)
        test_pts = np.r_[test_pts, x_suggestion]
        next_val = xgb_model.predict(x_suggestion)
        test_vals = np.append(test_vals, next_val)
        x_suggestion = np.array(x_suggestion, dtype=object)
        train_data = np.r_[train_data, x_suggestion]
        target = xgb_model.predict(x_suggestion)
        if target > best_val:
            best_val = target
            best_pt = x_suggestion
        targets = np.r_[targets, target]
        print(x_suggestion, "x_n+1")
        # print(target, "target")
        # print(best_val, "best_val")
        best_record = np.expand_dims(best_val, axis=0)
        result = np.append(x_suggestion, target)
        result = np.expand_dims(result, axis=0)
        result = sca.inverse_transform(result)
        result = result[:, -1]
        print(result, "归一化后的结果")

if __name__ == '__main__':
    config = argsparser()
    main(config)
# print(train_data[DATA_LEN:], "训练数据")
# print(targets[DATA_LEN:], "值")
# print(np.max(targets[DATA_LEN:]))