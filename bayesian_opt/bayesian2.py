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
    targets = data[:, config["train_data_len"] - 1 + config["target_len"]]
    return data, train_data, targets


# data processing
def data_processing(config):
    data, train_data, targets = get_data(config)
    sca = MinMaxScaler()
    data = sca.fit_transform(data[:, 0:8])
    train_data = data[:, 0:7]
    targets = data[:, 7]
    return sca, train_data, targets



def expect_imp(data, targets, best_val, test_pts, test_vals, best_pt,gp):
    xi = 0.01
    best_ei = None
    # gp = GaussianProcessRegressor().fit(test_pts, test_vals)
    for pt in data:
        pt = np.expand_dims(pt, axis=0)
        pt_mean, pt_std = gp.predict(pt, return_std=True)
        Z = (pt_mean - best_val - xi) / pt_std
        pt_ei = 0
        if pt_std != 0:
            st_norm = norm()
            cdf_Z = st_norm.cdf(Z)
            pdf_Z = st_norm.pdf(Z)
            pt_ei = (pt_mean - best_val - xi) * cdf_Z + (pt_std * pdf_Z)
        if best_ei == None:
            best_ei = pt_ei
            best_pt = pt
        elif pt_ei > best_ei:
            best_ei = pt_ei
            best_pt = pt
    return best_pt


def PI(data, targets, best_val, test_pts, test_vals, best_pt):
    xi = 0.0
    best_ei = None
    gp = GaussianProcessRegressor().fit(data, targets)
    for pt in data:
        pt = np.expand_dims(pt, axis=0)
        pt_mean, pt_std = gp.predict(pt, return_std=True)
        cdf_Z = norm.cdf((pt_mean - best_val) / (pt_std+1E-9))

        if best_ei == None:
            best_ei = cdf_Z
            best_pt = pt
        elif cdf_Z > best_ei:
            best_ei = cdf_Z
            best_pt = pt
    return best_pt



def UCB(x, gp, kappa):
    mean, std = gp.predict(x, return_std=True)
    upper = mean + kappa * std
    x_max = x[upper.argmax()]
    return x_max



def main(config):
    sca, train_data, targets = data_processing(config)
    best_val = -1
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(train_data, targets)
    test_pts = train_data
    test_vals = targets
    best_pt = np.asarray([[8.8, 2, 2.2, 0.06, 0.25, 132, 2200]], dtype=object)
    qq = []
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(train_data, targets)
    next_val = -1
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(train_data, targets)
    gp = GaussianProcessRegressor().fit(train_data, targets)
    for i in range(5000):
        y_min = np.min(test_vals)
        # x_suggestion = PI4(next_val, train_data, 0.01, test_pts, test_vals)
        x_suggestion = expect_imp(train_data, targets, best_val, test_pts, test_vals, best_pt, gp)
        # x_suggestion = PI3(train_data, y_min, 0.01, test_pts, test_vals)
        x_suggestion = UCB(train_data, gp, 0.2)
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
        gp = GaussianProcessRegressor().fit(train_data, targets)
        best_record = np.expand_dims(best_val, axis=0)
        result = np.append(x_suggestion, target)
        result = np.expand_dims(result, axis=0)
        result = sca.inverse_transform(result)
        result1 = result[:, -1]
        print(result, "结果")
        qq.extend(result)
        print(result1, "归一化后的结果")
        df = pd.DataFrame(qq)
        df.to_csv('bayesian_output1.csv', index=None)


if __name__ == '__main__':
    config = argsparser()
    main(config)
# print(train_data[DATA_LEN:], "训练数据")
# print(targets[DATA_LEN:], "值")
# print(np.max(targets[DATA_LEN:]))

