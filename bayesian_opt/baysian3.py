import argparse
import pandas as pd
import numpy as np
from zoopt import Dimension, Objective, Parameter, ExpOpt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from scipy.stats import norm
from code_summary.bayesian_opt import generate_re


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
    data = data[:, 0:8]
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    # sca = MinMaxScaler()
    # data = sca.fit_transform(data[:, 0:8])
    data = (data-min)/(max-min)
    train_data = data[:, 0:7]
    targets = data[:, 7]
    return train_data, targets, min, max



def PI(data, targets, best_val, test_pts, test_vals, best_pt, gp):
    xi = 0.01
    best_ei = None
    # gp = GaussianProcessRegressor().fit(test_pts, test_vals)
    for pt in data:
        pt = np.expand_dims(pt, axis=0)
        pt_mean, pt_std = gp.predict(pt, return_std=True)
        Z = (pt_mean - best_val - xi) / pt_std
        if pt_std != 0:
            st_norm = norm()
            cdf_Z = st_norm.cdf(Z)
        if best_ei == None:
            best_ei = cdf_Z
            best_pt = pt
        elif cdf_Z > best_ei:
            best_ei = cdf_Z
            best_pt = pt
    return best_pt


def surrogate(model, X):
    # catch any warning generated when making a prediction
    return model.predict(X, return_std=True)


# probability of improvement acquisition function
def acquisition_EI(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = np.max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    # mu = mu[:, 0]
    # calculate the probability of improvement
    # probs = norm.cdf((mu - best -0.01) / (std + 1E-9))
    a = (mu - best)
    print("*************************************")
    print(a)
    z = a / std + 1E-9
    return a * norm.cdf(z) + std * norm.pdf(z)
    # return probs

def acquisition_PI(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = np.max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    # mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((best - mu) / (std + 1E-9))
    return probs

def acquisition_UCB(X, Xsamples, model):

    mu, std = surrogate(model, Xsamples)
    return mu + 0.01 * std


# optimize the acquisition function
def opt_acquisition(X, y, model):
    # random search, generate random samples
    # Y = X
    # np.random.shuffle(Y)
    # idx_shuffle = np.random.choice(len(X), len(X))
    # Xsamples = X

    # Xsamples = np.asarray([[8.8, 2, 2.2, 0.06, 0.25, 132, 2200]], dtype=object)
    # Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    Xsamples = generate_re.compute(100)
    # Xsamples = X
    x_min = np.min(Xsamples, axis=0)
    x_max = np.max(Xsamples, axis=0)
    Xsamples = (Xsamples - x_min) / (x_max - x_min)
    Xsamples = X
    scores = acquisition_EI(X, Xsamples, model)
    # locate the index of the largest scores
    x_max = Xsamples[scores.argmax()]
    print("bb", scores.argmax())
    return [x_max]


def main(config):

    train_data, targets, min, max = data_processing(config)
    best_val = -1
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(train_data, targets)
    targets = xgb_model.predict(train_data)
    model = GaussianProcessRegressor()
    # fit the model
    model.fit(train_data, targets)
    # plot before hand
    # perform the optimization process

    for i in range(100000):
        # select the next point to sample
        x = opt_acquisition(train_data, targets, model)
        # Xsamples = np.r_[Xsamples, x]
        # sample the point
        actual = xgb_model.predict(x)
        # print(actual, x)
        # summarize the finding
        # est, _ = surrogate(model, x)

        # add the data to the dataset
        train_data = np.vstack((train_data, x))
        print(train_data.shape)
        targets = np.r_[targets, actual]
        # update the model
        model.fit(train_data, targets)
        result = np.append(x, actual)
        result = np.expand_dims(result, axis=0)
        result = result * (max - min) + min
        result1 = result[:, -1]
        print(result, "结果")
        print(result1, "归一化后的结果")

if __name__ == '__main__':
    config = argsparser()
    main(config)
# print(train_data[DATA_LEN:], "训练数据")
# print(targets[DATA_LEN:], "值")
# print(np.max(targets[DATA_LEN:]))

