from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np
from scipy.stats import norm
import xgboost as xgb
import random
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process.kernels import RBF
import itertools
# 获取数据
DATA_LEN = 1016
x_pts = np.linspace(0, 4, 30)
y_pts = np.linspace(-2, 2, 30)
from sklearn.preprocessing import MinMaxScaler
# data points
data_pts = list(itertools.product(x_pts,y_pts))

def get_data():

    data = pd.read_excel('dataset.xlsx')
    data = data.values
    # index = np.random.permutation(1015)
    # data = data[index]

    # data = sca.fit_transform(data[:, 0:8])
    train_data = data[:, 0:7]
    targets = data[:, 7]
    return train_data[:DATA_LEN], targets[:DATA_LEN]

# 定义先验
# GP = GaussianProcessRegressor(kernel=RBF)
# train_data, targets = get_data()
# GP.fit(train_data, targets)
# dim_size = 7

def target_f(x):
    xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
    xgb_model.fit(train_data, targets)
    return xgb_model.predict(x)

# train_data, targets = get_data()
sca = MinMaxScaler()
data = pd.read_excel('dataset.xlsx')
data = data.values
# index = np.random.permutation(1015)
# data = data[index]
# data = sca.fit_transform(data[:, 0:8])
train_data = data[:, 0:7]
targets = data[:, 7]

# xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)
# xgb_model.fit(train_data, targets)

def get_param(new_pt, data, target, xi, xgb_model):
    gp = GaussianProcessRegressor().fit(data, target)
    new_pt = np.expand_dims(new_pt, axis=0)
    mean, std = gp.predict(new_pt, return_std=True)
    Z = 0
    if std != 0:
        best_val = xgb_model.predict(best_pt)
        Z = (mean - best_val -xi) / std

    return [mean, std, Z]

def expect_imp(data, best_val, test_pts, test_val, xgb_model):
    xi = 0.0
    best_ei = None
    for pt in data:
        pt_mean, pt_std, Z = get_param(pt, test_pts, test_val, xi, xgb_model)
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
            # if best_ei > 60:
            #     qq.append(best_pt)
            # print(best_ei)

    return best_pt

def PI(data, best_val, test_pts, test_val, xgb_model):
    xi = 0.0
    best_ei = None
    for pt in data:
        pt_mean, pt_std, Z = get_param(pt, test_pts, test_val, xi, xgb_model)
        pt_ei = 0
        if pt_std != 0:
            st_norm = norm()
            cdf_Z = st_norm.cdf(Z)

        if best_ei == None:
            best_ei = cdf_Z
            best_pt = pt
        elif cdf_Z > best_ei :
            best_ei = cdf_Z
            best_pt = pt
            # if best_ei > 60:
            #     qq.append(best_pt)
            # print(best_ei)

    return best_pt

def UCB(x, gp, kappa):
    mean, std = gp.predict(x, return_std=True)
    return mean + kappa * std


best_val = -1


xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1)




GP = GaussianProcessRegressor()
next_pt = np.asarray([[7.4, 0, 0.2, 0, 0.1, 150, 226]], dtype=object)
best_pt = np.asarray([[7.4, 0, 0.2, 0, 0.1, 150, 226]], dtype=object)
# next_pt =  (2.0,0)
next_val = 88
test_pts = np.asarray([[7.4, 0, 0.2, 0, 0.1, 150, 226]], dtype=object)
test_vals = np.asarray([next_val])
xgb_model.fit(train_data, targets)
for i in range(300):

    # train_data, targets = get_data()
    # GP = GP.fit(train_data, targets)
    y_max = np.min(targets)
    x_suggestion = expect_imp(train_data, y_max, test_pts, test_vals, xgb_model)
    x_suggestion = np.expand_dims(x_suggestion, axis=0)
    test_pts = np.r_[test_pts, x_suggestion]
    next_val = xgb_model.predict(x_suggestion)
    test_vals = np.append(test_vals, next_val)

    x_suggestion = np.array(x_suggestion, dtype=object)
    # x_suggestion = np.expand_dims(x_suggestion, axis=0)
    train_data = np.r_[train_data, x_suggestion]
    target = xgb_model.predict(x_suggestion)
    if target > best_val:
        best_val = target
        best_pt = x_suggestion
    targets = np.r_[targets, target]
    # print(x_suggestion, "x_n+1")
    # print(target, "target")
    # print(best_val, "best_val")
    best_record = np.expand_dims(best_val, axis=0)
    result = np.append(x_suggestion, target)
    result = np.expand_dims(result, axis=0)
    # result = sca.inverse_transform(result)
    result = result[:, -1]
    print(result, "归一化后的结果")

print(train_data[DATA_LEN:], "训练数据")
print(targets[DATA_LEN:], "值")
print(np.max(targets[DATA_LEN:]))


