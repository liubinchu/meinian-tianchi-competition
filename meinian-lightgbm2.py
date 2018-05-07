
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
import xgboost as xgb

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train_X.values, train_Y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


train=pd.read_csv("./train_set_new.csv")
test=pd.read_csv("./test_set_new.csv")

train_num=train.select_dtypes(float)
train_X=train_num.iloc[:,5:88]


del train_X['669001']
del train_X['31']

train_X=train_X.fillna(train_X.median())

test_new = test.loc[:, train_X.columns]  # 对test进行处理和train一样
test_new = test_new.fillna(test_new.median())  # test集缺失取中位数

for i in range(0,5):
    train_Y = train_num.iloc[:, i]
    train_Y = train_Y.fillna(train_Y.median())

    # model_lgb
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.01, n_estimators=9200,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    model_lgb.fit(train_X, train_Y)  # 训练
    lgb_train_pred = model_lgb.predict(train_X)  # 获得训练模型
    lgb_test = model_lgb.predict(test_new)  # 得到test的预测结果
    lgb_test_pred = lgb_test.reshape(-1, 1)
    if i!=0:
      lgb_pred= np.concatenate((lgb_pred, lgb_test_pred), axis=1)
    else:
      lgb_pred=lgb_test_pred

for i in range(0,5):
    train_Y = train_num.iloc[:, i]
    train_Y = train_Y.fillna(train_Y.median())
    # model_xgb
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.7817, n_estimators=300,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1,
                                 random_state=7, nthread=-1)

    model_xgb.fit(train_X, train_Y)  # 训练
    xgb_train_pred = model_xgb.predict(train_X)  # 获得训练模型
    xgb_test = model_xgb.predict(test_new)  # 得到test的预测结果
    xgb_test_pred = xgb_test.reshape(-1, 1)
    if i != 0:
        xgb_pred = np.concatenate((xgb_pred, xgb_test_pred), axis=1)
    else:
        xgb_pred = xgb_test_pred

pred=lgb_pred*0.7+xgb_pred*0.3;
final=pd.DataFrame(pred)
 #final=pd.DataFrame({u"收缩压":pred[:,0],u"舒张压":pred[:,1],u"血清甘油三酯":pred[:,2],u"血清高密度脂蛋白":pred[:,3],u"血清低密度脂蛋白":pred[:,4]})
file=pd.concat([test.vid,final],axis=1)
file.to_csv("meinian4.csv", index = False)



