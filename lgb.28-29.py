#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cPickle as pk
import time
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import Imputer
import warnings
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
import math
# from joblib import Parallel, delayed
import gc
import random
import operator
# from imblearn.under_sampling import RandomUnderSampler
from sklearn import model_selection, preprocessing, ensemble
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
import sys
print "脚本名：",sys.argv[0]

# dir # 
cache = 'cache_28-29'
result_path = './final_28-29/'
model_name = "model"

if not os.path.exists(result_path):
    os.mkdir(result_path)

def read_init_datafile(filepath, dtype_dict):
    dump_path = os.path.join(cache, filepath[5:-4]+'.pickle')
    if not os.path.exists(cache):
        os.mkdir(cache)
    if os.path.exists(dump_path):
        data = pd.read_pickle(dump_path)
    else:
        data = pd.read_csv(filepath, dtype = dtype_dict).to_sparse(fill_value = 0)
        data.to_pickle(dump_path)
    return data

# features #
click_size_features = ["userID_click_size","creativeID_click_size","appID_click_size","positionID_click_size",
         "camgaignID_click_size","adID_click_size","sitesetID_click_size","advertiserID_click_size",
         "appPlatform_click_size","connectionType_click_size","telecomsOperator_click_size","positionType_click_size",
         "gender_click_size","education_click_size","hometown_click_size","residence_click_size","marriageStatus_click_size",
         "age_click_size","appCategory_click_size","clickhour_min_click_size","clickhour_click_size","clickhour"]
pcvr_features = ["hcc_userID","hcc_creativeID","hcc_appID","hcc_positionID","hcc_camgaignID","hcc_adID",
         "hcc_sitesetID","hcc_advertiserID","hcc_appPlatform","hcc_connectionType","hcc_telecomsOperator",
         "hcc_positionType","hcc_gender","hcc_education","hcc_hometown","hcc_residence","hcc_marriageStatus",
         "hcc_age","hcc_appCategory","hcc_userID_creativeID","hcc_positionID_positionType",
         "hcc_positionID_advertiserID","hcc_positionID_gender","hcc_hometown_residence","hcc_gender_education",
         "hcc_positionID_marriageStatus","hcc_age_marriageStatus","hcc_positionID_age","hcc_positionID_appID",
         "hcc_positionID_hometown","hcc_positionID_telecomsOperator","hcc_positionID_creativeID",
         "hcc_positionID_education","hcc_camgaignID_connectionType","hcc_positionID_connectionType",
         "hcc_userID_adID","hcc_creativeID_connectionType","hcc_userID_connectionType","hcc_creativeID_gender",
         "hcc_positionID_userID","hcc_positionID_camgaignID","hcc_age_gender","hcc_camgaignID_age",
         "hcc_adID_connectionType","hcc_camgaignID_gender","hcc_userID_appCategory","hcc_advertiserID_connectionType",
         "hcc_positionID_adID","hcc_positionID_appCategory","hcc_positionID_haveBaby","hcc_clickhour_min","hcc_clickhour"]
rank_features = ["creativeID_rank_day","creativeID_rank","hcc_creativeID_rank_day","hcc_creativeID_rank_day_appID",
         "hcc_creativeID_rank_day_positionID","hcc_creativeID_rank_day_connectionType",
         "hcc_creativeID_rank_day_appID_positionID","hcc_creativeID_rank_day_connectionType_positionID",
         "appID_rank_day", "appID_rank"]
time_diff_features = ["min_user_click_diff","min_user_creativeid_click_diff","min_user_next_click_diff",
         "min_user_creativeid_next_click_diff","min_user_appid_click_diff","min_user_appid_next_click_diff",
         "hcc_min_user_click_diff1","hcc_min_user_next_click_diff1","hcc_min_user_creativeid_click_diff1",
         "hcc_min_user_creativeid_next_click_diff1","hcc_min_user_appid_click_diff1","hcc_min_user_appid_next_click_diff1",
         "hcc_min_user_click_diff1_positionID","hcc_min_user_click_diff1_appID","hcc_min_user_next_click_diff1_positionID",
         "hcc_min_user_next_click_diff1_appID","hcc_min_user_creativeid_click_diff1_positionID",
         "hcc_min_user_creativeid_click_diff1_appID","hcc_min_user_creativeid_next_click_diff1_positionID",
         "hcc_min_user_creativeid_next_click_diff1_appID","hcc_min_user_appid_click_diff1_positionID",
         "hcc_min_user_appid_click_diff1_appID","hcc_min_user_appid_next_click_diff1_positionID",
         "hcc_min_user_appid_next_click_diff1_appID"]
hist_features = ["appCategory1","app_actioned","a_size","user_appC_a_rate","user_appC_a","app_installed","i_size",
         "user_appC_i_rate","user_appC_i"]

features_to_use = time_diff_features + pcvr_features + rank_features + click_size_features + hist_features
print (features_to_use)


# path # 
start = time.time()
train_path = os.path.join(cache, 'train.pickle')
test_path = os.path.join(cache, 'test.pickle')
val_path = os.path.join(cache, 'val.pickle')
if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
    train = pd.read_pickle(train_path)
    test = pd.read_pickle(test_path)
    val = pd.read_pickle(val_path)
else:
    train_hist_path = 'data/train_history.28-30.csv'
    train_count_path = 'data/train_count.w7.28-30.csv'
    train_pcvr_path = 'data/train_pcvr.28-30.csv'
    test_hist_path = 'data/test_history.csv'
    test_count_path = 'data/test_count.w7.csv'
    test_pcvr_path = 'data/test_pcvr.csv'

    # read_data #
    dtype_dict = {}
    for key in features_to_use:
        dtype_dict[key] = np.float32

    start = time.time()
    train_hist = read_init_datafile(train_hist_path, dtype_dict)
    test_hist = read_init_datafile(test_hist_path, dtype_dict)
    train_hist = train_hist[hist_features]
    test_hist = test_hist[hist_features]

    train_count = read_init_datafile(train_count_path, dtype_dict)
    test_count = read_init_datafile(test_count_path, dtype_dict)
    train_count = train_count[["instanceID", "label", "day"]+click_size_features]
    test_count = test_count[["instanceID", "label", "day"]+click_size_features]
    train_hist = pd.concat([train_hist, train_count], axis=1)
    del train_count
    test_hist = pd.concat([test_hist, test_count], axis=1)
    del test_count

    train_pcvr = read_init_datafile(train_pcvr_path, dtype_dict)
    test_pcvr = read_init_datafile(test_pcvr_path, dtype_dict)
    train_pcvr = train_pcvr[pcvr_features+rank_features+time_diff_features]
    test_pcvr = test_pcvr[pcvr_features+rank_features+time_diff_features]
    train_hist = pd.concat([train_hist, train_pcvr], axis=1)
    del train_pcvr
    test_hist = pd.concat([test_hist, test_pcvr], axis=1)
    del test_pcvr

    train = train_hist[train_hist["day"]<30]
    val = train_hist[train_hist["day"]==30]
    test = test_hist
    train.to_pickle(train_path)
    test.to_pickle(test_path)
    val.to_pickle(val_path)


train, train_label, train_ids = train[features_to_use], train["label"], train["instanceID"]
val, val_label, val_ids = val[features_to_use], val["label"], val["instanceID"]

print("shape(train, val, test):", train.shape, val.shape, test.shape)
print("prob(train, val)", train_label.mean(), val_label.mean())
print('data prepare time used = {0:.0f}'.format(time.time()-start))
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.82,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 0
}
print(train.shape, val.shape, test.shape)

print(train_label.mean(), val_label.mean())

lgb_train = lgb.Dataset(train, label=train_label.tolist())
lgb_eval = lgb.Dataset(val, label=val_label.tolist(), reference=lgb_train)

print('Start training...')
# train
start = time.time()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_eval, lgb_train],
                early_stopping_rounds=10)


# test # 
y = gbm.predict(test)
res = test_ids.to_dense().to_frame()
res['prob'] = y

res.to_csv(result_path+'submission.csv', index=False)
gbm.save_model(result_path+model_name)
print('Feature importances:', list(gbm.feature_importance()))

"""
i = 0
val_sub = pd.DataFrame()
cv_scores = []
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(range(training_data.shape[0])):
    print("NO."+str(i))
    train, val = training_data.loc[dev_index,:], training_data.loc[val_index,:]
    train_label, val_label = label.loc[dev_index], label.loc[val_index]
    val_ids = train_ids[val_index]

    print(train.shape, val.shape, test.shape)

    print(train_label.mean(), val_label.mean())

    lgb_train = lgb.Dataset(train, label=train_label.tolist())
    lgb_eval = lgb.Dataset(val, label=val_label.tolist(), reference=lgb_train)

    print('Start training...')
    # train
    start = time.time()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_eval, lgb_train],
                    early_stopping_rounds=10)
    
    # val # 
    preds = gbm.predict(val)
    res = val_ids.to_dense().to_frame()
    res['prob'] = preds
    val_sub = val_sub.append(res, ignore_index=True)
    
    # test # 
    y = gbm.predict(test)
    res = test_ids.to_dense().to_frame()
    res['prob'] = y

    res.to_csv(result_path+'submission.'+str(i)+'.csv', index=False)
    gbm.save_model(result_path + model_name+str(i))
    i = i + 1
    print('train time used = {0:.0f}'.format(time.time()-start))
print('Feature importances:', list(gbm.feature_importance()))

val_sub = val_sub.sort_values(by=["instanceID"])
val_sub.to_csv(result_path+'val_sub.csv', index=False)
"""

