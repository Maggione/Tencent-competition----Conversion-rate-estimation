#!/usr/bin/env python
# coding=utf-8
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import cPickle as pk
import time
from sklearn.model_selection import train_test_split
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

cache = 'cache'

def read_init_datafile(filepath, dtype_dict):
    dump_path = os.path.join(cache, filepath[13:-5]+'.w.pickle')
    if not os.path.exists(cache):
        os.mkdir(cache)
    if os.path.exists(dump_path):
        data = pd.read_pickle(dump_path)
    else:
        data = pd.read_csv(filepath, dtype = dtype_dict).to_sparse(fill_value = 0)
        data.to_pickle(dump_path)
    return data
# features_to_use = ["creativeID_size","hcc_creativeID_label","userID_size","hcc_userID_label","positionID_size","hcc_positionID_label","connectionType_size","hcc_connectionType_label","telecomsOperator_size","hcc_telecomsOperator_label","age_size","hcc_age_label","gender_size","hcc_gender_label","education_size","hcc_education_label","marriageStatus_size","hcc_marriageStatus_label","haveBaby_size","hcc_haveBaby_label","hometown_size","hcc_hometown_label","residence_size","hcc_residence_label","sitesetID_size","hcc_sitesetID_label","positionType_size","hcc_positionType_label","adID_size","hcc_adID_label","camgaignID_size","hcc_camgaignID_label","advertiserID_size","hcc_advertiserID_label","appID_size","hcc_appID_label","appPlatform_size","hcc_appPlatform_label","appCategory_size","hcc_appCategory_label","clickhour_min_size","hcc_clickhour_min_label","clickhour_size","hcc_clickhour_label"]
features_to_use = ["creativeID_size","hcc_creativeID_label","userID_size","hcc_userID_label","positionID_size","hcc_positionID_label","connectionType_size","hcc_connectionType_label","telecomsOperator_size","hcc_telecomsOperator_label","age_size","hcc_age_label","gender_size","hcc_gender_label","education_size","hcc_education_label","marriageStatus_size","hcc_marriageStatus_label","haveBaby_size","hcc_haveBaby_label","hometown_size","hcc_hometown_label","residence_size","hcc_residence_label","sitesetID_size","hcc_sitesetID_label","positionType_size","hcc_positionType_label","adID_size","hcc_adID_label","camgaignID_size","hcc_camgaignID_label","advertiserID_size","hcc_advertiserID_label","appID_size","hcc_appID_label","appPlatform_size","hcc_appPlatform_label","appCategory_size","hcc_appCategory_label","clickhour_min_size","hcc_clickhour_min_label","clickhour_size","hcc_clickhour_label","positionID_connectionType_size","hcc_positionID_connectionType_label","positionID_appID_size","hcc_positionID_appID_label","positionID_advertiserID_size","hcc_positionID_advertiserID_label","hometown_residence_size","hcc_hometown_residence_label","age_education_size","hcc_age_education_label","appCategory_age_size","hcc_appCategory_age_label","appCategory_gender_size","hcc_appCategory_gender_label","appCategory_education_size","hcc_appCategory_education_label","appCategory_marriageStatus_size","hcc_appCategory_marriageStatus_label","appCategory_haveBaby_size","hcc_appCategory_haveBaby_label","appCategory_hometown_size","hcc_appCategory_hometown_label","appCategory_residence_size","hcc_appCategory_residence_label","appCategory1","install_size","appCategory_hist_rate","appCategory_hist","action_7_size","appCategory_7_rate","appCategory_7","cvr_pre_count","app_fix","pv_pre_count","pv_pre_count_day","hcc_cvr_pre_count_label","hcc_pv_pre_count_label","hcc_pv_pre_count_day_label","hcc_cvr_pre_count-app_label","hcc_cvr_pre_count-pos_c_type_label","hcc_pv_pre_count-pos_c_type_label","hcc_pv_pre_count_day-pos_c_type_label","hcc_pv_pre_count-app_label","hcc_pv_pre_count_day-app_label","hcc_app-pos_label","hcc_app-pos-pv_pre_label","hcc_app-pos-pv_day_pre_label","hcc_app-pos-cv_pre_label"]
print (features_to_use)

dtype_dict = {}
for key in features_to_use:
    dtype_dict[key] = np.float32
# read_data #
# train24 = pd.read_csv("./final_data/train_final.csv.24.half")
# train25 = pd.read_csv("./final_data/train_final.csv.25.half")
# train26 = pd.read_csv("./final_data/train_final.csv.26.half")
# train27 = pd.read_csv("./final_data/train_final.csv.27.half")
# train28 = pd.read_csv("./final_data/train_final.csv.28.half")
# train29 = pd.read_csv("./final_data/train_final.csv.29.half")
# train30 = pd.read_csv("./final_data/train_final.csv.30.half")
# test = pd.read_csv("./final_data/test_final.csv")

def data_get(train_set):
    train = pd.DataFrame()
    label = []
    train_ids = []
    if 24 in train_set:    
        # train24 = pd.read_csv("./final_data/train_final.csv.24.half",dtype = dtype_dict).to_sparse(fill_value = 0)
        # train24 = train24[features_to_use]
        train24 = read_init_datafile('./final_data/train_final.csv.24.half', dtype_dict)
        label24 = train24["label"].tolist()
        train_ids24 = train24["instanceID"].tolist()
        train24 = train24[features_to_use]
        train = train.append(train24, ignore_index=True)
        label.extend(label24)
        train_ids.extend(train_ids24)
        del train24, label24, train_ids24
    if 25 in train_set:    
        # train25 = pd.read_csv("./final_data/train_final.csv.25.half",dtype = dtype_dict).to_sparse(fill_value = 0)
        train25 = read_init_datafile('./final_data/train_final.csv.25.half', dtype_dict)
        # train25 = train25[features_to_use]
        label25 = train25["label"].tolist()
        train_ids25 = train25["instanceID"].tolist()
        train25 = train25[features_to_use]
        train = train.append(train25, ignore_index=True)
        label.extend(label25)
        train_ids.extend(train_ids25)
        del train25, label25, train_ids25
    if 26 in train_set:    
        # train26 = pd.read_csv("./final_data/train_final.csv.26.half",dtype = dtype_dict).to_sparse(fill_value = 0)
        train26 = read_init_datafile('./final_data/train_final.csv.26.half', dtype_dict)
        # train26 = train26[features_to_use]
        label26 = train26["label"].tolist()
        train_ids26 = train26["instanceID"].tolist()
        train26 = train26[features_to_use]
        train = train.append(train26, ignore_index=True)
        label.extend(label26)
        train_ids.extend(train_ids26)
        del train26, label26, train_ids26
    if 27 in train_set:    
        # train27 = pd.read_csv("./final_data/train_final.csv.27.half",dtype = dtype_dict).to_sparse(fill_value = 0)
        train27 = read_init_datafile('./final_data/train_final.csv.27.half', dtype_dict)
        # train27 = train27[features_to_use]
        label27 = train27["label"].tolist()
        train_ids27 = train27["instanceID"].tolist()
        train27 = train27[features_to_use]
        train = train.append(train27, ignore_index=True)
        label.extend(label27)
        train_ids.extend(train_ids27)
        del train27, label27, train_ids27
    if 28 in train_set:    
        # train28 = pd.read_csv("./final_data/train_final.csv.28.half",dtype = dtype_dict).to_sparse(fill_value = 0)
        train28 = read_init_datafile('./final_data/train_final.csv.28.half', dtype_dict)
        # train28 = train28[features_to_use]
        label28 = train28["label"].tolist()
        train_ids28 = train28["instanceID"].tolist()
        train28 = train28[features_to_use]
        train = train.append(train28, ignore_index=True)
        label.extend(label28)
        train_ids.extend(train_ids28)
        del train28, label28, train_ids28
    if 29 in train_set:    
        # train29 = pd.read_csv("./final_data/train_final.csv.29.half",dtype = dtype_dict).to_sparse(fill_value = 0)
        train29 = read_init_datafile('./final_data/train_final.csv.29.half', dtype_dict)
        # train29 = train29[features_to_use]
        label29 = train29["label"].tolist()
        train_ids29 = train29["instanceID"].tolist()
        train29 = train29[features_to_use]
        train = train.append(train29, ignore_index=True)
        label.extend(label29)
        train_ids.extend(train_ids29)
        del train29, label29, train_ids29
    if 30 in train_set:    
        # train30 = pd.read_csv("./final_data/train_final.csv.30.half",dtype = dtype_dict).to_sparse(fill_value = 0)
        train30 = read_init_datafile('./final_data/train_final.csv.30.half', dtype_dict)
        # train30 = train30[features_to_use]
        label30 = train30["label"].tolist()
        train_ids30 = train30["instanceID"].tolist()
        train30 = train30[features_to_use]
        train = train.append(train30, ignore_index=True)
        label.extend(label30)
        train_ids.extend(train_ids30)
        del train30, label30, train_ids30
    return train, np.array(label), train_ids

start = time.time()
train_set = [26,27,28,29,30]
val_set = 25
print (train_set, val_set)
# train data prepare #
train, train_label, train_ids = data_get(train_set)
print('train prepare time used = {0:.0f}'.format(time.time()-start))

# val data prepare #
val = pd.read_csv("./final_data/train_final.csv."+str(val_set)+".half",dtype = dtype_dict).to_sparse(fill_value = 0)
val, val_label, val_ids = val[features_to_use], val["label"], val["instanceID"]

# test data prepare #
test = pd.read_csv("./final_data/test_final.csv", dtype = dtype_dict).to_sparse(fill_value = 0)        
test, test_label, test_ids = test[features_to_use], test["label"], test["instanceID"]

result_path = './final_model1/w_'
model_name = "final1_w"

print(train.shape, val.shape, test.shape)

print(train_label.mean(), val_label.mean())

dtrain = xgb.DMatrix(train, label=train_label)
del train
dtest  = xgb.DMatrix(val, label=val_label)
del val

sub_training_data_ = xgb.DMatrix(test)

param = {'learning_rate' : 0.05, 'n_estimators': 1000, 'max_depth': 7, 
         'min_child_weight': 5, 'gamma': 0, 'subsample':  0.9, 'colsample_bytree':  0.82,
         'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic', 
         'booster': 'gbtree'}

print(param)
    
num_round = 1000
param['nthread'] = 128
param['eval_metric'] = "auc"
plst = param.items()
plst += [('eval_metric', 'logloss')]

# evallist = [(dtest, 'eval')]
evallist = [(dtrain,'train'), (dtest, 'test')]
bst=xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=20)
"""
# val # 
preds = bst.predict(dtest)
res = val_ids.to_frame()
res['prob'] = preds
val_sub = val_sub.append(res, ignore_index=True)
cv_scores.append(log_loss(y_test, preds))
"""
# test # 
y = bst.predict(sub_training_data_)
res = test_ids.to_dense().to_frame()
res['prob'] = y

res.to_csv(result_path+'submission'+'.csv', index=False)
bst.save_model(result_path + model_name)

print('train time used = {0:.0f}'.format(time.time()-start))


"""
# random_split
# train = train.ix[train["day"]<30]
training_data, label, train_ids = train, train["label"], train["instanceID"]
training_data = np.array(training_data[features_to_use])
X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.1, random_state=0)
"""

"""
for i in range(24,30):
    print ("val:",i)
    train_data = train.ix[train["day"]!=i]
    y_train, train_ids = train_data["label"], train_data["instanceID"]
    val_data = train.ix[train["day"] == i]
    y_test, val_ids = val_data["label"], val_data["instanceID"]

    X_train = np.array(train_data[features_to_use])
    X_test = np.array(val_data[features_to_use])
    
    print X_train.shape, X_test.shape
    print(y_train.mean(), y_test.mean())

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    sub_training_data_ = xgb.DMatrix(sub_training_data)

    param = {'learning_rate' : 0.05, 'n_estimators': 1000, 'max_depth': 7, 
             'min_child_weight': 5, 'gamma': 0, 'subsample':  0.9, 'colsample_bytree':  0.82,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic', 
             'booster': 'gbtree'}

    print(param)
        
    num_round = 1000
    param['nthread'] = 128
    param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]

    # evallist = [(dtest, 'eval')]
    evallist = [(dtrain,'train'), (dtest, 'test')]
    bst=xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=20)

    # val # 
    preds = bst.predict(dtest)
    res = val_ids.to_frame()
    res['prob'] = preds
    val_sub = val_sub.append(res, ignore_index=True)
    cv_scores.append(log_loss(y_test, preds))

    # test # 
    y = bst.predict(sub_training_data_)
    res = instanceIDs.to_frame()
    res['prob'] = y

    res.to_csv(result_path+'submission_fix'+str(i)+'.csv', index=False)
    bst.save_model(result_path + model_name + str(i))

print (cv_scores)
val_sub.to_csv(result_path+'val_sub_fix.csv', index=False)
"""

"""
#trainning
i = 0
val_sub = pd.DataFrame()
cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(range(training_data.shape[0])):
    print("NO."+str(i))
    dev_X, val_X = training_data[dev_index,:], training_data[val_index,:]
    dev_y, val_y = label[dev_index], label[val_index]
    val_ids = train_ids[val_index]
    dtrain = xgb.DMatrix(dev_X, label=dev_y)
    dtest  = xgb.DMatrix(val_X, label=val_y)
    evallist = [(dtrain,'train'), (dtest, 'test')]
    bst=xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=20)

    # val
    preds = bst.predict(dtest)
    res = val_ids.to_frame()
    res['prob'] = preds
    val_sub = val_sub.append(res, ignore_index=True)
    cv_scores.append(log_loss(val_y, preds))

    # test
    y = bst.predict(sub_training_data_)
    res = instanceIDs.to_frame()
    res['prob'] = y
    
    res.to_csv(result_path+'submission'+str(i)+'.csv', index=False)
    bst.save_model(result_path+model_name+str(i))
    i = i + 1
print (cv_scores)
val_sub.to_csv(result_path+'val_sub.csv', index=False)
"""

def ceate_feature_map(features):
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

        outfile.close()


ceate_feature_map(features_to_use)
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv(result_path + "hccimportance.csv", index = False, encoding = "utf-8")
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig(result_path+'hccfeature_importance_xgb.png')

