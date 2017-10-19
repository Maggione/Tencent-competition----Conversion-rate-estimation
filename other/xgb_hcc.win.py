#!/usr/bin/env python
# coding=utf-8
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import cPickle as pk
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
# 路径变量
ad_path = "../data/ad.csv"
app_categories_path = "../data/app_categories.csv"
position_path = "../data/position.csv"
train_path = "../data/train.csv"
test_path = "../data/test.csv"
user_app_actions_path = "../data/user_app_actions.csv"
user_installedapps_path = "../data/user_installedapps.csv"
user_path = '../data/user.csv'

# 初始数据
cache = 'cache'
user = None
ad = None
app_categories = None
position = None
user_app_actions = None
user_installedapps = None
user_installedapps_if = None
train_ = None
test_ = None


# 工具函数
def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=50) as parallel:
        retLst = parallel( delayed(func)(group) for name, group in dfGrouped )
        return pd.concat(retLst)
    
def read_init_datafile(filepath):
    dump_path = os.path.join(cache, filepath[8:-4]+'.pickle')
    if not os.path.exists(cache):
        os.mkdir(cache)
    if os.path.exists(dump_path):
        data = pd.read_pickle(dump_path)
    else:
        data = pd.read_csv(filepath)
        data.to_pickle(dump_path)
    return data

def init_data():
    global user ,ad, app_categories, position, user_app_actions, user_installedapps, user_installedapps_if, train_, test_
    
    user = read_init_datafile(user_path)
    ad = read_init_datafile(ad_path)
    app_categories = read_init_datafile(app_categories_path)
    position = read_init_datafile(position_path)
    user_app_actions = read_init_datafile(user_app_actions_path)
    user_installedapps = read_init_datafile(user_installedapps_path)
    user_installedapps_if = user_installedapps.groupby(['userID', 'appID']).size().reset_index().rename(columns={0: 'install_if'})

    train_ = read_init_datafile(train_path)
    test_ = read_init_datafile(test_path)
def make_train_set():
    dump_path = "train_set_%s" %('20160524_')
    if os.path.exists(dump_path):
        train = pd.read_pickle(dump_path)
    else:
        train = pd.merge(train_, user, how='left', on='userID')
        train = pd.merge(train, ad, how='left', on='creativeID')
        train = pd.merge(train, app_categories, how='left', on='appID')
        train = pd.merge(train, position, how='left', on='positionID')
        #train = pd.merge(train, user_installedapps_if , how='left', on=['userID', 'appID'])
        #train['install_if'] = train['install_if'].fillna(0)
        #train.to_pickle(dump_path)
    return train
    
def make_test_set():
    dump_path = 'test_set_%s' %('20160524_')
    if os.path.exists(dump_path):
        test = pd.read_pickle(dump_path)
    else:
        test = pd.merge(test_, user, how='left', on='userID')
        test = pd.merge(test, ad, how='left', on='creativeID')
        test = pd.merge(test, app_categories, how='left', on='appID')
        test = pd.merge(test, position, how='left', on='positionID')
        #test = pd.merge(test, user_installedapps_if , how='left', on=['userID', 'appID'])
        #test['install_if']  = test['install_if'].fillna(0)
        #test.to_pickle(dump_path)
    return test
"""
init_data()
train = make_train_set()
test = make_test_set()

temp = pd.merge(train, user_app_actions.drop('appID', axis=1), how='left', on='userID')
temp['user_before_installed'] = temp.installTime < temp.clickTime
train = pd.merge(train, temp.groupby(['userID', 'clickTime'], as_index=False)['user_before_installed'].sum(), how='left', on=['userID', 'clickTime'])
temp = pd.merge(train, user_app_actions, how='left', on=['userID', 'appID'])
temp['user_app_before_installed'] = temp.installTime < temp.clickTime
train = pd.merge(train, temp.groupby(['userID', 'appID','clickTime'], as_index=False)['user_app_before_installed'].sum(), how='left', on=['userID', 'appID','clickTime'])

temp = pd.merge(test, user_app_actions.drop('appID', axis=1), how='left', on='userID')
temp['user_before_installed'] = temp.installTime < temp.clickTime
test = pd.merge(test, temp.groupby(['userID', 'clickTime'], as_index=False)['user_before_installed'].sum(), how='left', on=['userID', 'clickTime'])
temp = pd.merge(test, user_app_actions, how='left', on=['userID', 'appID'])
temp['user_app_before_installed'] = temp.installTime < temp.clickTime
test = pd.merge(test, temp.groupby(['userID', 'appID','clickTime'], as_index=False)['user_app_before_installed'].sum(), how='left', on=['userID', 'appID','clickTime'])

train.to_csv("train_xgb.csv", index=False, encoding='utf-8')
test.to_csv("test_xgb.csv", index=False, encoding='utf-8')
"""
def cv_level_statistics(train_df, test_df, f_id):
    random.seed(1234)
    # random.seed(2017)
    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    val = pd.DataFrame()

    for i in range(10):
        
        val_index=index[int((i*train_df.shape[0])/10):int(((i+1)*train_df.shape[0])/10)]
        train_index=list(set(index).difference(val_index))

        f_level = train_df.loc[train_index].groupby([f_id], as_index=False)["label"].mean()
        f_level[f_id + "_cv"] = f_level["label"]
        f_level = f_level[[f_id, f_id + "_cv"]]
        val0 = train_df.loc[val_index].merge(f_level, on=[f_id], how="left")
        val = val.append(val0, ignore_index=True)
    
    train_df = val

    f_level = train_df.groupby([f_id], as_index=False)["label"].mean()
    f_level[f_id + "_cv"] = f_level["label"]
    f_level = f_level[[f_id, f_id + "_cv"]]
    test_df = test_df.merge(f_level, on=[f_id], how="left")
    return train_df, test_df

def cal_ratio(train_df, test_df, train_before, f_id):
    f_level = train_before.groupby([f_id], as_index=False)["label"].mean()
    f_level[f_id + "_cvb"] = f_level["label"]
    f_level = f_level[[f_id, f_id + "_cvb"]]
    test_df = test_df.merge(f_level, on=[f_id], how="left")
    train_df = train_df.merge(f_level, on=[f_id], how="left")
    return train_df, test_df

def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper

    if update_df is None: update_df = test_df
    if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
    update_df.update(df)
    return

def fea_conbine(pd_, categ1, categ2):
        pd_[categ1+'_'+categ2] = pd_[categ1].astype(str) + '_' + pd_[categ2].astype(str)
        return pd_

def data_get():
 train24 = pd.read_csv("./data/data3/train_deal3.csv.low.24.f")
 train25 = pd.read_csv("./data/data3/train_deal3.csv.low.25.f")
 train26 = pd.read_csv("./data/data3/train_deal3.csv.low.26.f")
 train27 = pd.read_csv("./data/data3/train_deal3.csv.low.27.f")
 train28 = pd.read_csv("./data/data3/train_deal3.csv.low.28.f")
 train29 = pd.read_csv("./data/data3/train_deal3.csv.low.29.f")
 train30 = pd.read_csv("./data/data3/train_deal3.csv.low.30.f")
# train_ = pd.read_csv("data/train_dropduplicate_orign.csv")
# print (train_.shape)
 train = train24
 train = train.append(train25, ignore_index=True)
 train = train.append(train26, ignore_index=True)
 train = train.append(train27, ignore_index=True)
 train = train.append(train28, ignore_index=True)
 train = train.append(train29, ignore_index=True)
 train = train.append(train30, ignore_index=True)
 print(train.shape)
 train.to_csv("data/train_deal_final5.csv", index=False)
 test = pd.read_csv("./data/data3/test_deal3.csv.low.f")
 test.to_csv("data/test_deal_final5.csv", index=False) 

#data_get()

train_path = "data/train_deal_final5.csv" # "train_xgb1.csv"
test_path = "data/test_deal_final5.csv" # "test_xgb.csv"
train_path0 = "data/data3/train_deal3.csv.low.24-31.f" # "train_xgb1.csv"
test_path0 = "data/data3/test_deal3.csv.low.f" # "test_xgb.csv"

result_path = './test_log/'
model_name = "tickmodel"

print(train_path, test_path)

train = pd.read_csv(train_path)
test = pd. read_csv(test_path)
# train0 = pd.read_csv(train_path0)
# test0 = pd.read_csv(test_path0)

# train = train.merge(train0,on=["instanceID","label","day"])
# test = test.merge(test0,on=["instanceID","label","day"])

print(train.shape, test.shape)
#hcc#
features_to_use = ["appbist_size","appCategory_hist_rate","appCategory_7_rate","install_size","action7_size","appaction_size","cvr_pre_count","pv_pre_count","pv_pre_count_day","creativeID_size","hcc_creativeID_label","userID_size","hcc_userID_label","positionID_size","hcc_positionID_label","connectionType_size","hcc_connectionType_label","telecomsOperator_size","hcc_telecomsOperator_label","age_size","hcc_age_label","gender_size","hcc_gender_label","education_size","hcc_education_label","marriageStatus_size","hcc_marriageStatus_label","haveBaby_size","hcc_haveBaby_label","hometown_size","hcc_hometown_label","residence_size","hcc_residence_label","adID_size","hcc_adID_label","camgaignID_size","hcc_camgaignID_label","advertiserID_size","hcc_advertiserID_label","appID_size","hcc_appID_label","appPlatform_size","hcc_appPlatform_label","appCategory_size","hcc_appCategory_label","sitesetID_size","hcc_sitesetID_label","positionType_size","hcc_positionType_label","appCategory1_size","hcc_appCategory1_label","clickhour_size","hcc_clickhour_label","clickhour_min_size","hcc_clickhour_min_label","appCategory_age_size","hcc_appCategory_age_label","appCategory_gender_size","hcc_appCategory_gender_label","appCategory_education_size","hcc_appCategory_education_label","appCategory_marriageStatus_size","hcc_appCategory_marriageStatus_label","appCategory_haveBaby_size","hcc_appCategory_haveBaby_label","appCategory_hometown_size","hcc_appCategory_hometown_label","appCategory_residence_size","hcc_appCategory_residence_label","positionID_appID_size","hcc_positionID_appID_label","positionID_connectionType_size","hcc_positionID_connectionType_label","residence_hometown_size","hcc_residence_hometown_label","hcc_positionID_rank_label","hcc_connectionType_rank_label","hcc_positionID_connectionType_rank_label","hcc_appID_rank_label","hcc_cvr_pre_count-app_label","hcc_cvr_pre_count-pos_c_type_label","hcc_pv_pre_count-pos_c_type_label","hcc_pv_pre_count_day-pos_c_type_label","hcc_pv_pre_count-app_label","hcc_pv_pre_count_day-app_label","hcc_cvr_pre_count_label","hcc_pv_pre_count_label","hcc_pv_pre_count_day_label"]
# features_to_use = ['creativeID_size', 'hyper_creativeID', 'userID_size', 'hyper_userID', 'positionID_size', 'hyper_positionID', 'connectionType_size', 'telecomsOperator_size', 'age_size', 'hyper_age', 'gender_size', 'education_size', 'hyper_education', 'marriageStatus_size', 'hyper_marriageStatus', 'haveBaby_size', 'hyper_haveBaby', 'hometown_size', 'hyper_hometown', 'residence_size', 'hyper_residence', 'adID_size', 'hyper_adID', 'camgaignID_size', 'hyper_camgaignID', 'advertiserID_size', 'hyper_advertiserID', 'appID_size', 'hyper_appID', 'sitesetID_size', 'positionType_size', 'hyper_positionType', 'appCategory_age_size', 'hyper_appCategory_age', 'appCategory_gender_size', 'hyper_appCategory_gender', 'appCategory_education_size', 'hyper_appCategory_education', 'appCategory_marriageStatus_size', 'hyper_appCategory_marriageStatus', 'appCategory_haveBaby_size', 'hyper_appCategory_haveBaby', 'appCategory_hometown_size', 'hyper_appCategory_hometown', 'appCategory_residence_size', 'hyper_appCategory_residence', 'positionID_appID_size', 'hyper_positionID_appID', 'positionID_connectionType_size', 'hyper_positionID_connectionType', 'residence_hometown_size', 'hyper_residence_hometown', 'clickhour_size', 'hyper_clickhour', 'clickhour_min_size', 'hyper_clickhour_min', 'hyper_positionID_rank', 'hyper_connectionType_rank', 'hyper_positionID_connectionType_rank', 'hyper_appID_rank', 'hyper_cvr_pre_count-app', 'hyper_cvr_pre_count-pos_c_type', 'hyper_pv_pre_count-app', 'hyper_pv_pre_count_day-app', 'hyper_pv_pre_count', 'appbist_size', 'appCategory_hist_rate', 'appCategory_7_rate', 'install_size', 'action7_size', 'appaction_size', 'cvr_pre_count', 'pv_pre_count', 'pv_pre_count_day', 'hyper_app-pos-pv_pre', 'hyper_app-pos-pv_day_pre', 'hyper_app-pos-cv_pre','before_size', 'before_c_rate', 'before_c']
# all #
# features_to_use = ["creativeID_size","hyper_creativeID","userID_size","hyper_userID","positionID_size","hyper_positionID","connectionType_size","hyper_connectionType","telecomsOperator_size","hyper_telecomsOperator","age_size","hyper_age","gender_size","hyper_gender","education_size","hyper_education","marriageStatus_size","hyper_marriageStatus","haveBaby_size","hyper_haveBaby","hometown_size","hyper_hometown","residence_size","hyper_residence","adID_size","hyper_adID","camgaignID_size","hyper_camgaignID","advertiserID_size","hyper_advertiserID","appID_size","hyper_appID","appPlatform_size","hyper_appPlatform","appCategory_size","hyper_appCategory","sitesetID_size","hyper_sitesetID","positionType_size","hyper_positionType","appCategory1_size","hyper_appCategory1","appCategory_age_size","hyper_appCategory_age","appCategory_gender_size","hyper_appCategory_gender","appCategory_education_size","hyper_appCategory_education","appCategory_marriageStatus_size","hyper_appCategory_marriageStatus","appCategory_haveBaby_size","hyper_appCategory_haveBaby","appCategory_hometown_size","hyper_appCategory_hometown","appCategory_residence_size","hyper_appCategory_residence","positionID_appID_size","hyper_positionID_appID","positionID_connectionType_size","hyper_positionID_connectionType","residence_hometown_size","hyper_residence_hometown","clickhour_size","hyper_clickhour","clickhour_min_size","hyper_clickhour_min","hyper_positionID_rank","hyper_connectionType_rank","hyper_positionID_connectionType_rank","hyper_appID_rank","hyper_cvr_pre_count-app","hyper_cvr_pre_count-pos_c_type","hyper_pv_pre_count-pos_c_type","hyper_pv_pre_count_day-pos_c_type","hyper_pv_pre_count-app","hyper_pv_pre_count_day-app","hyper_cvr_pre_count","hyper_pv_pre_count","hyper_pv_pre_count_day","appbist_size","appCategory_hist_rate","appCategory_7_rate","install_size","action7_size","appaction_size","cvr_pre_count","pv_pre_count","pv_pre_count_day","hyper_app-pos-pv_pre","hyper_app-pos-pv_day_pre","hyper_app-pos-cv_pre"]#app_size

#del less del app_size# features_to_use = ["creativeID_size","hyper_creativeID","userID_size","hyper_userID","positionID_size","hyper_positionID","connectionType_size","telecomsOperator_size","age_size","hyper_age","gender_size","education_size","hyper_education","marriageStatus_size","hyper_marriageStatus","haveBaby_size","hyper_haveBaby","hometown_size","hyper_hometown","residence_size","hyper_residence","adID_size","hyper_adID","camgaignID_size","hyper_camgaignID","advertiserID_size","hyper_advertiserID","appID_size","hyper_appID","sitesetID_size","positionType_size","hyper_positionType","appCategory_age_size","hyper_appCategory_age","appCategory_gender_size","hyper_appCategory_gender","appCategory_education_size","hyper_appCategory_education","appCategory_marriageStatus_size","hyper_appCategory_marriageStatus","appCategory_haveBaby_size","hyper_appCategory_haveBaby","appCategory_hometown_size","hyper_appCategory_hometown","appCategory_residence_size","hyper_appCategory_residence","positionID_appID_size","hyper_positionID_appID","positionID_connectionType_size","hyper_positionID_connectionType","residence_hometown_size","hyper_residence_hometown","clickhour_size","hyper_clickhour","clickhour_min_size","hyper_clickhour_min","hyper_positionID_rank","hyper_connectionType_rank","hyper_positionID_connectionType_rank","hyper_appID_rank","appbist_size","appCategory_hist_rate","appCategory_7_rate","install_size","action7_size","appaction_size"]
#"hyper_cvr_pre_count-app","hyper_cvr_pre_count-pos_c_type","hyper_pv_pre_count-app","hyper_pv_pre_count_day-app","hyper_pv_pre_count","cvr_pre_count","pv_pre_count","pv_pre_count_day","hyper_app-pos-pv_pre","hyper_app-pos-pv_day_pre","hyper_app-pos-cv_pre"]
# features_to_use = ['positionID_connectionType_size','hyper_positionID_connectionType','creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory', 'appCategory1', 'app_size', 'appCategory_hist_rate', 'appCategory_hist', 'action7_size', 'appCategory_7_rate', 'appCategory_7', 'click', 'clickhour_min', 'clickhour', 'clickmin', 'hyper_creativeID', 'creativeID_size', 'hyper_userID', 'userID_size', 'hyper_positionID', 'positionID_size', 'hyper_connectionType', 'connectionType_size', 'hyper_telecomsOperator', 'telecomsOperator_size', 'hyper_age', 'age_size', 'hyper_gender', 'gender_size', 'hyper_education', 'education_size', 'hyper_marriageStatus', 'marriageStatus_size', 'hyper_haveBaby', 'haveBaby_size', 'hyper_hometown', 'hometown_size', 'hyper_residence', 'residence_size', 'hyper_adID', 'adID_size', 'hyper_camgaignID', 'camgaignID_size', 'hyper_advertiserID', 'advertiserID_size', 'hyper_appID', 'appID_size', 'hyper_appPlatform', 'appPlatform_size', 'hyper_appCategory', 'appCategory_size', 'hyper_sitesetID', 'sitesetID_size', 'hyper_positionType', 'positionType_size', 'hyper_appCategory1', 'appCategory1_size', 'hyper_appCategory_age', 'appCategory_age_size', 'hyper_appCategory_gender', 'appCategory_gender_size', 'hyper_appCategory_education', 'appCategory_education_size', 'hyper_appCategory_marriageStatus', 'appCategory_marriageStatus_size', 'hyper_appCategory_haveBaby', 'appCategory_haveBaby_size', 'hyper_appCategory_hometown', 'appCategory_hometown_size', 'hyper_appCategory_residence', 'appCategory_residence_size', 'hyper_clickhour', 'clickhour_size', 'hyper_clickmin', 'clickmin_size', 'hyper_clickhour_min', 'clickhour_min_size']

# features_to_use = ['hyper_positionID_click','hyper_positionID_is_click','clickhour_size', 'hometown_size', 'camgaignID', 'hyper_advertiserID', 'hyper_appCategory_education', 'appCategory_hometown_size', 'adID', 'camgaignID_size', 'hyper_clickhour', 'hyper_education', 'creativeID_size', 'hometown', 'adID_size', 'age_size', 'hyper_age', 'appCategory_residence_size', 'hyper_hometown', 'appCategory_age_size', 'hyper_userID', 'creativeID', 'residence_size', 'hyper_residence', 'clickhour_min_size', 'hyper_appCategory_hometown', 'positionID', 'residence', 'hyper_appCategory_residence', 'hyper_camgaignID', 'hyper_adID', 'age', 'hyper_clickhour_min', 'hyper_connectionType', 'appCategory_hist_rate', 'hyper_creativeID', 'clickhour_min', 'hyper_appCategory_age', 'app_size', 'positionID_size', 'hyper_positionID']

# features_to_use = ["install_size","appCategory_hist_rate","action7_size","appCategory_7_rate","creativeID","positionID","connectionType","telecomsOperator","age","gender","education","marriageStatus","haveBaby","hometown","residence","adID","camgaignID","advertiserID","appID","appPlatform","appCategory","sitesetID","positionType","user_app_before_installed","appCategory1","app_size","appCategory_age","appCategory_gender","appCategory_education","appCategory_marriageStatus","appCategory_haveBaby","appCategory_hometown","appCategory_residence","clickhour","clickmin","clickhour_min","hyper_creativeID","hyper_userID","hyper_positionID","hyper_connectionType","hyper_telecomsOperator","hyper_age","hyper_gender","hyper_education","hyper_marriageStatus","hyper_haveBaby","hyper_hometown","hyper_residence","hyper_adID","hyper_camgaignID","hyper_advertiserID","hyper_appID","hyper_appPlatform","hyper_appCategory","hyper_sitesetID","hyper_positionType","hyper_appCategory1","hyper_appCategory_age","hyper_appCategory_gender","hyper_appCategory_education","hyper_appCategory_marriageStatus","hyper_appCategory_haveBaby","hyper_appCategory_hometown","hyper_appCategory_residence","hyper_clickhour","hyper_clickmin","hyper_clickhour_min"]
# features_to_use = ['creativeID_size', 'hyper_creativeID', 'userID_size', 'hyper_userID', 'positionID_size', 'hyper_positionID', 'connectionType_size', 'telecomsOperator_size', 'age_size', 'hyper_age', 'gender_size', 'education_size', 'hyper_education', 'marriageStatus_size', 'hyper_marriageStatus', 'haveBaby_size', 'hyper_haveBaby', 'hometown_size', 'hyper_hometown', 'residence_size', 'hyper_residence', 'adID_size', 'hyper_adID', 'camgaignID_size', 'hyper_camgaignID', 'advertiserID_size', 'hyper_advertiserID', 'appID_size', 'hyper_appID', 'sitesetID_size', 'positionType_size', 'hyper_positionType', 'appCategory_age_size', 'hyper_appCategory_age', 'appCategory_gender_size', 'hyper_appCategory_gender', 'appCategory_education_size', 'hyper_appCategory_education', 'appCategory_marriageStatus_size', 'hyper_appCategory_marriageStatus', 'appCategory_haveBaby_size', 'hyper_appCategory_haveBaby', 'appCategory_hometown_size', 'hyper_appCategory_hometown', 'appCategory_residence_size', 'hyper_appCategory_residence', 'positionID_appID_size', 'hyper_positionID_appID', 'positionID_connectionType_size', 'hyper_positionID_connectionType', 'residence_hometown_size', 'hyper_residence_hometown', 'clickhour_size', 'hyper_clickhour', 'clickhour_min_size', 'hyper_clickhour_min', 'hyper_positionID_rank', 'hyper_connectionType_rank', 'hyper_positionID_connectionType_rank', 'hyper_appID_rank', 'hyper_cvr_pre_count-app', 'hyper_cvr_pre_count-pos_c_type', 'hyper_pv_pre_count-app', 'hyper_pv_pre_count_day-app', 'hyper_pv_pre_count', 'appCategory_hist_rate', 'install_size', 'cvr_pre_count', 'pv_pre_count', 'pv_pre_count_day', 'hyper_app-pos-pv_pre', 'hyper_app-pos-pv_day_pre', 'hyper_app-pos-cv_pre']
#del trick# features_to_use = ['creativeID_size', 'hyper_creativeID', 'userID_size', 'hyper_userID', 'positionID_size', 'hyper_positionID', 'connectionType_size', 'telecomsOperator_size', 'age_size', 'hyper_age', 'gender_size', 'education_size', 'hyper_education', 'marriageStatus_size', 'hyper_marriageStatus', 'haveBaby_size', 'hyper_haveBaby', 'hometown_size', 'hyper_hometown', 'residence_size', 'hyper_residence', 'adID_size', 'hyper_adID', 'camgaignID_size', 'hyper_camgaignID', 'advertiserID_size', 'hyper_advertiserID', 'appID_size', 'hyper_appID', 'sitesetID_size', 'positionType_size', 'hyper_positionType', 'appCategory_age_size', 'hyper_appCategory_age', 'appCategory_gender_size', 'hyper_appCategory_gender', 'appCategory_education_size', 'hyper_appCategory_education', 'appCategory_marriageStatus_size', 'hyper_appCategory_marriageStatus', 'appCategory_haveBaby_size', 'hyper_appCategory_haveBaby', 'appCategory_hometown_size', 'hyper_appCategory_hometown', 'appCategory_residence_size', 'hyper_appCategory_residence', 'positionID_appID_size', 'hyper_positionID_appID', 'positionID_connectionType_size', 'hyper_positionID_connectionType', 'residence_hometown_size', 'hyper_residence_hometown', 'clickhour_size', 'hyper_clickhour', 'clickhour_min_size', 'hyper_clickhour_min', 'hyper_positionID_rank', 'hyper_connectionType_rank', 'hyper_positionID_connectionType_rank', 'hyper_appID_rank', 'appbist_size', 'appCategory_hist_rate', 'appCategory_7_rate', 'install_size', 'action7_size', 'appaction_size']
"""
features_to_use0 = []
for fea in features_to_use:
    features_to_use0.extend([fea+"_x", fea+"_y"])
features_to_use = features_to_use0
"""
instanceIDs = test["instanceID"]
sub_training_data = np.array(test[features_to_use])
print(features_to_use)

# cut for day
train_b = train.ix[train["day"]<30]
train_30 = train.ix[train["day"]==30]
train_30 = train_30.ix[train_30["appID"]!=360]
train = train_b.append(train_30,ignore_index=True)
# train = train.ix[train["day"]<30]
train_data = train.ix[train["day"]!=24]
y_train, train_ids = train_data["label"], train_data["instanceID"]
val_data = train.ix[train["day"] == 24]
y_test, val_ids = val_data["label"], val_data["instanceID"]

X_train = np.array(train_data[features_to_use])
X_test = np.array(val_data[features_to_use])
print X_train.shape, X_test.shape
"""
# random_split
# train = train.ix[train["day"]<30]
training_data, label, train_ids = train, train["label"], train["instanceID"]
training_data = np.array(training_data[features_to_use])
X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.1, random_state=0)
"""
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

y = bst.predict(sub_training_data_)
res = instanceIDs.to_frame()
res['prob'] = y

res.to_csv(result_path+'hccsubmission_win.csv', index=False)
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
df.to_csv("hccimportance_win.csv", index = False, encoding = "utf-8")
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig(result_path+'hccfeature_importance_xgb_win.png')

