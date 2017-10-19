#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import sys
# import xgboost as xgb
import lightgbm as lgb
import os

test_path = sys.argv[1]
result_path = sys.argv[2]
model_name = sys.argv[3]

cache = "cache_all"

def read_init_datafile(filepath, dtype_dict):
    dump_path = os.path.join(cache, filepath[13:-5]+'.add.pickle')
    if not os.path.exists(cache):
        os.mkdir(cache)
    if os.path.exists(dump_path):
        data = pd.read_pickle(dump_path)
    else:
        data = pd.read_csv(filepath, dtype = dtype_dict).to_sparse(fill_value = 0)
        data.to_pickle(dump_path)
    return data


features_to_use = ["creativeID_size","hcc_creativeID_label","userID_size","hcc_userID_label","positionID_size","hcc_positionID_label","connectionType_size","hcc_connectionType_label","telecomsOperator_size","hcc_telecomsOperator_label","age_size","hcc_age_label","gender_size","hcc_gender_label","education_size","hcc_education_label","marriageStatus_size","hcc_marriageStatus_label","haveBaby_size","hcc_haveBaby_label","hometown_size","hcc_hometown_label","residence_size","hcc_residence_label","sitesetID_size","hcc_sitesetID_label","positionType_size","hcc_positionType_label","adID_size","hcc_adID_label","camgaignID_size","hcc_camgaignID_label","advertiserID_size","hcc_advertiserID_label","appID_size","hcc_appID_label","appPlatform_size","hcc_appPlatform_label","appCategory_size","hcc_appCategory_label","clickhour_min_size","hcc_clickhour_min_label","clickhour_size","hcc_clickhour_label","positionID_connectionType_size","hcc_positionID_connectionType_label","positionID_appID_size","hcc_positionID_appID_label","positionID_advertiserID_size","hcc_positionID_advertiserID_label","hometown_residence_size","hcc_hometown_residence_label","age_education_size","hcc_age_education_label","appCategory_age_size","hcc_appCategory_age_label","appCategory_gender_size","hcc_appCategory_gender_label","appCategory_education_size","hcc_appCategory_education_label","appCategory_marriageStatus_size","hcc_appCategory_marriageStatus_label","appCategory_haveBaby_size","hcc_appCategory_haveBaby_label","appCategory_hometown_size","hcc_appCategory_hometown_label","appCategory_residence_size","hcc_appCategory_residence_label","appCategory1","install_size","appCategory_hist_rate","appCategory_hist","action_7_size","appCategory_7_rate","appCategory_7","cvr_pre_count","app_fix","pv_pre_count","pv_pre_count_day","hcc_cvr_pre_count_label","hcc_pv_pre_count_label","hcc_pv_pre_count_day_label","hcc_cvr_pre_count-app_label","hcc_cvr_pre_count-pos_c_type_label","hcc_pv_pre_count-pos_c_type_label","hcc_pv_pre_count_day-pos_c_type_label","hcc_pv_pre_count-app_label","hcc_pv_pre_count_day-app_label","hcc_app-pos_label","hcc_app-pos-pv_pre_label","hcc_app-pos-pv_day_pre_label","hcc_app-pos-cv_pre_label","min_user_click_diff","min_user_creativeid_click_diff","min_user_next_click_diff","min_user_creativeid_next_click_diff","app_installed_size","user_app_installed","app_actioned_size","user_actions",'creativeID_size_all','userID_size_all','positionID_size_all','connectionType_size_all','telecomsOperator_size_all','age_size_all','gender_size_all','education_size_all','marriageStatus_size_all','haveBaby_size_all','hometown_size_all','residence_size_all','sitesetID_size_all','positionType_size_all','adID_size_all','camgaignID_size_all','advertiserID_size_all','appID_size_all','appPlatform_size_all','appCategory_size_all','clickhour_min_size_all','clickhour_size_all','clickmin_size_all','day_size_all','positionID_connectionType_size_all','positionID_appID_size_all','positionID_advertiserID_size_all','hometown_residence_size_all','age_education_size_all','appCategory_age_size_all','appCategory_gender_size_all','appCategory_education_size_all','appCategory_marriageStatus_size_all','appCategory_haveBaby_size_all','appCategory_hometown_size_all','appCategory_residence_size_all']
dtype_dict = {}
for key in features_to_use:
    dtype_dict[key] = np.float32


test = read_init_datafile(test_path, dtype_dict)
feature_to_replace =["creativeID_size","userID_size","positionID_size","connectionType_size","telecomsOperator_size","age_size","gender_size","education_size","marriageStatus_size","haveBaby_size","hometown_size","residence_size","sitesetID_size","positionType_size","adID_size","camgaignID_size","advertiserID_size","appID_size","appPlatform_size","appCategory_size","clickhour_min_size","clickhour_size","positionID_connectionType_size","positionID_appID_size","positionID_advertiserID_size","hometown_residence_size","age_education_size","appCategory_age_size","appCategory_gender_size","appCategory_education_size","appCategory_marriageStatus_size","appCategory_haveBaby_size","appCategory_hometown_size","appCategory_residence_size"]
test2 = read_init_datafile("./final_data/test_final.csv.b.hh.all.aaaa", dtype_dict)
for fid in feature_to_replace:
    test[fid] = test2[fid]

test, test_label, test_ids = test[features_to_use], test["label"], test["instanceID"]

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

for i in range(7):
    model_name_all = result_path + model_name + str(i)
    bst = lgb.Booster(params, model_file=model_name_all)
    y = bst.predict(test)
    res = test_ids.to_dense().to_frame()
    res['prob'] = y
    res.to_csv(result_path+'submission_b-30'+str(i)+'.csv', index=False)

