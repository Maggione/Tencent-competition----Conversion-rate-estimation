#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import sys
import xgboost as xgb


test = pd.read_csv("./data/test_deal_final2.csv")
features_to_use = ['creativeID_size', 'hcc_creativeID_label', 'userID_size', 'hcc_userID_label', 'positionID_size', 'hcc_positionID_label', 'connectionType_size', 'hcc_connectionType_label', 'telecomsOperator_size', 'hcc_telecomsOperator_label', 'age_size', 'hcc_age_label', 'gender_size', 'hcc_gender_label', 'education_size', 'hcc_education_label', 'marriageStatus_size', 'hcc_marriageStatus_label', 'haveBaby_size', 'hcc_haveBaby_label', 'hometown_size', 'hcc_hometown_label', 'residence_size', 'hcc_residence_label', 'sitesetID_size', 'hcc_sitesetID_label', 'positionType_size', 'hcc_positionType_label', 'adID_size', 'hcc_adID_label', 'camgaignID_size', 'hcc_camgaignID_label', 'advertiserID_size', 'hcc_advertiserID_label', 'appID_size', 'hcc_appID_label', 'appPlatform_size', 'hcc_appPlatform_label', 'appCategory_size', 'hcc_appCategory_label', 'clickhour_min_size', 'hcc_clickhour_min_label', 'clickhour_size', 'hcc_clickhour_label', 'positionID_connectionType_size', 'hcc_positionID_connectionType_label', 'positionID_appID_size', 'hcc_positionID_appID_label', 'positionID_advertiserID_size', 'hcc_positionID_advertiserID_label', 'hometown_residence_size', 'hcc_hometown_residence_label', 'age_education_size', 'hcc_age_education_label', 'appCategory_age_size', 'hcc_appCategory_age_label', 'appCategory_gender_size', 'hcc_appCategory_gender_label', 'appCategory_education_size', 'hcc_appCategory_education_label', 'appCategory_marriageStatus_size', 'hcc_appCategory_marriageStatus_label', 'appCategory_haveBaby_size', 'hcc_appCategory_haveBaby_label', 'appCategory_hometown_size', 'hcc_appCategory_hometown_label', 'appCategory_residence_size', 'hcc_appCategory_residence_label', 'appCategory1', 'install_size', 'appCategory_hist_rate', 'appCategory_hist', 'action_7_size', 'appCategory_7_rate', 'appCategory_7', 'cvr_pre_count', 'app_fix', 'pv_pre_count', 'pv_pre_count_day', 'hcc_cvr_pre_count_label', 'hcc_pv_pre_count_label', 'hcc_pv_pre_count_day_label', 'hcc_cvr_pre_count-app_label', 'hcc_cvr_pre_count-pos_c_type_label', 'hcc_pv_pre_count-pos_c_type_label', 'hcc_pv_pre_count_day-pos_c_type_label', 'hcc_pv_pre_count-app_label', 'hcc_pv_pre_count_day-app_label', 'hcc_app-pos_label', 'hcc_app-pos-pv_pre_label', 'hcc_app-pos-pv_day_pre_label', 'hcc_app-pos-cv_pre_label', 'min_user_click_diff', 'min_user_creativeid_click_diff', 'min_user_next_click_diff', 'min_user_creativeid_next_click_diff']

test_data = np.array(test[features_to_use])
instanceIDs = test["instanceID"]
model_name0 = sys.argv[1]
result_path= sys.argv[2]
for i in range(7):
    model_name = model_name0+str(i)
    bst = xgb.Booster() #init model
    bst.load_model(model_name) # load data
    dtest  = xgb.DMatrix(test_data)
    y = bst.predict(dtest)
    res = instanceIDs.to_frame()
    res['prob'] = y
    res.to_csv(result_path+'submission'+str(i)+'.csv', index=False)

