#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cPickle as pk
import pandas as pd
import os
import math
import sys
print "脚本名：",sys.argv[0]

# variable
DIR = sys.argv[1]
begin_day = int(sys.argv[2])
window = int(sys.argv[3])
correct = 0

# ---functions--- #
# filter 
def designate_less_observations(df1, df2, size, column):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().reset_index().rename(columns={0: column+"_size"})
    print grouped.head()
    tmp1 = df1.merge(grouped, on = column, how = "left")
    tmp2 = df2.merge(grouped, on = column, how = "left")
    df1.loc[tmp1[column+"_size"] <= size, column] = -1
    df2.loc[tmp2[column+"_size"] <= size, column] = -1

    return df1, df2
# conbine
def fea_conbine(pd_, categ1, categ2):
        pd_[categ1+'_'+categ2] = pd_[categ1].astype(str) + '_' + pd_[categ2].astype(str)
        return pd_

if os.path.exists(DIR + "/train_count.w"+str(window)+"."+str(begin_day)+"-30.csv")==0 \
           or os.path.exists(DIR + "/test_count.w"+str(window)+".csv")==0 or correct:
    # read origin file
    train_file = DIR + "/train_small.csv"
    test_file = DIR + "/test_small.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    print "origin:", train.shape, test.shape

    user_df = pd.read_csv(DIR+'user.csv')
    ad_df = pd.read_csv(DIR+'ad.csv')
    app_df = pd.read_csv(DIR+'app_categories.csv')
    position_df = pd.read_csv(DIR+'position.csv')

    # get basic features
    train = pd.merge(train,user_df,on="userID",how="left")
    test = pd.merge(test,user_df,on="userID",how="left")
    train = pd.merge(train,position_df,on="positionID",how="left")
    test = pd.merge(test,position_df,on="positionID",how="left")
    train = pd.merge(train,ad_df,on="creativeID",how="left")
    test = pd.merge(test,ad_df,on="creativeID",how="left")
    train = pd.merge(train,app_df,on="appID",how="left")
    test = pd.merge(test,app_df,on="appID",how="left")

    del user_df
    del ad_df
    del app_df
    del position_df

    # inital
    train["clickhour_min"] = train["clickTime"].apply(lambda x: x%1000000/100)
    test["clickhour_min"] = test["clickTime"].apply(lambda x: x%1000000/100)
    train["clickhour"] = train["clickhour_min"].apply(lambda x: x/100)
    test["clickhour"] = test["clickhour_min"].apply(lambda x: x/100)
    # train["clickmin"] = train["clickTime"].apply(lambda x: x%100)
    # test["clickmin"] = test["clickTime"].apply(lambda x: x%100)
    train["day"] = train["clickTime"].apply(lambda x : x/1000000)
    test["day"] = test["clickTime"].apply(lambda x: x/1000000)

    print "basic:", train.shape, test.shape

    basic_features=["userID", "creativeID", "appID", "positionID", "camgaignID", "adID", "sitesetID", "advertiserID", "appPlatform", "connectionType", "telecomsOperator", "positionType", "gender", "education", "hometown", "residence", "marriageStatus", "age", "appCategory", "clickhour", "clickhour_min"]
    con_features = [('userID', 'creativeID'), ('positionID', 'positionType'), ('positionID', 'advertiserID'), ('positionID', 'gender'), ('hometown', 'residence'),('gender', 'education'), ('positionID', 'marriageStatus'), ('age', 'marriageStatus'), ('positionID', 'age'), ('positionID', 'appID'),('positionID', 'hometown'), ('positionID', 'telecomsOperator'), ('positionID', 'creativeID'), ('positionID', 'education'), ('camgaignID', 'connectionType'), ('positionID', 'connectionType'), ('userID', 'adID'), ('creativeID', 'connectionType'), ('userID', 'connectionType'),('creativeID', 'gender'), ('positionID', 'userID'), ('positionID', 'camgaignID'), ('age',  'gender'), ('camgaignID', 'age' ), ('adID', 'connectionType'), ('camgaignID', 'gender'), ('userID', 'appCategory'),( 'advertiserID', 'connectionType' ), ('positionID', 'adID'), ('positionID', 'appCategory'), ('positionID', 'haveBaby')]

print "start get count:"
# train_file
if os.path.exists(DIR + "/train_count.w"+str(window)+"."+str(begin_day)+"-30.csv")==0 or correct:
    val_df = []
    for i in range(31)[begin_day:31]:
        if os.path.exists(train_file+".count.w"+str(window)+'.'+str(i)) and correct == 0:
            val_ = pd.read_csv(train_file+".count.w."+str(window)+'.'+str(i))
        else:
            print "day:", i
            val_ = train[train["day"] == i]
            train_ = train[train["day"] < i]
            train_ = train_[train_["day"] >= i-window]
            for f_id in basic_features:
                train_f_id = train_[[f_id, "label"]]
                grouped = train_f_id.groupby([f_id]).size().reset_index().rename(columns={0: f_id+"_click_size"})
                val_ = val_.merge(grouped, on=f_id, how="left")

                train_f_id = train_f_id[train_f_id["label"] == 1]
                grouped = train_f_id.groupby([f_id]).size().reset_index().rename(columns={0: f_id+"_active_size"})
                val_ = val_.merge(grouped, on=f_id, how="left")

            for f_id in con_features:
                train_f_id = train_[[f_id[0], f_id[1], "label"]]
                grouped = train_f_id.groupby([f_id[0], f_id[1]]).size().reset_index().rename(columns={0: f_id[0]+"_"+f_id[1]+"_click_size"})
                val_ = val_.merge(grouped, on=[f_id[0],f_id[1]], how="left")

                train_f_id = train_f_id[train_f_id["label"] == 1]
                grouped = train_f_id.groupby([f_id[0], f_id[1]]).size().reset_index().rename(columns={0: f_id[0]+"_"+f_id[1]+"_active_size"})
                val_ = val_.merge(grouped, on=f_id, how="left")
            if i == 30:
                instanceID_of_30 = pd.read_csv(DIR + "/instanceID_of_30.csv")
                val_ = instanceID_of_30.merge(val_, on="instanceID", how="left")
            val_.to_csv(train_file+".count.w."+str(window)+'.'+str(i), index=False)
        val_df.append(val_)
    train_new = pd.concat(val_df)
    train_new = train_new.sort_values(by=['instanceID'])
    train_new.to_csv(DIR + "/train_count.w"+str(window)+"."+str(begin_day)+"-30.csv", index=False)

# test_file
if os.path.exists(DIR + "/test_count.w"+str(window)+".csv") == 0 or correct:
    print "test"
    train = train[train["day"] >= 31-window]
    for f_id in basic_features:
        train_f_id = train[[f_id, "label"]]
        grouped = train_f_id.groupby([f_id]).size().reset_index().rename(columns={0: f_id+"_click_size"})
        test = test.merge(grouped, on=f_id, how="left")

        train_f_id = train_f_id[train_f_id["label"] == 1]
        grouped = train_f_id.groupby([f_id]).size().reset_index().rename(columns={0: f_id+"_active_size"})
        test = test.merge(grouped, on=f_id, how="left")

    for f_id in con_features:
        train_f_id = train[[f_id[0], f_id[1], "label"]]
        grouped = train_f_id.groupby([f_id[0], f_id[1]]).size().reset_index().rename(columns={0: f_id[0]+"_"+f_id[1]+"_click_size"})
        test = test.merge(grouped, on=[f_id[0],f_id[1]], how="left")

        train_f_id = train_f_id[train_f_id["label"] == 1]
        grouped = train_f_id.groupby([f_id[0], f_id[1]]).size().reset_index().rename(columns={0: f_id[0]+"_"+f_id[1]+"_active_size"})
        test = test.merge(grouped, on=f_id, how="left")

    test = test.sort_values(by=['instanceID'])
    test.to_csv(DIR + "/test_count.w"+str(window)+".csv", index=False)

print ('\nwindow count calculation finish.')
