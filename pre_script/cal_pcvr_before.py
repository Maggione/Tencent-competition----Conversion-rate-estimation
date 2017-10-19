#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cPickle as pk
import pandas as pd
import os
import math
import random
import sys
print "脚本名：",sys.argv[0]
from hyper import HyperParam

# ---function--- #
def applyParallel_feature(funcs, execu):
    ''' 利用joblib来并行提取特征
    '''
    with Parallel(n_jobs=8) as parallel:
        retLst = parallel( delayed(execu)(func) for func in funcs )
        return None

def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=8) as parallel:
        retLst = parallel( delayed(func)(group) for group in dfGrouped )
        return pd.concat(retLst, axis=0)    
def execu(func):
    print (func)
    x = func[0](*func[1])
    del x
    gc.collect()
    print (func, 'done')

def designate_less_observations(df1, df2, size, column):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().reset_index().rename(columns={0: column+"_size"})
    print grouped.head()
    tmp1 = df1.merge(grouped, on = column, how = "left")
    tmp2 = df2.merge(grouped, on = column, how = "left")
    df1.loc[tmp1[column+"_size"] <= size, column] = -1
    df2.loc[tmp2[column+"_size"] <= size, column] = -1
    return df1, df2

def fea_conbine(pd_, categ1, categ2):
    pd_[categ1+'_'+categ2] = pd_[categ1].astype(str) + '_' + pd_[categ2].astype(str)
    return pd_

def age_scope(x):
    if x < 5: return 0
    if x < 10: return 5
    if x < 15: return 10
    if x < 20: return 15
    if x < 25: return 20
    if x < 30: return 25
    if x < 35: return 30
    if x < 40: return 35
    if x < 45: return 40
    if x < 50: return 45
    if x < 55: return 50
    if x < 60: return 55
    if x < 65: return 60
    if x < 70: return 65
    if x < 75: return 70
    if x < 80: return 75
    if x < 85: return 80

# get features pcvr
def smooth(train_, val_, f_id, file_name=None):
    bayes_name = '_'.join(["bayes"] + f_id)
    print '\r%s' % (bayes_name+"            "),
    train_f_id = train_[f_id + ["label"]]
    train_f_id["count"] = 1
    # val_f_id = val_f_id[["instanceID"] + f_id]
    grouped = train_f_id.groupby(f_id)[['count', 'label']].agg('sum').reset_index() 
    hyper = HyperParam(1,1)
    I = grouped['count'] * 1.0
    C = grouped['label'] * 1.0
    # pcvr_smoothing
    hyper.update_from_data_by_FPI(I, C, 1000, 0.00001)
    grouped[bayes_name] = (hyper.alpha + grouped['label']).values / (hyper.alpha + hyper.beta + grouped['count']).values
    grouped.drop(['count','label'], axis=1, inplace=True)
    val_ = val_.merge(grouped, on=f_id, how="left")
    return val_
    # val_f_id = val_f_id.merge(grouped, on=f_id, how="left")
    # val_f_id.drop(f_id, axis=1, inplace=True)
    # val_f_id.to_csv(file_name, index=False)

def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc"] + variable)
    print '\r%s' % (hcc_name+"            "),
    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[variable].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper

    # if update_df is None: update_df = test_df
    if hcc_name not in test_df.columns: test_df[hcc_name] = np.nan
    test_df.update(df)
    return

# variable
DIR = sys.argv[1]
begin_day = int(sys.argv[2])
correct = 0

# origin file
train_file = DIR + "/train.csv.good"
test_file = DIR + "/test_small.csv"

if os.path.exists(DIR + "/train_pcvr."+str(begin_day)+"-30.csv")==0 \
    or os.path.exists(DIR + "/test_pcvr.csv")==0 or correct:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    print "origin:", train.shape, test.shape

    # get basic features
    user_df = pd.read_csv(DIR+'user.csv')
    ad_df = pd.read_csv(DIR+'ad.csv')
    app_df = pd.read_csv(DIR+'app_categories.csv')
    position_df = pd.read_csv(DIR+'position.csv')

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

    # get rank features
    train_app_rank_df = pd.read_csv(DIR+"train_rank_appID.csv")
    test_app_rank_df = pd.read_csv(DIR+"test_rank_appID.csv")
    train_creativeID_rank_df = pd.read_csv(DIR+"train_rank_creativeID.csv")
    test_creativeID_rank_df = pd.read_csv(DIR+"test_rank_creativeID.csv")

    train_app_rank_df = train_app_rank_df.drop(["instanceID","day"], axis=1)
    test_app_rank_df = test_app_rank_df.drop(["instanceID","day"], axis=1)
    train_creativeID_rank_df = train_creativeID_rank_df.drop(["instanceID","day"], axis=1)
    test_creativeID_rank_df = test_creativeID_rank_df.drop(["instanceID","day"], axis=1)
    train = pd.concat([train, train_app_rank_df], axis=1)
    train = pd.concat([train, train_creativeID_rank_df], axis=1)
    test = pd.concat([test, test_app_rank_df], axis=1)
    test = pd.concat([test, test_creativeID_rank_df], axis=1)

    del train_app_rank_df
    del test_app_rank_df
    del train_creativeID_rank_df
    del test_creativeID_rank_df

    # get time diff features
    train_time_df = pd.read_csv(DIR+"train_time_diff.csv")
    test_time_df = pd.read_csv(DIR+"test_time_diff.csv")

    train_time_df = train_time_df.drop(["instanceID"], axis=1)
    test_time_df = test_time_df.drop(["instanceID"], axis=1)
    train = pd.concat([train, train_time_df], axis=1)
    test = pd.concat([test, test_time_df], axis=1)

    del train_time_df
    del test_time_df


    # inital
    train["clickhour_min"] = train["clickTime"].apply(lambda x: x%1000000/100)
    test["clickhour_min"] = test["clickTime"].apply(lambda x: x%1000000/100)
    train["clickhour"] = train["clickhour_min"].apply(lambda x: x/100)
    test["clickhour"] = test["clickhour_min"].apply(lambda x: x/100)
    # train["clickmin"] = train["clickTime"].apply(lambda x: x%100)
    # test["clickmin"] = test["clickTime"].apply(lambda x: x%100)
    train["day"] = train["clickTime"].apply(lambda x : x/1000000)
    test["day"] = test["clickTime"].apply(lambda x: x/1000000)

    train['age_scope'] = pd.cut(train['age'], 16, labels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], retbins=True)[0]
    train.ix[train['age']==0]['age_scope'] = -1 
    test['age_scope'] = pd.cut(test['age'], 16, labels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], retbins=True)[0]
    test.ix[test['age']==0]['age_scope'] = -1 

    train["age_scope"] = train["age_scope"].astype(int)
    test["age_scope"] = test["age_scope"].astype(int)

    print "basic:", train.shape, test.shape

    basic_features=["userID", "creativeID", "appID", "positionID", "camgaignID", "adID", "sitesetID", "advertiserID", 
                "appPlatform", "connectionType", "telecomsOperator", "positionType", "gender", "education", 
                "hometown", "residence", "marriageStatus", "age", "appCategory", "clickhour", "clickhour_min"]
    trick_features=["appID_rank" , "appID_rank_day", "appID_rank_day_sign", "creativeID_rank", "creativeID_rank_day", 
               "creativeID_rank_day_sign", "min_user_click_diff1", "min_user_next_click_diff1", "min_user_creativeid_click_diff1",
               "min_user_creativeid_next_click_diff1", "min_user_appid_click_diff1", "min_user_appid_next_click_diff1"]
    con_features = [('age', 'marriageStatus'), ('userID', 'creativeID'), ('positionID', 'positionType'), 
                ('positionID', 'advertiserID'), ('positionID', 'gender'), ('hometown', 'residence'),
                ('gender', 'education'), ('positionID', 'marriageStatus'), ('positionID', 'age'), 
                ('positionID', 'appID'),('positionID', 'hometown'), ('positionID', 'telecomsOperator'), 
                ('positionID', 'creativeID'), ('positionID', 'education'), ('camgaignID', 'connectionType'), 
                ('positionID', 'connectionType'), ('userID', 'adID'), ('creativeID', 'connectionType'), 
                ('userID', 'connectionType'),('creativeID', 'gender'), ('positionID', 'userID'), 
                ('positionID', 'camgaignID'), ('age',  'gender'), ('camgaignID', 'age' ), 
                ('adID', 'connectionType'), ('camgaignID', 'gender'), ('userID', 'appCategory'),
                ( 'advertiserID', 'connectionType' ), ('positionID', 'adID'), ('positionID', 'appCategory'), 
                ('positionID', 'haveBaby'),("positionID", "clickhour_min"),("positionID", "clickhour")]
    trick_con_features = []
    for i in trick_features:
        trick_con_features.append((i,"positionID"))
        trick_con_features.append((i,"appID"))
    trick_con_features = trick_con_features + [('creativeID_rank_day', 'connectionType'), ('creativeID_rank_day', 'connectionType', 'positionID'), ('creativeID_rank_day', 'appID', 'positionID')]
    prior_prob = train["label"].mean()
    # for f_id in basic_features:
    #     designate_less_observations(train, test, 5, f_id)

# funcs_list=[]
print "start pcvr(smoothing) calculation:"
val_df = []
if os.path.exists(DIR + "/train_pcvr."+str(begin_day)+"-30.csv")==0 or correct:
    for i in range(31)[begin_day:]:
        if os.path.exists(train_file+".pcvr."+str(i)) and correct == 0:
            val_ = pd.read_csv(train_file+".pcvr."+str(i))
        else:
            print "\nday", i
            val_ = train[train["day"] == i]
            train_ = train[train["day"] < i]
            for f_id in basic_features + trick_features:

                # Bayes smoothing
                # val_ = smooth(train_, val_, [f_id])

                # High-Cardinality Categorical Attributes
                hcc_encode(train_, val_, [f_id], "label", prior_prob, 5, f=1, g=1, r_k=None, update_df=None)
        
                """
                # parallel
                # name = '_'.join(f_id)
                ## train_f_id = train_[list(f_id) + ["label"]]
                # funcs_list.append([pcvr, (train_, val_, list(f_id), "train_"+name+str(i)+".csv")])
                """

            for f_id in con_features + trick_con_features:

                # Bayes smoothing
                # val_ = smooth(train_, val_, list(f_id))

                # High-Cardinality Categorical Attributes
                hcc_encode(train_, val_, list(f_id), "label", prior_prob, 5, f=1, g=1, r_k=None, update_df=None)
    
            val_.to_csv(train_file+".pcvr."+str(i), index=False)
        
        val_df.append(val_)
    train_new = pd.concat(val_df)
    train_new = train_new.sort_values(by=['instanceID'])
    train_new.to_csv(DIR + "/train_pcvr."+str(begin_day)+"-30.csv", index=False)

# test #
if os.path.exists(DIR + "/test_pcvr.csv") == 0 or correct:
    print "\ntest"
    for f_id in basic_features + trick_features:
    
        # Bayes smoothing
        # test = smooth(train, test, [f_id])

        # High-Cardinality Categorical Attributes
        hcc_encode(train, test, [f_id], "label", prior_prob, 5, f=1, g=1, r_k=None, update_df=None)
    
        """
        # parallel
        # name = '_'.join(f_id)
        ## train_f_id = train[list(f_id) + ["label"]]
        # funcs_list.append([pcvr, (train_, test, f_id, "test_"+name+".csv")])
        """
    for f_id in con_features + trick_con_features:

        # Bayes smoothing
        # test = smooth(train, test, list(f_id))

        # High-Cardinality Categorical Attributes
        hcc_encode(train, test, list(f_id), "label", prior_prob, 5, f=1, g=1, r_k=None, update_df=None)

    test = test.sort_values(by=['instanceID'])
    test.to_csv(DIR + "/test_pcvr.csv", index=False)

    # applyParallel_feature(iter(funcs_list), execu) 
print ('\npcvr calculation finish.')
