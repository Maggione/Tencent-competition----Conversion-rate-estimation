#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import gc
import sys
import os
print "脚本名：",sys.argv[0]
DIR = sys.argv[1]

# time
def delta_disperse(delta):
    if delta == -1: return 0
    elif delta == 0: return 1
    elif delta < 60: return 2
    elif delta < 10 * 60: return 3
    elif delta < 60 * 60: return 4
    elif delta < 60 * 60 * 6: return 5
    elif delta < 60 * 60 * 24: return 6
    elif delta < 60 * 60 * 24 * 3: return 7
    elif delta < 60 * 60 * 24 * 7: return 8
    elif delta < 60 * 60 * 24 * 11: return 9
    else: return 10

# time difference from the last click
def get_min_click_ID_time_diff(x):
    x['user'] = x['userID'].shift(1) 
    x['min_user_click_diff'] = x['clickTime']- x['clickTime'].shift(1)
    x.loc[x.user!=x.userID, 'min_user_click_diff'] = -1
    x['min_user_click_diff1'] = x['min_user_click_diff'].apply(lambda x : delta_disperse(x))
    return x.drop('user', axis=1)
def get_min_click_IDS_time_diff(x):
    x['user'] = x['userID'].shift(1) 
    x['creative'] = x['creativeID'].shift(1)
    x['min_user_creativeid_click_diff'] = x['clickTime']- x['clickTime'].shift(1)
    x.loc[((x.user!=x.userID) | (x.creative!=x.creativeID)), 'min_user_creativeid_click_diff'] = -1
    x['min_user_creativeid_click_diff1'] = x['min_user_creativeid_click_diff'].apply(lambda x : delta_disperse(x))
    return x.drop(['user', 'creative'], axis=1)
def get_min_click_IDS_app_time_diff(x):
    x['user'] = x['userID'].shift(1) 
    x['app'] = x['appID'].shift(1)
    x['min_user_appid_click_diff'] = x['clickTime']- x['clickTime'].shift(1)
    x.loc[((x.user!=x.userID) | (x.app!=x.appID)), 'min_user_appid_click_diff'] = -1
    x['min_user_appid_click_diff1'] = x['min_user_appid_click_diff'].apply(lambda x : delta_disperse(x))
    return x.drop(['user', 'app'], axis=1)

# time difference from the next click
def get_min_next_click_ID_time_diff(x):
    x['user'] = x['userID'].shift(-1) 
    x['min_user_next_click_diff'] = x['clickTime'].shift(-1)- x['clickTime']
    x.loc[x.user!=x.userID, 'min_user_next_click_diff'] = -1
    x['min_user_next_click_diff1'] = x['min_user_next_click_diff'].apply(lambda x : delta_disperse(x))
    return x.drop('user', axis=1)
def get_min_next_click_IDS_time_diff(x):
    x['user'] = x['userID'].shift(-1) 
    x['creative'] = x['creativeID'].shift(-1)
    x['min_user_creativeid_next_click_diff'] = x['clickTime'].shift(-1)- x['clickTime']
    x.loc[((x.user!=x.userID) | (x.creative!=x.creativeID)), 'min_user_creativeid_next_click_diff'] = -1
    x['min_user_creativeid_next_click_diff1'] = x['min_user_creativeid_next_click_diff'].apply(lambda x : delta_disperse(x))
    return x.drop(['user', 'creative'], axis=1)
def get_min_next_click_IDS_app_time_diff(x):
    x['user'] = x['userID'].shift(-1) 
    x['app'] = x['appID'].shift(-1)
    x['min_user_appid_next_click_diff'] = x['clickTime'].shift(-1)- x['clickTime']
    x.loc[((x.user!=x.userID) | (x.app!=x.appID)), 'min_user_appid_next_click_diff'] = -1
    x['min_user_appid_next_click_diff1'] = x['min_user_appid_next_click_diff'].apply(lambda x : delta_disperse(x))
    return x.drop(['user', 'app'], axis=1)


train = pd.read_csv(DIR + "/train_small.csv")
test = pd.read_csv(DIR + "/test_small.csv")
ad = pd.read_csv(DIR + "/ad.csv")
train = train.merge(ad, on="creativeID", how="left")
test = test.merge(ad, on="creativeID", how="left")
train = train[['instanceID','clickTime','userID','creativeID','appID']]
test = test[['instanceID','clickTime','userID','creativeID','appID']]
train["source"] = 1
test["source"] = 0
train["day"] = train["clickTime"].apply(lambda x: x/1000000)
test["day"] = test["clickTime"].apply(lambda x: x/1000000)

actions = train.append(test,ignore_index=True)
actions["clickTime"] = actions["clickTime"].apply(lambda x: x/1000000*24*60*60+ x%1000000/10000*60*60 + x%10000/100*60 + x%100) 
actions = actions.sort_values(by=['userID', 'clickTime'])
actions = get_min_click_ID_time_diff(actions)
actions = get_min_next_click_ID_time_diff(actions)

actions = actions.sort_values(by=['userID', 'creativeID', 'clickTime'])
actions = get_min_click_IDS_time_diff(actions)
actions = get_min_next_click_IDS_time_diff(actions)

actions = actions.sort_values(by=['userID', 'appID', 'clickTime'])
actions = get_min_click_IDS_app_time_diff(actions)
actions = get_min_next_click_IDS_app_time_diff(actions)

train = actions[actions["source"] == 1]
test = actions[actions["source"] == 0]

train = train.drop(["source","clickTime",'userID','creativeID','appID'],axis=1)
test = test.drop(["source","clickTime",'userID','creativeID','appID'],axis=1)

train_before = train[train["day"]<30]
train_30 = train[train["day"]==30]
instanceID_of_30 = pd.read_csv(DIR + "/instanceID_of_30.csv")
train_30 = instanceID_of_30.merge(train_30, on="instanceID", how="left")
train = pd.concat([train_before, train_30])

train = train.sort_values(by=['instanceID'])
test = test.sort_values(by=['instanceID'])

train.to_csv(DIR + "/train_time_diff.csv", index=False)
test.to_csv(DIR + "/test_time_diff.csv", index=False)

print ('finish calculating time difference.')
