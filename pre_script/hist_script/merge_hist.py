#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import sys
print "脚本名：",sys.argv[0]

DIR = sys.argv[1]
begin_day = sys.argv[2]
win = sys.argv[3]

print "-----merge features to train.csv-----"

# origin file
train_file = DIR + "train.csv.good"
test_file = DIR + "test_small.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
print "origin:", train.shape, test.shape

ad_df = pd.read_csv(DIR + "ad.csv")
app_df = pd.read_csv(DIR + 'app_categories.csv')
train = pd.merge(train,ad_df,on="creativeID",how="left")
test = pd.merge(test,ad_df,on="creativeID",how="left")
train = pd.merge(train,app_df,on="appID",how="left")
test = pd.merge(test,app_df,on="appID",how="left")

del ad_df
del app_df

train["day"] = train["clickTime"].apply(lambda x : x/1000000)
test["day"] = test["clickTime"].apply(lambda x: x/1000000)

train = train[["instanceID", "userID", "appID", "appCategory", "day"]]
test = test[["instanceID", "userID", "appID", "appCategory", "day"]]

appc_size = 31
dictionary = {0: 0, 1: 1, 2: 2, 401: 22, 402: 23, 403: 24, 405: 25, 406: 26, 407: 27, 
             408: 28, 409: 29, 301: 20, 303: 21, 201: 13, 203: 14, 204: 15, 205: 16, 
             209: 17, 210: 18, 211: 19, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 
             107: 9, 108: 10, 109: 11, 110: 12, 503: 30}

train["appCategory1"] = train["appCategory"].astype(str).apply(lambda x: x[0]).astype(int)
test["appCategory1"] = test["appCategory"].astype(str).apply(lambda x: x[0]).astype(int)

train["appCategory"] = train["appCategory"].apply(lambda x: dictionary[x])
test["appCategory"] = test["appCategory"].apply(lambda x: dictionary[x])

# user_action
print "merge user_action..."
train_df = []
for day in range(32)[int(begin_day):]:

	app_actioned_file = DIR + "/user_app_actions/app_actioned.csv.w" + win + "." + str(day)
	user_app_actioned_file = DIR + "/user_app_actions/user_app_actioned.csv.w" + win + "." + str(day) + ".c"
	app_actioned = pd.read_csv(app_actioned_file)
	user_app_actioned = pd.read_csv(user_app_actioned_file)

	if day == 31:
		train_ = test
	else:
		train_ = train[train["day"]==day]
	train_ = train_.merge(app_actioned, on="appID", how="left")
	train_ = train_.merge(user_app_actioned, on="userID", how="left")

	drop_features=[]
	for i in range(appc_size):
		col = "a"+str(i)
		drop_features.extend([col, col+"_rate"])
		train_[col+"_rate"] = train_[col]/train_["a_size"]

	# the times of user and appC appearing at the same time in user_apps_actions.csv
	train_["user_appC_a_rate"] = train_.apply(lambda x: x["a"+str(int(x["appCategory"]))+"_rate"], axis=1)
	train_["user_appC_a"] = train_.apply(lambda x: x["a"+str(int(x["appCategory"]))], axis=1)
	train_ = train_.drop(drop_features, axis=1)
	if day == 31:
		test = train_
	else:
		train_df.append(train_)

train_new = pd.concat(train_df)
del app_actioned
del user_app_actioned

# user_install
print "merge user_install..."
app_installed_file = DIR + "/user_installedapps/app_installed.csv"
user_app_installed_file = DIR + "/user_installedapps/user_app_installed.csv.c"
app_installed = pd.read_csv(app_installed_file)
user_app_installed = pd.read_csv(user_app_installed_file)

train_new = train_new.merge(app_installed, on="appID", how="left")
train_new = train_new.merge(user_app_installed, on="userID", how="left")
test = test.merge(app_installed, on="appID", how="left")
test = test.merge(user_app_installed, on="userID", how="left")
del app_installed
del user_app_installed

drop_features=['userID','appID','appCategory','day']
for i in range(appc_size):
	col = "i"+str(i)
	drop_features.extend([col, col+"_rate"])
	train_new[col+"_rate"] = train_new[col]/train_new["i_size"]
	test[col+"_rate"] = test[col]/test["i_size"]

# the times of user and appC appearing at the same time in user_installedapps.csv
train_new["user_appC_i_rate"] = train_new.apply(lambda x: x["i"+str(int(x["appCategory"]))+"_rate"], axis=1)
train_new["user_appC_i"] = train_new.apply(lambda x: x["i"+str(int(x["appCategory"]))], axis=1)
train_new = train_new.drop(drop_features, axis=1)
test["user_appC_i_rate"] = test.apply(lambda x: x["i"+str(int(x["appCategory"]))+"_rate"], axis=1)
test["user_appC_i"] = test.apply(lambda x: x["i"+str(int(x["appCategory"]))], axis=1)
test = test.drop(drop_features, axis=1)

print "get hist features:", ','.join(test.columns.values[1:])

train_new.to_csv(DIR + "train_history."+begin_day+"-30.csv", index=False)
test.to_csv(DIR + "test_history.csv", index=False)
