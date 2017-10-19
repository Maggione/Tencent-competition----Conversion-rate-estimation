#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import csv
import sys
import os
print "脚本名：",sys.argv[0]

DIR = sys.argv[1]
app_or_creativeID = sys.argv[2]

if not os.path.exists("tmp"):
    os.mkdir("tmp")

train_path = DIR + "train_small.csv"
test_path = DIR + "test_small.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print train.shape, test.shape
train_len = train.shape[0]

if app_or_creativeID == "appID":
    ad_df = pd.read_csv(DIR + "ad.csv")
    train = pd.merge(train,ad_df,on="creativeID",how="left")
    test = pd.merge(test,ad_df,on="creativeID",how="left")

train["day"] = train["clickTime"].apply(lambda x: x/1000000)
test["day"] = test["clickTime"].apply(lambda x: x/1000000)

train[app_or_creativeID+"_rank_day"] = train["day"].astype(str)+'_'+train[app_or_creativeID].astype(str)+'_'+train["userID"].astype(str)+'_'+train["positionID"].astype(str) 
test[app_or_creativeID+"_rank_day"] = test["day"].astype(str)+'_'+test[app_or_creativeID].astype(str)+'_'+test["userID"].astype(str)+'_'+test["positionID"].astype(str) 

train[app_or_creativeID+"_rank"] = train[app_or_creativeID].astype(str)+'_'+train["userID"].astype(str)+'_'+train["positionID"].astype(str) 
test[app_or_creativeID+"_rank"] = test[app_or_creativeID].astype(str)+'_'+test["userID"].astype(str)+'_'+test["positionID"].astype(str) 

# test["instanceID"] = test["instanceID"] + train.shape[0] - 1

train = train[["instanceID", app_or_creativeID+"_rank_day", app_or_creativeID+"_rank", "day"]]
test = test[["instanceID", app_or_creativeID+"_rank_day", app_or_creativeID+"_rank", "day"]]

all = train.append(test,ignore_index=True)
all.to_csv("./tmp/all_for_rank.csv", index=False)

print "ready for rank..."

all_file = file("./tmp/all_for_rank.csv", "rb")
all_reader = csv.reader(all_file)
rank_file = file("./tmp/rank.csv", "wb")
rank_writer = csv.writer(rank_file)
for_rank_day = dict()
for_rank = dict()
for line in all_reader:
    rank_writer.writerow(line+["rank_day", "rank"])
    print line[1],line[2]
    break
i = 0
for line in all_reader:
    i = i + 1
    print '\r%d' % (i),
    if line[1] not in for_rank_day:
        for_rank_day[line[1]] = 1
    else:
        for_rank_day[line[1]] += 1
    if line[2] not in for_rank:
        for_rank[line[2]] = 1
    else:
        for_rank[line[2]] += 1
    rank_writer.writerow(line + [for_rank_day[line[1]], for_rank[line[2]]])

rank_file.close()
all_file.close()
print "\ncreate rank.csv"

rank_file = file("./tmp/rank.csv", "rb")
rank_reader = csv.reader(rank_file)
for line in rank_reader:
    print ' '.join(line)
    break

train_rank_file = file("./tmp/train_rank.csv", "wb")
train_rank_writer = csv.writer(train_rank_file)
test_rank_file = file("./tmp/test_rank.csv", "wb")
test_rank_writer = csv.writer(test_rank_file)

train_rank_writer.writerow([line[0], line[3], line[4], line[5], "is_duplicate_day", "is_duplicate", "is_last_day", "is_last"])
test_rank_writer.writerow([line[0], line[3], line[4], line[5], "is_duplicate_day", "is_duplicate", "is_last_day", "is_last"])
i = 0
for line in rank_reader:
    i = i + 1
    print '\r%d' % (i),
    rank_last_day = 1
    rank_last = 1
    is_duplicate_day = 1
    is_duplicate = 1
    if for_rank_day[line[1]] == int(line[3]):
        rank_last_day = -1
    if for_rank[line[2]] == int(line[4]):
        rank_last = -1
    if for_rank_day[line[1]] == 1:
        is_duplicate_day = 0
    if for_rank[line[2]] == 1:
        is_duplicate = 0
    if i <= train_len:
        train_rank_writer.writerow([line[0], line[3], line[4], line[5], is_duplicate_day, is_duplicate, rank_last_day, rank_last])
    else:
        test_rank_writer.writerow([line[0], line[3], line[4], line[5], is_duplicate_day, is_duplicate, rank_last_day, rank_last])

train_rank_file.close()
test_rank_file.close()
rank_file.close()

print "\nready for sign"

train_rank = pd.read_csv("./tmp/train_rank.csv")
test_rank = pd.read_csv("./tmp/test_rank.csv")

train_rank[app_or_creativeID+"_rank"] = train_rank["rank"] - 1
train_rank[app_or_creativeID+"_rank_day"] = train_rank["rank_day"] * train_rank["is_duplicate_day"]
train_rank[app_or_creativeID+"_rank_day_sign"] = train_rank["rank_day"] * train_rank["is_last_day"]

test_rank[app_or_creativeID+"_rank"] = test_rank["rank"] - 1
test_rank[app_or_creativeID+"_rank_day"] = test_rank["rank_day"] * test_rank["is_duplicate_day"]
test_rank[app_or_creativeID+"_rank_day_sign"] = test_rank["rank_day"] * test_rank["is_last_day"]

def sign(x):
    if x < 0:
        return 3
    elif x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return 2

train_rank[app_or_creativeID+"_rank_day_sign"] = train_rank[app_or_creativeID+"_rank_day_sign"].apply(lambda x:sign(x))
test_rank[app_or_creativeID+"_rank_day_sign"] = test_rank[app_or_creativeID+"_rank_day_sign"].apply(lambda x:sign(x))

train_rank.drop(["rank", "rank_day", "is_duplicate", "is_duplicate_day", "is_last", "is_last_day"], axis=1, inplace=True)
test_rank.drop(["rank", "rank_day", "is_duplicate", "is_duplicate_day", "is_last", "is_last_day"], axis=1, inplace=True)

train_rank_before = train_rank[train_rank["day"]<30]
train_rank_30 = train_rank[train_rank["day"]==30]
instanceID_of_30 = pd.read_csv(DIR + "/instanceID_of_30.csv")
train_rank_30 = instanceID_of_30.merge(train_rank_30, on="instanceID", how="left")
train_rank = pd.concat([train_rank_before, train_rank_30])

train = train.sort_values(by=['instanceID'])
test = test.sort_values(by=['instanceID'])

train_rank.to_csv(DIR + "/train_rank_"+app_or_creativeID+".csv", index=False)
test_rank.to_csv(DIR + "/test_rank_"+app_or_creativeID+".csv", index=False)
print "finish rank " + app_or_creativeID

