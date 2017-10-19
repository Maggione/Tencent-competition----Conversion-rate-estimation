#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
import sys
import csv
import os
print "脚本名：",sys.argv[0]

DIR = sys.argv[1]

if not os.path.exists("tmp"):
    os.mkdir("tmp")

train = pd.read_csv(DIR + "/train_small.csv")
train_columns = train.columns.values.tolist()
ad_df = pd.read_csv(DIR + "/ad.csv")

train["day"] = train["clickTime"].apply(lambda x : x/1000000)
train = train.merge(ad_df, on="creativeID", how="left")

train_before = train[train["day"] < 30]
train_30 = train[train["day"] == 30]
train_before.to_csv("./tmp/train_before.csv", index=False)
train_30.to_csv("./tmp/train_30.csv", index=False)

train_before = train_before[["label", "appID"]]
train_30 = train_30[["label", "appID"]]

group_before = train_before.groupby(["appID"]).mean().reset_index().rename(columns={'label':'pcvr'})
group_30 = train_30.groupby(["appID"]).mean().reset_index().rename(columns={'label':'cvr'})

group_30 = group_30.merge(group_before, on="appID", how="left")

def find_bad(row):
	if row["cvr"] == 0 and row["pcvr"] > 0:
		return 1
	else:
		return 0

group_30["bad"] = group_30.apply(find_bad, axis=1)
bad_list = group_30[group_30["bad"]==1]["appID"].tolist()
print "bad_list:", bad_list

rfile = file("./tmp/train_30.csv", "rb")
csv_reader = csv.reader(rfile)
wfile = file("./tmp/train_30.csv.good", "wb")
csv_writer = csv.writer(wfile)

for line in csv_reader:
    print "confirm the value is appID:", line[-2]
    csv_writer.writerow(line)
    break

for line in csv_reader:
    if int(line[-2]) not in bad_list:
        csv_writer.writerow(line)

wfile.close()
rfile.close()

train_30 = pd.read_csv("./tmp/train_30.csv.good")
train_before = pd.read_csv("./tmp/train_before.csv")

train = pd.concat([train_before, train_30])
train = train[train_columns]
train.to_csv(DIR + "/train.csv.good", index=False)
train_30 = train_30[["instanceID"]]
train_30.to_csv(DIR + "/instanceID_of_30.csv", index=False)

