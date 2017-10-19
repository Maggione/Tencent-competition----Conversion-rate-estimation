#!/usr/bin/env python
#i coding=utf-8
"""
Get features: user_app_actioned, user_appC_actioned, user_action_size
Calculate the number of times each appID is installed by user in history
"""
import pandas as pd
import csv
import sys
import time
print "脚本名：",sys.argv[0]
start_time = time.time()
# path
DIR = "../data/user_app_actions/"
filename = "user_app_actions_small.csv"

print "-----Get features: user_app_actioned, user_appC_actioned, user_action_size-----"

# app_category dict
app_category_df = pd.read_csv("../data/app_categories.csv")
appid = app_category_df["appID"].astype(str).tolist()
app_category = app_category_df["appCategory"].astype(str).tolist()
app_category_dict = dict()
for i in range(len(appid)):
    app_category_dict[appid[i]] = app_category[i]
print "get dict time used = {0:.3f}".format(time.time()-start_time)

# inital
day = int(sys.argv[1])
windows = int(sys.argv[2])
#[down_time, up_time)
up_time = day*1000000
down_time = (day - windows)*1000000

# count and write
# header-- userID,installTime,appID
print "start counting and writing"
rfile = file(DIR+filename, "rb")
csv_reader = csv.reader(rfile)
wfile = file(DIR+"user_app_actioned.csv.w"+str(windows)+"."+str(day), "wb")
csv_writer = csv.writer(wfile)

i = 0
for line in csv_reader:
    print "confirm the value is user : " + line[0]
    print "confirm the value is installTime : " + line[1] 
    print "confirm the value is appID : " + line[2]
    break

print "finished:"
user_id = "userID"
app_str = "user_app_actioned"
appC_str = "user_appC_actioned"
user_action_size = "user_action_size"
for line in csv_reader:
    i = i + 1
    print '\r%d' % (i),
    if int(line[1]) < down_time:
        continue
    if int(line[1]) >= up_time:
        continue
    if line[0] != user_id:
        csv_writer.writerow([user_id,app_str,appC_str,user_action_size])
        app_str = ""
        appC_str = ""
        user_action_size = 0
    user_id = line[0]
    app_str = app_str + "_" + str(line[2])
    appC_str = appC_str + "_" + app_category_dict[line[2]]
    user_action_size += 1
csv_writer.writerow([user_id, app_str, appC_str, user_action_size])

rfile.close()
wfile.close()
print "user_app_actioned." + str(day) + "-w" + str(windows) + " done, time used = {0:.3f}".format(time.time()-start_time)

