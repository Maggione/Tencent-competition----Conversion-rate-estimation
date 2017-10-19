#!/usr/bin/env python
#i coding=utf-8
"""
Get features: user_app_installed, user_appC_installed, user_install_size
Analyze the appID is installed by user in history
"""
import pandas as pd
import csv
import time
import sys

print "脚本名：",sys.argv[0]
start_time = time.time()

# path
DIR = "../data/user_installedapps/"
filename = "user_installedapps_small.csv"

print "-----Get features: user_app_installed, user_appC_installed, user_install_size-----"

# app_category dict
app_category_df = pd.read_csv("../data/app_categories.csv")
appid = app_category_df["appID"].astype(str).tolist()
app_category = app_category_df["appCategory"].astype(str).tolist()
app_category_dict = dict()
for i in range(len(appid)):
    app_category_dict[appid[i]] = app_category[i]
print "get dict time used = {0:.3f}".format(time.time()-start_time)


# count and write
# header-- userID,appID
print "start counting and writing"
rfile = file(DIR+filename, "rb")
csv_reader = csv.reader(rfile)
wfile = file(DIR+"user_app_installed.csv", "wb")
csv_writer = csv.writer(wfile)

i = 0
for line in csv_reader:
    print "confirm the value is userID :", line[0]
    print "confirm the value is appID :", line[1]
    break

print "finished:"
user_id = "userID"
app_str = "user_app_installed"
appC_str = "user_appC_installed"
user_install_size = "user_install_size"
for line in csv_reader:
    if line[0] != user_id:
        csv_writer.writerow([user_id,app_str, appC_str, user_install_size])
        app_str = ""
        appC_str = ""
        user_install_size = 0
    user_id = line[0]
    app_str = app_str + "_" + str(line[1])
    appC_str = appC_str + "_" + app_category_dict[line[1]]
    user_install_size += 1
    i = i + 1
    print '\r%d' % (i),
csv_writer.writerow([user_id, app_str, appC_str, user_install_size])

rfile.close()
wfile.close()
print "\nuser_app_installed done, time used = {0:.3f}".format(time.time()-start_time)

