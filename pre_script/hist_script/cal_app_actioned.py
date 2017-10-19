#!/usr/bin/env python
# coding=utf-8
"""
Get features: app_actioned
Calculate the number of times each appID is installed in app_actions.csv
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

print "-----Get features: app_actioned-----"

# inital
day = int(sys.argv[1])
windows = int(sys.argv[2])
#[down_time, up_time)
up_time = day*1000000
down_time = (day - windows)*1000000

# counting
# header-- userID,installTime,appID
print "start counting..."
rfile = file(DIR+filename, "rb")
csv_reader = csv.reader(rfile)

i = 0
app_count = dict()
for line in csv_reader:
    print "confirm the value is installTime : " + line[1] 
    print "confirm the value is appID : " + line[2]
    break

print "finished:"

for line in csv_reader:
    i = i + 1
    print '\r%d' % (i),
    if int(line[1]) < down_time:
        continue
    if int(line[1]) >= up_time:
        continue
    if line[2] not in app_count:
        app_count[line[2]] = 1
    else:
        app_count[line[2]] += 1

# write in file
print "\nstart writing..."
wfile = file(DIR+"app_actioned.csv.w"+str(windows)+"."+str(day), "wb")
csv_writer = csv.writer(wfile)
key = app_count.keys()

csv_writer.writerow(["appID", "app_actioned"])
for i in key:
    csv_writer.writerow([i, app_count[i]])
wfile.close()

print "app_actioned." + str(day) + "-w" + str(windows) + " done, time used = {0:.3f}".format(time.time()-start_time)

