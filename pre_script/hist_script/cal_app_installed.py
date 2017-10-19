#!/usr/bin/env python
#i coding=utf-8

"""
Get features: app_installed
Calculate the number of times each appID is installed in history
"""

import pandas as pd
import csv
import time
import sys
print "脚本名：",sys.argv[0]

start_time = time.time()

print "-----Get features: app_installed-----"

# path
DIR = "../data/user_installedapps/"
filename = "user_installedapps_small.csv"

# counting
# header -- userID,appID
print "start counting..." 
rfile = file(DIR+filename, "rb")
csv_reader = csv.reader(rfile)

i = 0
app_count = dict()
for line in csv_reader:
    print "confirm the value that is counted is appID : " + line[1]
    break

print "finished:"

for line in csv_reader:
    if line[1] not in app_count:
        app_count[line[1]] = 1
    else:
        app_count[line[1]] += 1
    i = i + 1
    print '\r%d' % (i),

# write in file
print "\nstart writing ..."
wfile = file(DIR+"/app_installed.csv", "wb")
csv_writer = csv.writer(wfile)
key = app_count.keys()

csv_writer.writerow(["appID","app_installed"])

for i in key:
    csv_writer.writerow([i,app_count[i]])

wfile.close()

print "app_installed done"
