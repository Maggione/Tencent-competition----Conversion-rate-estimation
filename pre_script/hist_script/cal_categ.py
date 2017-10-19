#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import csv
import time
import sys
print "脚本名：",sys.argv[0]
"""
Get features: i/a0, i/a1, i/a2, ...
Calculate the number of times of each kind of appCategory is installed
"""
start_time = time.time()
print "-----Get features: i/a0, i/a1, i/a2, ...-----"

# path and for install or action
inputfile = sys.argv[1]
i_or_a = sys.argv[2]

# init the dict
"""
# get appcategory_dict
appcategory = pd.read_csv("../app_categories.csv")
grouped = appcategory.groupby(["appCategory"]).size().reset_index().rename(columns={0:"size"})
print(grouped.shape)
grouped.to_csv("../appCategory_Size.csv", index=False)

appc_size = grouped.shape[0]
appc = grouped["appCategory"].tolist()

appc_dict = dict()
for i in range(len(appc)):
    appc_dict[appc[i]] = i
print appc_dict
del appcategory
"""
# once you get the dict
appc_size = 31
appc_dict = {0: 0, 1: 1, 2: 2, 401: 22, 402: 23, 403: 24, 405: 25, 406: 26, 407: 27, 408: 28, 409: 29, 301: 20, 303: 21, 201: 13, 203: 14, 204: 15, 205: 16, 209: 17, 210: 18, 211: 19, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 12, 503: 30}
title = ['userID',i_or_a+"_size"]
for i in range(appc_size):
	title.append(i_or_a+str(i))

rfile = file(inputfile, "rb")
reader = csv.reader(rfile)
wfile = file(inputfile+".c", 'wb')
writer = csv.writer(wfile)

i = 0
for line in reader:
	print "confirm the value is userID :", line[0]
	print "confirm the value is C :", line[2]
	break
writer.writerow(title)

print "finished:"
for line in reader:
	i = i + 1
	print '\r%d' % (i),
	appC = [0]*appc_size
	app_c = [int(x) for x in line[2].split('_')[1:]]
	for c in app_c:
		appC[appc_dict[c]] += 1
	writer.writerow([line[0],line[-1]]+appC)

print "\n" + inputfile + "_" + i_or_a + " done, time used = {0:.3f}".format(time.time()-start_time)







