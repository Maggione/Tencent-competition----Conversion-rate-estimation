#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
import sys
import math
import csv

reload(sys) 
sys.setdefaultencoding('utf8')
def load_fcn_result(filename):
        rfile = file(filename, 'rb')
        reader = csv.reader(rfile)
        fcn_result = []
        i = 0
        for line in reader:
		    if i > 0:
                	fcn_result.append([float(x) for x in line[0:2]])
		    i = i + 1
        return np.array(fcn_result)

def load_fea(filename):
	rfile = file(filename, 'rb')
	reader = csv.reader(rfile)
	features = []
	i = 0
	for line in reader:
		if i > 0:
			features.append([float(x) for x in line[4:]])
		i = i + 1
	return np.array(features)

result0 = load_fcn_result("./sub_prior.csv")
result1 = load_fcn_result("./sub_prior1.csv")
# result2 = load_fcn_result("./final_random_28-29/submission.2.csv")
# result3 = load_fcn_result("./final_random_28-29/submission.3.csv")
# result4 = load_fcn_result("./final_random_28-29/submission.4.csv")
# result5 = load_fcn_result("./final_random_28-29/submission.5.csv")
# result6 = load_fcn_result("./final_random_28-29/submission.6.csv")
# result7 = load_fcn_result("./final_random_28-29/submission.7.csv")
# result8 = load_fcn_result("./final_random_28-29/submission.8.csv")
# result9 = load_fcn_result("./final_random_28-29/submission.9.csv")
# result1 = load_fcn_result("xgb/lgb_submission_prior.fix.25.csv")
# result2 = load_fcn_result("xgb/lgb_submission_prior.fix.26.csv")
# result3 = load_fcn_result("xgb/lgb_submission_prior.win.csv")
# result4 = load_fcn_result("lr/lgb_submission.csv")

result = []
r = file("./sub_prior2.csv", "wb")
writer = csv.writer(r)
writer.writerow(["instanceID", "prob"])
for i in range(result0.shape[0]):
	listing_id = int(result0[i][0])
        score = (result0[i][1] + result1[i][1])/2 # + result2[i][1]  + result3[i][1] + result4[i][1] + result5[i][1] +result6[i][1] + result7[i][1] + result8[i][1] + result9[i][1])/10
	writer.writerow([listing_id, score])
