#!/bin/bash

# clean data of day 30
python clean_30.py ../data/

# calculate rank of repeated data
python cal_rank.py ../data/ appID #(userID, appID)
python cal_rank.py ../data/ creativeID #(userID, creativeID)

# calculate time difference between two click
python cal_time_diff.py ../data/

# calculate count of feature using data before the day with a window of 7
python cal_count_window.py ../data/ 28 7

# calculate cvr of feature using all data before the day 
python cal_pcvr_before.py ../data/ 28

# calculate history of user in user_app_actions.csv and user_installedapps.csv
sh hist_pre.sh
