#!/bin/bash

# app_actioned
python hist_script/cal_app_actioned.py 28 7 
python hist_script/cal_app_actioned.py 29 7 
python hist_script/cal_app_actioned.py 30 7 
python hist_script/cal_app_actioned.py 31 7 

# user_app_actioned
python hist_script/cal_user_app_actioned.py 28 7 
python hist_script/cal_user_app_actioned.py 29 7 
python hist_script/cal_user_app_actioned.py 30 7 
python hist_script/cal_user_app_actioned.py 31 7 

# app_installed
python hist_script/cal_app_installed.py

# user_app_installed
python hist_script/cal_user_app_installed.py

# categ
python hist_script/cal_categ.py ../data/user_installedapps/user_app_installed.csv i
python hist_script/cal_categ.py ../data/user_app_actions/user_app_actioned.csv.w7.28 a
python hist_script/cal_categ.py ../data/user_app_actions/user_app_actioned.csv.w7.29 a
python hist_script/cal_categ.py ../data/user_app_actions/user_app_actioned.csv.w7.30 a
python hist_script/cal_categ.py ../data/user_app_actions/user_app_actioned.csv.w7.31 a

python hist_script/merge_hist.py ../data/ 28 7



