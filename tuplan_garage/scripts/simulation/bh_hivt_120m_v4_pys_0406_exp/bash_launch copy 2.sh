#!/bin/bash
# python sim_OLS_exp1.py &&
# python sim_OLS_exp2.py &&
# python sim_OLS_exp3.py &&
# python sim_OLS_exp4.py &&
# python sim_OLS_exp5.py 

python sim_CLS_NR_ver8_v4_loss_add_features.py &&
python sim_CLS_R_ver8_v4_loss_add_features.py


# 1. route를 안 벗어나는 경로 선택 후, confidence score로 선택
# 2. confidence score threshold 적용 후, route distance를 기준으로 선택
# 3. Route_distance + confidence score
# 4. GT와 가장 가까운 모달
# 5. confidence score로 선택
