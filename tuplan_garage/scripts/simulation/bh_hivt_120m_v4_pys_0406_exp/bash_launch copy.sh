#!/bin/bash
# python sim_OLS_ver8.py &&
# python sim_OLS_ver8_v3.py &&
python sim_OLS_ver8_v4.py &&
# python sim_OLS_ver9.py &&
python sim_OLS_ver9_v2.py &&
python sim_OLS_ver8_v2.py


# 1. route를 안 벗어나는 경로 선택 후, confidence score로 선택
# 2. confidence score threshold 적용 후, route distance를 기준으로 선택
# 3. Route_distance + confidence score
# 4. GT와 가장 가까운 모달
# 5. confidence score로 선택