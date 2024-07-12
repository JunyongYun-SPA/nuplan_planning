#!/bin/bash

MAX_TEMP=94 # 최대 허용 온도
SLEEP_INTERVAL=6 # 모니터링 간격(초)

while true; do
    # 모든 GPU의 온도를 읽어옵니다.
    IFS=$'\n' # 줄바꿈 문자를 필드 구분자로 설정
    GPU_TEMPS=($(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits))

    for i in "${!GPU_TEMPS[@]}"; do
        GPU_TEMP=${GPU_TEMPS[$i]}
        if [[ $GPU_TEMP -ge $MAX_TEMP ]]; then
            echo "GPU $i 의 온도가 $MAX_TEMP°C를 초과했습니다: 현재 온도는 $GPU_TEMP°C"
	    PIDS=$(nvidia-smi | grep 'python' | awk '{ print $5 }')

	   for PID in $PIDS; do
               echo "Killing Python process with PID: $PID"
               sudo kill $PID
	   done

	   docker kill nuplan_bev

            break 2 # 두 개의 루프 모두를 종료합니다.
        fi
    done

    sleep $SLEEP_INTERVAL
done






exit 0
