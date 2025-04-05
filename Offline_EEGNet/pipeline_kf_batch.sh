#!/bin/bash
source ~/.offeeg/bin/activate

# See https://stackoverflow.com/questions/3004811/how-do-you-run-multiple-programs-in-parallel-from-a-bash-script/52033580#52033580
(trap 'kill 0' SIGINT; \
python pipeline_kf.py config_2023-09-19_2023-08-15_A2_OL_1_MSE_64-1.yaml & \
python pipeline_kf.py config_2023-09-19_2023-08-15_A2_OL_1_MSE_16-2.yaml & \
python pipeline_kf.py config_2023-09-19_2023-08-15_A2_OL_1_MSE_8-2.yaml & \
wait)
