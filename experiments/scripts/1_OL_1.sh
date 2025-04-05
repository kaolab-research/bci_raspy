#!/usr/bin/env bash
conda activate eegenv
cd /home/necl-eeg/sangjoon/bci_raspy_kalcop/bci_raspy-kc-10-16b
gnome-terminal -- bash -c "read -p 'Press any key to start streaming...'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"
python ./main/main2b.py exp/SJ-text-gaze --data_folder {date}_AA_OL_{counter}