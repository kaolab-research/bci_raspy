python train_kf_offset.py \
    --task_data_path /home/necl-eeg/data/raspy/2023-11-14_A2_CL_3/task.bin \
    --kf_init_path /home/necl-eeg/data/raspy/2023-11-14_A2_CL_3/init_kf.npz \
    --dt 50000 \
    --half_life 1200.0 \
    --offset_s 0.0 \
    --A_gain 0.5 \
    --kf_save_path /home/necl-eeg/data/raspy/kf_models/2023-11-11_A2_CL_3-kf-adapted.npz
