# Manual for generating yaml and sh files for running experimental sessions.

## Each Day
* check the IP address of EEG computer. EEG and gaze computer should both be on ncel-wifi-2.
* Calibrate eye tracking (on eye tracking computer). Desktop->eeg_calibration.bat. Press escape to quit.
* Mark center of seat by placing tape on floor.
* edit (on eye tracking computer) Desktop->eeg_stream.bat --ip info

## Impedance checking, manual streaming
* see bci_ant_streaming repository. Remember to turn on amplifier before plugging in USB cable.
1. Navigate to directory: `cd ~/bci_ant_streaming/impedance`
2. run `conda activate eegenv`.
3. Run `python plot_impeeg_astext2.py`. Ctrl+C to stop. `clear` to clear screen. 
4. If needed, `cd ~/bci_ant_streaming/stream_filter`, then run `./stream_data.out` (equivalent to `./stream_data.out 127.0.0.1 7779`).

## Day 1
Before OL:
0. run `conda activate eegenv`. Navigate to directory `cd ~/sangjoon/bci_raspy_kalcop/bci_raspy-kc-10-16b/`
1. run `python experiments/generate_sh.py sid --ip CHECK_IP_ADDRESS` # sid is subject_id
2. Run OL as `source experiments/scripts/1_OL_1.sh`. Move and maximize pygame window. Start gaze stream on gaze laptop. Start EEG stream on 2nd terminal. Describe the task: "Four actions, labeled 'Left Hand', 'Right Hand', 'Both Hands', and 'Feet' (SCI: 'Left Leg', 'Right Leg', 'Both Legs', 'Still': no attempted movement) will be shown in any order. When each action is prompted, move (or attempt to move) each body part shown. For example, if the prompt is 'Left Hand', you may wiggle the fingers of your left hand repetitively while keeping your wrist and arm still. If the prompt is 'Feet', you may repetitively wiggle/curl your toes with your feet in place. In between each prompt there will be a red square lasting two seconds. You do not need to perform any action when the red square is shown.". Instruct the participant to perform the action (repetitively) for the entire duration that it is present. Inform the participant that they may blink whenever they need to.
After OL: 
1. train and save model, then COPY and rename copied directory as OL_{sid}. You may have to restart the computer after training and delete incomplete sessions if there are many timing violations after training. MAKE A NOTE OF WHICH FOLD IS USED. Optionally, delete all other fold files from `/home/necl-eeg/data/raspy/trained_models/OL_{sid}`
Edit config.yaml `(~/sangjoon/bci_raspy_kalcop/Offline_EEGNet-kalman/config.yaml)` -> data_names, then:
```
conda activate eegenv
cd ~/sangjoon/bci_raspy_kalcop/Offline_EEGNet-kalman
python pipeline_kf_func.py config.yaml
```
2. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --ip CHECK_IP_ADDRESS`.
3. Run Decorrelated task as `source experiments/scripts/1_CL_1.sh`. Describe the task to the participant: "Each trial, lasting for , a moving cursor (white circle) will be shown on screen. In between every trial, the red square indicating no action will be shown and the cursor will be reset to the center position. For each trial, you should perform only the action shown, and you should perform the prompted action for the entire duration of the trial, both when the cursor is red and when the cursor is white. You should continue performing the action even if the cursor reaches the target. The cursor will move between the center position and the action prompt, and the cursor's movement will reflect the decoding of your EEG activity. The action prompt will show up in any of 8 locations on the screen, spaced around a circle, and its position will be random and generally will not correspond to the action shown. For example, the 'Left Hand' prompt may appear to the right of the center position." Instruct the participant to perform (repetitively) the action for the entire duration that it is present.
After Decorr:
4. train and save model (keep data_kinds as `-OL`), then COPY and rename copied directory as CL_{sid}. If the smallest diagonal of the confusion matrix is less than 60%, then run another decorrelated session (using either decoder), concatenate the new decorrelated session to the config, and retrain. You may have to restart the computer after training and delete incomplete sessions if there are many timing violations after training. MAKE A NOTE OF WHICH FOLD IS USED. Optionally, delete all other fold files from `/home/necl-eeg/data/raspy/trained_models/CL_{sid}`
5. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --ip CHECK_IP_ADDRESS`
6. Run center-out task as `source experiments/scripts/1_CL_2.sh`. The diameter is 0.6 units (0.7 units distance from center). The session will last AT LEAST 10 minutes, but will most likely go over by several minutes. The countdown will reach 0, but the task will not end until the current block is complete (up to 16 TrialCount). Describe the task: "In the center-out task, The cursor will appear as a grey ball, and your task will be to navigate the cursor to the targets, which will initially appear as green balls. During the first 4 trials, the cursor will not move from the center position and the targets at the Left, Right, Top, and Bottom positions will be shown. During this time, you should perform the actions for Left: Left Hand, Right: Right Hand, Up: Both Hands, and Down: Feet while the targets are shown. After this calibration period, targets will appear in one of 8 radial positions, with the target appearing in the center position in between. For example, you may see targets ordered as: Left, Center, Top Left, Center, Bottom, Center, Top Right, Center, etc. You should attempt to navigate the cursor to the center of the target ball as quickly as possible by performing the corresponding actions -- Left: Left Hand, Right: Right Hand, Up: Both Hands, and Down: Feet. The cursor may not necessarily move in the direction of your performed action. When the center of the cursor lies within the target ball, the target will turn blue. You must keep the cursor within the target for 0.5 seconds before the trial is counted as a success and the next target is shown. If the allotted time expires (24 seconds), the next target will be shown. If the next target is an outer target, then the cursor position will be reset to the center. You should continue performing this task until the window closes by itself. The time shown does not reflect the end of the session. The task will last for more than 10 minutes. On following days, we may decrease the size of the target.". 
7. If time allows, run center-out-8 with target_size of 0.4: run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --ip CHECK_IP_ADDRESS`.
8. Run center-out task with target_size 0.4 as `source experiments/scripts/n_CL_1.sh`. n_CL_1 has an evalutation phase at the beginning of the session (KF adapt on, no KF sychronization). Inform participant that the target will be smaller than the first session.
## Day n (2, 3, 4):
generate_sh.py must be run every time a new session is run. DIAMETER may be 0.4 by default, and may be reduced to 0.30 if >= 95% hitRate/trialCount And trials/minute*(10 minutes) >= 65. When the target size is reduced, tell the participant that the size of the target is smaller than during previous sessions.

0. run `conda activate eegenv`
1. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`
2. Run CL_1 as `source experiments/scripts/n_CL_1.sh`. n_CL_1 has an evalutation phase at the beginning of the session (KF adapt on, no KF sychronization).
3. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`
4. Run CL_2 as `source experiments/scripts/n_CL_2.sh`. n_CL_2 does not have an evaluation phase at the beginning of the session
5. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`
6. Run CL_3 as `source experiments/scripts/n_CL_2.sh`
7. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`
8. Run CL_4 as `source experiments/scripts/n_CL_2.sh`

## Day 5: Pinball & Pizza days
0. run `conda activate eegenv`
1. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`
2. Run CL_1 as `source experiments/scripts/n_CL_1.sh`
3. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`
4. Run Pinball as `source experiments/scripts/n_CL_pinball.sh`. Describe the task: "In the pinball task, the cursor and target will have the same ball appearance as the center-out task. The first 4 trials will have the same calibration period. However, rather than having the target appear at one of a number of set locations, the target will instead appear at random locations within the bounding box. Your task will be the same: navigate the cursor to the center of each target as quickly as possible and hold for 0.5 seconds to achieve a successful trial.".
5. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`
6. Run Pizza1 as `source experiments/scripts/n_CL_pizza.sh`. Describe the task: "In the pizza task, the cursor will appear as a white circle, with targets appearing as trapezoids spaced around an octogon. The target which you should navigate the cursor to will be white, while all other targets will be grey. The first target that your cursor touches will be selected, and the number of correct and incorrect selections will be recorded. You should attempt to reach the white target as quickly as possible while avoiding grey targets.".
7. run `python experiments/generate_sh.py sid --ol_fold whatever_fold_number --decorr_fold whatever_fold_number --kf_seed auto --ip CHECK_IP_ADDRESS --target_diameter DIAMETER`. This should inherit kf_seed from the last center-out session. You can also add `--include_pinball_pizza` to instead inherit from the last of any closed-loop KF session; however, we should not use this here.
8. Run Pizza2 as `source experiments/scripts/n_CL_pizza.sh`

#### to-do:


```
usage: generate_sh.py [-h] [--raspy_dir RASPY_DIR] [--scripts_dir SCRIPTS_DIR] [--stream_filter_dir STREAM_FILTER_DIR] [--ol_fold OL_FOLD]
                      [--decorr_fold DECORR_FOLD]
                      subject_id

positional arguments:
  subject_id            ID of subject

options:
  -h, --help            show this help message and exit
  --raspy_dir RASPY_DIR
                        path of raspy directory
  --scripts_dir SCRIPTS_DIR
                        path of directory to deposit .sh scripts
  --stream_filter_dir STREAM_FILTER_DIR
                        path of directory to deposit .sh scripts
  --ol_fold OL_FOLD     fold of kfold to use for OL model
  --decorr_fold DECORR_FOLD
                        fold of kfold to use for OL model
```
## Copilot
1. inside n_CL_2 yaml, manually uncomment copilot_path
2. set kfCopilotAlpha to 0

