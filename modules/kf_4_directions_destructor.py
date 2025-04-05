# Auto README.txt.
import pathlib
import os
data_folder = params['data_folder']
session = os.path.basename(os.path.normpath(data_folder))
readme_path = pathlib.Path(data_folder) / 'README.txt'

# Please note that the values involving time do not distinguish between calibration (skipFirstNtrials) and normal trials!

task_name = 'NULL'
if task.usePizza or task.usePolygonTargets:
    task_name = f'Pizza {len(task.targetsInfo)}'
elif task.constrain1D:
    if len(task.renderAngles) > 1:
        task_name = '1D decorrelated'
    else:
        task_name = '1D default'
elif task.useRandomTargetPos:
    task_name = 'Pinball'
elif len(task.targetsInfo) == 4:
    task_name = '2D 4 targets'
else:
    task_name = f'2D center out {len(task.targetsInfo) - 1}'
intended_session_length = (task.sessionLength)/60 # minutes
try:
    active_time_start = task.startTime
    actual_session_length = time.time() - active_time_start
except:
    actual_session_length = -1
try:
    decoder_path = task.yaml_data['modules']['decoder_hidden']['params']['path']
    decoder_fold = task.yaml_data['modules']['decoder_hidden']['params']['fold']
except:
    decoder_path = 'Cannot find decoder path.'
    decoder_fold = 'Cannot find decoder fold'
try:
    kf_init_path = task.yaml_data['modules']['kf_clda']['params']['kf_init_path']
except:
    kf_init_path = 'Cannot find kf_init_path'
pass

try:
    trialCount = task.trialCount - task.skipFirstNtrials
except:
    trialCount = str(task.trialCount) + '***'

try:
    target_info = list(task.targetsInfo.items())[0][1] # get target size of first element in targetsInfo. May be inaccurate if target sizes are different.
except:
    target_info = 'Cannot find target size'
try:
    hit_percent = task.hitRate/trialCount
except:
    hit_percent = 'Error while calculating'
try:
    trials_per_600s = trialCount/(actual_session_length - task.skipFirstNtrials*task.calibrationLength)*600
except:
    trials_per_600s = 'Error while calculating'
try:
    if 'center out' in task_name:
        distance = np.linalg.norm(target_info[0])
        target_size = target_info[1][0]
        index_of_difficulty = np.log2((distance + target_size)/target_size)
    else:
        raise ValueError('invalid task for this calculation')
except:
    index_of_difficulty = 'N/A'
try:
    out_hit_incorrect_timeout = f'{task.outHitRate}, {task.outMissRate - task.outTimeoutRate}, {task.outTimeoutRate}'
except:
    out_hit_incorrect_timeout = 'N/A'
try:
    out_time = task.outTime
except:
    out_time = 'N/A'

readme_contents  = ''
readme_contents += f'session                          : {session}\n'
readme_contents += f'sessionLength (minutes)          : {intended_session_length}\n'
readme_contents += f'actual session time (seconds)    : {actual_session_length}\n'
readme_contents += f'actual session time (minutes)    : {actual_session_length/60}\n'
readme_contents += f'Task                             : {task_name}\n'
readme_contents += f'Target info                      : {target_info}\n'
readme_contents += f'Decoder path                     : {decoder_path}\n'
readme_contents += f'Decoder fold                     : {decoder_fold}\n'
readme_contents += f'kf_init_path                     : {kf_init_path}\n'
readme_contents += f'Number of hits                   : {task.hitRate}\n'
readme_contents += f'trialCount                       : {trialCount}\n'
readme_contents += f'hitRate/trialCount               : {hit_percent}\n'
readme_contents += f'trials/minute*(10 minutes)       : {trials_per_600s}\n'
readme_contents += f'Index of difficulty              : {index_of_difficulty}\n'
readme_contents += f'Out hit, incorrect, timeout      : {out_hit_incorrect_timeout}\n'
readme_contents += f'Out time                         : {out_time}\n'
readme_contents += f'kfCopilotAlpha (1.0: no copilot) : {task.kfCopilotAlpha}\n'
readme_contents += f'\nAdditional Notes:\n'

with open(readme_path, 'w') as f:
    f.write(readme_contents)

print('BEGIN README CONTENTS\n' + readme_contents + '\nEND README CONTENTS\n', end='')