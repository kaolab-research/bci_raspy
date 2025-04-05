# Auto README.txt.
import pathlib
import os
data_folder = params['data_folder']
session = os.path.basename(os.path.normpath(data_folder))
readme_path = pathlib.Path(data_folder) / 'README.txt'

intended_session_length = (task.sessionLength)/60 # minutes
try:
    actual_session_length = time.time() - task.startTime
except:
    actual_session_length = -1

readme_contents  = ''
readme_contents += f'session                          : {session}\n'
readme_contents += f'sessionLength (minutes)          : {intended_session_length}\n'
readme_contents += f'actual session time (seconds)    : {actual_session_length}\n'
readme_contents += f'actual session time (minutes)    : {actual_session_length/60}\n'
readme_contents += f'Task                             : OL\n'
readme_contents += f'Decoder path                     : N/A\n'
readme_contents += f'Decoder fold                     : N/A\n'
readme_contents += f'kf_init_path                     : N/A\n'
readme_contents += f'Number of hits                   : N/A\n'
readme_contents += f'trialCount                       : {task.trialCount}\n'
readme_contents += f'kfCopilotAlpha (1.0: no copilot) : N/A\n'
actions = {action: task.action2state_task.get(action, 'None') for action in task.testedAction}
readme_contents += f'actions                          : {actions}'

readme_contents += f'\nAdditional Notes:\n'

with open(readme_path, 'w') as f:
    f.write(readme_contents)

print('BEGIN README CONTENTS')
print(readme_contents)
print('END README CONTENTS')