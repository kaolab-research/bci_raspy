

import os
import pathlib
import argparse
import warnings

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class MyDumper(yaml.Dumper):
    '''https://stackoverflow.com/questions/25108581/python-yaml-dump-bad-indentation'''
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

# potential to-do: change gaze IP address.

'''
Generates:
1. .yaml files corresponding to experiments
2. .sh files to be run during experiments

Requires:
1. Subject ID
2. 
'''

'''
Run from bci_raspy directory please.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_id', help='ID of subject')
    parser.add_argument('--raspy_dir', default='.', help='path of raspy directory')
    parser.add_argument('--scripts_dir', default='./experiments/scripts/', help='path of directory to deposit .sh scripts')
    parser.add_argument('--stream_filter_dir', default='/home/necl-eeg/bci_ant_streaming/stream_filter', help='path of directory where stream_filter is')
    parser.add_argument('--ol_fold', type=int, default=-1, help='fold (integer) of kfold to use for OL model')
    parser.add_argument('--decorr_fold', type=int, default=-1, help='fold (integer) of kfold to use for decorr model')
    parser.add_argument('--kf_seed', type=str, default='', help='name of session to seed kf parameters from on day n')
    parser.add_argument('--ip', type=str, default='10.0.1.23', help='ip address to receive gaze data')
    parser.add_argument('--target_diameter', type=float, default=0.4, help='diameter of targets for center out and pinball')
    parser.add_argument('--include_pinball_pizza_kf', action='store_true', help='whether to include pinball and pizza for auto kf init')
    parser.add_argument('--ol_model_name', type=str, default='', help='name of model trained from open loop. default is OL_sid')
    parser.add_argument('--cl_model_name', type=str, default='', help='name of model trained from decorr. default is CL_sid')
    #parser.add_argument('models_dir', default='./models', help='path of models directory')
    args = parser.parse_args()

    subject_id = args.subject_id
    raspy_dir = args.raspy_dir
    scripts_dir = args.scripts_dir
    stream_filter_dir = args.stream_filter_dir
    IP = args.ip
    target_diameter = args.target_diameter
    if args.ol_fold == -1:
        warnings.warn('ol_fold not provided!')
    if args.decorr_fold == -1:
        warnings.warn('decorr_fold not provided!')
    if args.kf_seed == '':
        warnings.warn('kf_seed session not provided!')
        args.kf_seed = 'REPLACE_WITH_NAME_OF_PREVIOUS_SESSION'
    ol_fold = args.ol_fold
    decorr_fold = args.decorr_fold
    kf_seed_session = args.kf_seed
    if args.ol_model_name == '':
        ol_model_name = f'OL_{subject_id}'
    else:
        ol_model_name = args.ol_model_name
    if args.cl_model_name == '':
        cl_model_name = f'CL_{subject_id}'
    else:
        cl_model_name = args.cl_model_name
    

    stream_program_path_str = os.path.abspath(pathlib.Path(stream_filter_dir) / 'stream_data.out')

    models_dir = pathlib.Path(raspy_dir) / 'models'

    os.makedirs(pathlib.Path(models_dir) / 'exp', exist_ok=True)

    ### begin OL ###
    OL_yaml_path = pathlib.Path(models_dir) / 'templates' / 'SJ-text-gaze.yaml'
    with open(OL_yaml_path, 'r') as yaml_file:
        OL_yaml = yaml.load(yaml_file, Loader=Loader)
    OL_yaml_dest_path = pathlib.Path(models_dir) / 'exp' / 'SJ-text-gaze.yaml'
    OL_yaml['modules']['SJ_text_classification']['params']['yamlName'] = os.path.relpath(OL_yaml_dest_path, raspy_dir) # ***
    OL_yaml['modules']['recv_gaze_buffer']['params']['IP'] = IP
    with open(OL_yaml_dest_path, 'w') as f:
        yaml.dump(OL_yaml, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)

    

    # youy must run the generated script from the appropriate anaconda terminal
    OL_script_contents = '#!/usr/bin/env bash\n'
    OL_script_contents += 'conda activate eegenv\n'
    OL_script_contents += f'cd {os.path.abspath(raspy_dir)}\n'
    OL_script_contents += 'gnome-terminal -- bash -c "read -p \'Press any key to start streaming...\'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"\n'
    OL_script_contents += 'python ./main/main2b.py exp/SJ-text-gaze --data_folder {date}_' + f'{subject_id}_OL_' + '{counter}'

    OL_script_path = pathlib.Path(scripts_dir) / '1_OL_1.sh'
    with open(OL_script_path, 'w') as f:
        f.write(OL_script_contents)
    ### end OL   ###

    ### begin decorr ###
    decorr_yaml_path = pathlib.Path(models_dir) / 'templates' / 'kf-4-directions-1D-gaze.yaml'
    with open(decorr_yaml_path, 'r') as yaml_file:
        decorr_yaml = yaml.load(yaml_file, Loader=Loader)
    decorr_yaml_dest_path = pathlib.Path(models_dir) / 'exp' / 'kf-4-directions-1D-gaze.yaml'
    decorr_yaml['modules']['SJ_4_directions']['params']['yamlName'] = os.path.relpath(decorr_yaml_dest_path, raspy_dir) # ***
    decorr_yaml['modules']['decoder_hidden']['params']['path'] = f'/home/necl-eeg/data/raspy/trained_models/{ol_model_name}' # ***
    decorr_yaml['modules']['decoder_hidden']['params']['fold'] = ol_fold # FILL THIS IN # ***
    decorr_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/trained_models/{ol_model_name}/{ol_fold}_kf.npz' # CHANGE THE FOLD # ***
    decorr_yaml['modules']['recv_gaze_buffer']['params']['IP'] = IP
    with open(decorr_yaml_dest_path, 'w') as f:
        yaml.dump(decorr_yaml, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
    
    # you must run the generated script from the appropriate anaconda terminal
    decorr_script_contents = '#!/usr/bin/env bash\n'
    decorr_script_contents += 'conda activate eegenv\n'
    decorr_script_contents += f'cd {os.path.abspath(raspy_dir)}\n'
    decorr_script_contents += 'gnome-terminal -- bash -c "read -p \'Press any key to start streaming...\'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"\n'
    decorr_script_contents += 'python ./main/main2b.py exp/kf-4-directions-1D-gaze --data_folder {date}_' + f'{subject_id}_CL_' + '{counter}'

    decorr_script_path = pathlib.Path(scripts_dir) / '1_CL_1.sh'
    with open(decorr_script_path, 'w') as f:
        f.write(decorr_script_contents)
    ### end decorr   ###
    
    ### begin centerout ###
    centerout_yaml_path = pathlib.Path(models_dir) / 'templates' / 'kf-8-directions-gaze.yaml'
    with open(centerout_yaml_path, 'r') as yaml_file:
        centerout_yaml = yaml.load(yaml_file, Loader=Loader)
    centerout_yaml_dest_path = pathlib.Path(models_dir) / 'exp' / 'kf-8-directions-gaze.yaml'
    centerout_yaml['modules']['SJ_4_directions']['params']['yamlName'] = os.path.relpath(centerout_yaml_dest_path, raspy_dir) # ***
    centerout_yaml['modules']['decoder_hidden']['params']['path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}' # ***
    centerout_yaml['modules']['decoder_hidden']['params']['fold'] = decorr_fold # FILL THIS IN # ***
    centerout_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}/{decorr_fold}_kf.npz' # CHANGE THE FOLD # ***
    
    centerout_yaml['modules']['kf_clda']['params']['init_EBS_seconds'] = 180.0
    centerout_yaml['modules']['recv_gaze_buffer']['params']['IP'] = IP
    with open(centerout_yaml_dest_path, 'w') as f:
        yaml.dump(centerout_yaml, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
    
    # you must run the generated script from the appropriate anaconda terminal
    centerout_script_contents = '#!/usr/bin/env bash\n'
    centerout_script_contents += 'conda activate eegenv\n'
    centerout_script_contents += f'cd {os.path.abspath(raspy_dir)}\n'
    centerout_script_contents += 'gnome-terminal -- bash -c "read -p \'Press any key to start streaming...\'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"\n'
    centerout_script_contents += 'python ./main/main2b.py exp/kf-8-directions-gaze --data_folder {date}_' + f'{subject_id}_CL_' + '{counter}'

    centerout_script_path = pathlib.Path(scripts_dir) / '1_CL_2.sh'
    with open(centerout_script_path, 'w') as f:
        f.write(centerout_script_contents)
    ### end centerout ###

    ##### END DAY 1 #####
    
    ##### BEGIN DAY N #####
    
    # day N, first session (with 1 eval block)

    centerout_yaml_path = pathlib.Path(models_dir) / 'templates' / 'kf-8-directions-gaze-2.yaml'
    with open(centerout_yaml_path, 'r') as yaml_file:
        centerout_yaml = yaml.load(yaml_file, Loader=Loader)
    centerout_yaml_dest_path = pathlib.Path(models_dir) / 'exp' / 'kf-8-directions-gaze-2.yaml'
    centerout_yaml['modules']['SJ_4_directions']['params']['yamlName'] = os.path.relpath(centerout_yaml_dest_path, raspy_dir) # ***
    centerout_yaml['modules']['SJ_4_directions']['params']['numInitialEvaluationBlocks'] = 1 # ***
    targetsInfo = centerout_yaml['modules']['SJ_4_directions']['params']['targetsInfo'] # dictionary of [[centerx, centery], [xwidth, yheight]] or [[centerx, centery], [diameter, ignored]]
    centerout_yaml['modules']['SJ_4_directions']['params']['targetsInfo'] = {target: [info[0], [target_diameter, target_diameter]] for target, info in targetsInfo.items()}
    centerout_yaml['modules']['SJ_4_directions']['params']['defaultTargetSize'] = [target_diameter, target_diameter]
    centerout_yaml['modules']['decoder_hidden']['params']['path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}' # ***
    centerout_yaml['modules']['decoder_hidden']['params']['fold'] = decorr_fold # FILL THIS IN # ***
    #centerout_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}/{decorr_fold}_kf.npz' # CHANGE THE FOLD # ***
    centerout_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/{kf_seed_session}/final_kf.npz'
    centerout_yaml['modules']['kf_clda']['params']['init_EBS_seconds'] = 'prev'
    centerout_yaml['modules']['recv_gaze_buffer']['params']['IP'] = IP
    with open(centerout_yaml_dest_path, 'w') as f:
        yaml.dump(centerout_yaml, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
    
    # you must run the generated script from the appropriate anaconda terminal
    centerout_script_contents = '#!/usr/bin/env bash\n'
    centerout_script_contents += 'conda activate eegenv\n'
    centerout_script_contents += f'cd {os.path.abspath(raspy_dir)}\n'

    #centerout_script_contents += f'python experiments/set_kf_init.py {kf_seed_session}\n'

    centerout_script_contents += 'gnome-terminal -- bash -c "read -p \'Press any key to start streaming...\'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"\n'
    centerout_script_contents += 'python ./main/main2b.py exp/kf-8-directions-gaze-2 --data_folder {date}_' + f'{subject_id}_CL_' + '{counter}'

    centerout_script_path = pathlib.Path(scripts_dir) / 'n_CL_1.sh'
    with open(centerout_script_path, 'w') as f:
        f.write(centerout_script_contents)
    
    # day N, second session and beyond (no initial eval blocks)

    centerout_yaml_path = pathlib.Path(models_dir) / 'templates' / 'kf-8-directions-gaze-2.yaml'
    with open(centerout_yaml_path, 'r') as yaml_file:
        centerout_yaml = yaml.load(yaml_file, Loader=Loader)
    centerout_yaml_dest_path = pathlib.Path(models_dir) / 'exp' / 'kf-8-directions-gaze-2-eval0.yaml'
    centerout_yaml['modules']['SJ_4_directions']['params']['yamlName'] = os.path.relpath(centerout_yaml_dest_path, raspy_dir) # ***
    centerout_yaml['modules']['SJ_4_directions']['params']['numInitialEvaluationBlocks'] = 0 # ***
    targetsInfo = centerout_yaml['modules']['SJ_4_directions']['params']['targetsInfo'] # dictionary of [[centerx, centery], [xwidth, yheight]] or [[centerx, centery], [diameter, ignored]]
    centerout_yaml['modules']['SJ_4_directions']['params']['targetsInfo'] = {target: [info[0], [target_diameter, target_diameter]] for target, info in targetsInfo.items()}
    centerout_yaml['modules']['SJ_4_directions']['params']['defaultTargetSize'] = [target_diameter, target_diameter]
    centerout_yaml['modules']['decoder_hidden']['params']['path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}' # ***
    centerout_yaml['modules']['decoder_hidden']['params']['fold'] = decorr_fold # FILL THIS IN # ***
    #centerout_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}/{decorr_fold}_kf.npz' # CHANGE THE FOLD # ***
    centerout_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/{kf_seed_session}/final_kf.npz'
    centerout_yaml['modules']['kf_clda']['params']['init_EBS_seconds'] = 'prev'
    centerout_yaml['modules']['recv_gaze_buffer']['params']['IP'] = IP
    with open(centerout_yaml_dest_path, 'w') as f:
        yaml.dump(centerout_yaml, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
    
    # you must run the generated script from the appropriate anaconda terminal
    centerout_script_contents = '#!/usr/bin/env bash\n'
    centerout_script_contents += 'conda activate eegenv\n'
    centerout_script_contents += f'cd {os.path.abspath(raspy_dir)}\n'

    #centerout_script_contents += f'python experiments/set_kf_init.py {kf_seed_session}\n'

    centerout_script_contents += 'gnome-terminal -- bash -c "read -p \'Press any key to start streaming...\'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"\n'
    centerout_script_contents += 'python ./main/main2b.py exp/kf-8-directions-gaze-2-eval0 --data_folder {date}_' + f'{subject_id}_CL_' + '{counter}'

    centerout_script_path = pathlib.Path(scripts_dir) / 'n_CL_2.sh'
    with open(centerout_script_path, 'w') as f:
        f.write(centerout_script_contents)
    
    ############## pinball ###############
    pinball_yaml_path = pathlib.Path(models_dir) / 'templates' / 'kf-pinball-gaze.yaml'
    with open(pinball_yaml_path, 'r') as yaml_file:
        pinball_yaml = yaml.load(yaml_file, Loader=Loader)
    pinball_yaml_dest_path = pathlib.Path(models_dir) / 'exp' / 'kf-pinball-gaze.yaml'
    pinball_yaml['modules']['SJ_4_directions']['params']['yamlName'] = os.path.relpath(pinball_yaml_dest_path, raspy_dir) # ***
    pinball_yaml['modules']['SJ_4_directions']['params']['numInitialEvaluationBlocks'] = 1 # ***
    targetsInfo = pinball_yaml['modules']['SJ_4_directions']['params']['targetsInfo'] # dictionary of [[centerx, centery], [xwidth, yheight]] or [[centerx, centery], [diameter, ignored]]
    pinball_yaml['modules']['SJ_4_directions']['params']['targetsInfo'] = {target: [info[0], [target_diameter, target_diameter]] for target, info in targetsInfo.items()}
    pinball_yaml['modules']['SJ_4_directions']['params']['defaultTargetSize'] = [target_diameter, target_diameter]
    pinball_yaml['modules']['decoder_hidden']['params']['path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}' # ***
    pinball_yaml['modules']['decoder_hidden']['params']['fold'] = decorr_fold # FILL THIS IN # ***
    #pinball_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}/{decorr_fold}_kf.npz' # CHANGE THE FOLD # ***
    pinball_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/{kf_seed_session}/final_kf.npz'
    pinball_yaml['modules']['kf_clda']['params']['init_EBS_seconds'] = 'prev'
    pinball_yaml['modules']['recv_gaze_buffer']['params']['IP'] = IP
    with open(pinball_yaml_dest_path, 'w') as f:
        yaml.dump(pinball_yaml, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
    
    # you must run the generated script from the appropriate anaconda terminal
    pinball_script_contents = '#!/usr/bin/env bash\n'
    pinball_script_contents += 'conda activate eegenv\n'
    pinball_script_contents += f'cd {os.path.abspath(raspy_dir)}\n'

    #pinball_script_contents += f'python experiments/set_kf_init.py {kf_seed_session}\n'

    pinball_script_contents += 'gnome-terminal -- bash -c "read -p \'Press any key to start streaming...\'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"\n'
    pinball_script_contents += 'python ./main/main2b.py exp/kf-pinball-gaze --data_folder {date}_' + f'{subject_id}_CL_' + '{counter}'

    pinball_script_path = pathlib.Path(scripts_dir) / 'n_CL_pinball.sh'
    with open(pinball_script_path, 'w') as f:
        f.write(pinball_script_contents)
    
    ################### pizza #################
    pizza_yaml_path = pathlib.Path(models_dir) / 'templates' / 'kf-pizza-gaze.yaml'
    with open(pizza_yaml_path, 'r') as yaml_file:
        pizza_yaml = yaml.load(yaml_file, Loader=Loader)
    pizza_yaml_dest_path = pathlib.Path(models_dir) / 'exp' / 'kf-pizza-gaze.yaml'
    pizza_yaml['modules']['SJ_4_directions']['params']['yamlName'] = os.path.relpath(pizza_yaml_dest_path, raspy_dir) # ***
    pizza_yaml['modules']['SJ_4_directions']['params']['numInitialEvaluationBlocks'] = 1 # ***
    pizza_yaml['modules']['decoder_hidden']['params']['path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}' # ***
    pizza_yaml['modules']['decoder_hidden']['params']['fold'] = decorr_fold # FILL THIS IN # ***
    #pizza_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/trained_models/{cl_model_name}/{decorr_fold}_kf.npz' # CHANGE THE FOLD # ***
    pizza_yaml['modules']['kf_clda']['params']['kf_init_path'] = f'/home/necl-eeg/data/raspy/{kf_seed_session}/final_kf.npz'
    pizza_yaml['modules']['kf_clda']['params']['init_EBS_seconds'] = 'prev'
    pizza_yaml['modules']['recv_gaze_buffer']['params']['IP'] = IP
    with open(pizza_yaml_dest_path, 'w') as f:
        yaml.dump(pizza_yaml, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
    
    # you must run the generated script from the appropriate anaconda terminal
    pizza_script_contents = '#!/usr/bin/env bash\n'
    pizza_script_contents += 'conda activate eegenv\n'
    pizza_script_contents += f'cd {os.path.abspath(raspy_dir)}\n'

    #pizza_script_contents += f'python experiments/set_kf_init.py {kf_seed_session}\n'

    pizza_script_contents += 'gnome-terminal -- bash -c "read -p \'Press any key to start streaming...\'; /home/necl-eeg/bci_ant_streaming/stream_filter/stream_data.out 127.0.0.1 7779"\n'
    pizza_script_contents += 'python ./main/main2b.py exp/kf-pizza-gaze --data_folder {date}_' + f'{subject_id}_CL_' + '{counter}'

    pizza_script_path = pathlib.Path(scripts_dir) / 'n_CL_pizza.sh'
    with open(pizza_script_path, 'w') as f:
        f.write(pizza_script_contents)
    
    exclude_pinball_pizza = (not args.include_pinball_pizza_kf)
    exclude_pinball_pizza_text = " --exclude_pinball_pizza" if exclude_pinball_pizza else ""
    os.system(f'python experiments/set_kf_init.py {kf_seed_session} --subject_id {subject_id} --raspy_dir {raspy_dir} --data_dir /home/necl-eeg/data/raspy/{exclude_pinball_pizza_text}')
    pass