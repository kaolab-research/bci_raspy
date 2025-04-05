
import argparse
import pathlib
import os
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

def parse_readme(path):
    try:
        with open(path, 'r') as f:
            contents = f.read()
        lines = [line.split(':') for line in contents.split('\n') if ':' in line]
        readme = {line[0].strip(): line[1].strip() for line in lines}
    except:
        return {}
    return readme

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('session', help='ID of subject')
    parser.add_argument('--exclude_pinball_pizza', action='store_true', help='exclude pinball pizza from auto session')
    parser.add_argument('--raspy_dir', default='.', help='path of raspy directory')
    parser.add_argument('--data_dir', default='/home/necl-eeg/data/raspy/', help='path of raspy directory')
    parser.add_argument('--subject_id', default='NULL', help='subject_id')
    args = parser.parse_args()

    raspy_dir = args.raspy_dir
    data_dir = args.data_dir
    session = args.session
    subject_id = args.subject_id
    exclude_pinball_pizza = args.exclude_pinball_pizza

    if session == 'auto':
        subject_sessions = [folder for folder in os.listdir(data_dir) if subject_id in folder]
        kf_sessions = [folder for folder in subject_sessions if 'final_kf.npz' in os.listdir(pathlib.Path(data_dir) / folder)]
        if exclude_pinball_pizza:
            tasks = [parse_readme(pathlib.Path(data_dir) / folder / 'README.txt').get('Task', '') for folder in kf_sessions]
            #print(list(zip(kf_sessions, tasks)))
            kf_sessions = [folder for (folder, task) in zip(kf_sessions, tasks) if 'pinball' not in task.lower() and 'pizza' not in task.lower()]
        if len(kf_sessions) == 0:
            warnings.warn(f'No final_kf.npz available for subject {subject_id}')
        session = max(kf_sessions, key=lambda sesh: os.path.getmtime(pathlib.Path(data_dir) / sesh / 'final_kf.npz'))
        print(f'using session {session} as kf seed')

    models_dir = pathlib.Path(raspy_dir) / 'models'
    for model_file in ['kf-8-directions-gaze-2.yaml', 'kf-8-directions-gaze-2-eval0.yaml',  'kf-pinball-gaze.yaml', 'kf-pizza-gaze.yaml']:
        try:
            yaml_path = models_dir / 'exp' / model_file
            with open(yaml_path, 'r') as yaml_file:
                yaml_data = yaml.load(yaml_file, Loader=Loader)
            yaml_data['modules']['kf_clda']['params']['kf_init_path'] = os.path.abspath(pathlib.Path(data_dir) / session / 'final_kf.npz')
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(e)
    pass