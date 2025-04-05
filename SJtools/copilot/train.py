from SJtools.copilot.env import SJ4DirectionsEnv
from SJtools.copilot.policy.BNPolicy import BNPolicy 
from SJtools.copilot.policy.CNNPolicy import CNNPolicy 
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from SJtools.copilot.callbacks import LearningRateCallback,TensorboardLoggerCallback
from SJtools.copilot.trainUtil import fileOrganizer, getDevice
from stable_baselines3.common.monitor import Monitor


import os
import shutil
import sys
import numpy as np
import signal
import time
import argparse
import yaml
import torch

# hyper parameters
parser = argparse.ArgumentParser(description="Specify Copilot Argument")
parser.add_argument("-model",type=str,default="RecurrentPPO")
parser.add_argument("-timesteps",type=int,default=200000)
parser.add_argument("-n_steps",type=int,default=2048,help="number of steps taken before updating the model")
parser.add_argument("-batch_size",type=int,default=64)
parser.add_argument("-log_interval",type=int,default=4,help="number of gradient update before logging")
parser.add_argument("-wandb", default=True, action='store_true')
parser.add_argument("-no_wandb", dest='wandb', action='store_false')
parser.add_argument("-softmax_type", default='complex', type=str,help="choose between (simple, complex, two_peak,normal_target)")
parser.add_argument("-reward_type", default='exp_dist', type=str, help="choose between (lin_dist,exp_dist,lin_angle)")
parser.add_argument("-CS", default=0.7,type=float, help="percentage of correct softmax (does not apply for simple softmax)")
parser.add_argument("-stillCS", default=0.7, type=float, help="percentage of correct softmax for still state specifically (does not apply for simple softmax)")
parser.add_argument("-renderEval", default=False, action='store_true')
parser.add_argument("-renderTrain", default=False, action='store_true')
parser.add_argument("-showSoftmax", default=False, action='store_true')
parser.add_argument("-showVelocity", default=False, action='store_true')
parser.add_argument("-showHeatmap", default=False, action='store_true')
parser.add_argument("-showMass", default=False, action='store_true')
parser.add_argument("-alpha", default=None,help="float between -1 to 1. you can fix action[2] which is alpha to be a fixed number. action[2] controls convex combination of decoder + copilot")
parser.add_argument("-target_predictor", type=str, default=None, help='type "truth" for perfect target predictor')
parser.add_argument("-target_predictor_input", default="softmax", help="choose between (softmax,softmax_pos)")
parser.add_argument("-cursor_target_obs", default=False, action='store_true', help="if you only want 4 dim which is [cursor(2),target(2)], then set this flag. setting this flag must mean also using target_predictor")
parser.add_argument("-obs",default=[], nargs='+',help="can have several items: hold, vel, acc, time, targetTime, targetCenter, mass, trialEnd these information is always added in alphabetical order")
parser.add_argument("-no_holdtime", default=False, action='store_true', help="if you want to have no hold time (hold_time_thres = 0.02 instead of 50ms), then set this flag")
parser.add_argument("-holdtime", default=0.5, type=float, help="uses this original holdtime of 500ms unless you change it")

parser.add_argument("-lr_scheduler", type=str, default="constant", help="constant / reducelronplateau / linear / exp")
parser.add_argument("-lr", default=0.0003, type=float, help="base learning rate")
parser.add_argument("-lr_min", default=3*(10**-10), type=float, help="min learning rate")
parser.add_argument("-lr_patience", default=10, type=int, help="value only used for ReduceLROnPlateau")
parser.add_argument("-curriculum_yaml",default=None, type=str, help='feed in curriculum learning yaml with specific structure. example is provided in curriculum, default is None.')
parser.add_argument("-extra_targets",default={}, nargs='+',help='xy coordinate of new target (multipe of 2). i.e) 0.0 0.0 0.0 0.4')
parser.add_argument("-extra_targets_yaml",default=None, type=str, help='feed in new targets only by yaml file name i.e) grid-16.yaml. look inside SJtools/copilot/targets/')
parser.add_argument("-center_out_back", default=False, action='store_true', help="creates a game such that it is center out back")
parser.add_argument("-history",default=[1,0], nargs='+', help="[n, interval] n history with t interval ex) -history 3 100 looks at 3 history (including current) with 100 timestep interval")
parser.add_argument('-historyReset',type=str,default='zero',help='how to reset history (zero,last): zero means when history is reset all position is set to zero. last means all history uses last pos when reset')
parser.add_argument("-maskSoftmax", default=None, type=str, help='all or partial or None. all means removing all softmax history')
parser.add_argument("-binaryAlpha",default=False, action='store_true', help="alpha is treated as either full control of cursor or no control")
parser.add_argument("-obs_heatmap",default=None,type=int,help="put in one number (n) to create nxn heatmap to be fed as observation")
parser.add_argument("-obs_heatmap_option",default=[], nargs='+',help="should only be set up of args.obs_heatmap=int. may add other obs option such as (com = center of mass of heatmap (x,y), bcc= bincount of cursor (counter where cursor is located), only = no heatmap used in obs, df0.9 = decrement factor how much to decrement obs_heatmap) i.e [cf0.9, comonly]")
parser.add_argument("-action",default=["vx","vy","alpha"], nargs='+',help="can have several items: vx, vy, fx, fy, alpha, vm, click. these information is added in the above order where fx,fy is force applied. vm is velocity magnitude")
parser.add_argument("-action_param",default=[], nargs='+',help="action parameters. dictionary format. key1,value1,key1,value1")
parser.add_argument("-policy", type=str, default=None, help='default (None) is mlppolicy, but can pass in any network name you find from policy folder in copilot')
parser.add_argument("-policy_param_s", nargs='+', default=[], help='policy kwargs. determines depth and number of filters')
parser.add_argument("-policy_param_p", nargs='+', default=[64,64], help='policy kwargs. determines depth and number of filters for policy')
parser.add_argument("-policy_param_v", nargs='+', default=[64,64], help='policy kwargs. determines depth and number of filters for value')
parser.add_argument("-device",default='cpu', help='default will choose the best one, but can specify (cpu,cuda,mps // cpu is fastest on macbook)')
parser.add_argument('-save',default=True, action='store_true', help="saves model logs etc; by default it saves everything, unless no_save flag is on")
parser.add_argument('-no_save', dest='save', action='store_false', help="set this flag if you don't want to log/save anything while training; used for developing purpose")
parser.add_argument('-dev', dest='save', action='store_false', help="set this flag if you don't want to log/save anything while training; used for developing purpose; same as no_save")
parser.add_argument('-devSave', default=False, action='store_true', help="same as dev except saves temporary model in folder dev")
parser.add_argument('-velReplaceSoftmax',default=False,action='store_true',help='set this flag to have obs include velocity created from softmax instead of softmax itself')
parser.add_argument('-cardinalVelocityThres',default=False,help='set this flag to float if you wish to use cardinal velocity. [0.0~1.0] anything above this value will be treated as argmax. i.e)0.5 [0.1, 0.6, 0.2, 0, 0] ==> [0,1,0,0,0]')
parser.add_argument('-noSoftmax',default=False,action='store_true',help='set this flag to have obs exclude softmax or velocity or any information from the decoder')
parser.add_argument('-tools',nargs='+', default=[],help='set this flag to use various tools provided by raspy i.e) seeTrajectory')
parser.add_argument('-fileName',type=str,default='',help="setting fileName to something other than empty string will save it with that name")
parser.add_argument('-filePath',type=str,default='',help="this is the directory in which model will be stored. empty will automatically mean SJtools/copilot/runs/. note: it starts from bci_raspy folder")
parser.add_argument('-trial_per_episode',type=int,default=1,help="number of trials until episode outputs done. normally this is 1, but worth experimenting to help LSTM learn dynamic")



# command in string (print it one more time for logger)
fullCommand = "python -m SJtools.copilot.train "+" ".join(sys.argv[1:])

# preprocess some arguments
args = parser.parse_args()

# if args.softmax_type == 'simple' and (len(args.extra_targets) > 0 or args.extra_targets_yaml is not None):
#     print("ERROR: simple softmax cannot handle extra_targets. please use complex or two_peak if you choose to use extra targets")
#     exit(1)

# history first two are integers rest are solid arguments
try: args.history[0] = int(args.history[0]); args.history[1] = int(args.history[1])
except: print("ERROR: args.history must be 2 value list"); exit(1)

# determine number of eval episode
n_eval_episodes = 5
if len(args.extra_targets) > 0:
    if len(args.extra_targets) % 2 != 0: 
        print("ERROR: invalid extra_targets, must be multiple of two = (x,y)")
        exit(1)
    _extra_targets = {}
    for i in range(0,len(args.extra_targets),2):
        target = [float(args.extra_targets[i]),float(args.extra_targets[i+1])]
        targetSize = np.array([0.2,0.2])
        str_target = str(target)
        if np.linalg.norm(target) > 1:
            print(f"ERROR: invalid extra_target {target}, must be within +/-1 range")
            exit(1)
        _extra_targets[str_target] = [np.array(target),np.array(targetSize)]
    args.extra_targets = _extra_targets

    n_eval_episodes += len(_extra_targets) # should be 5 but you add extra ones
if args.extra_targets_yaml is not None:
    yamlpath = f"SJtools/copilot/targets/{args.extra_targets_yaml}"
    with open(yamlpath) as yaml_file:
        yaml_data = yaml.load(yaml_file,Loader=yaml.Loader)
    n_eval_episodes = len(yaml_data["targetsInfo"]) * (args.center_out_back + 1)
        
    # store target info


""" take care of dependent variable from argument """
wandbUsed = args.wandb and args.save
eval_freq = args.log_interval * args.n_steps
n_update_per_log = args.log_interval
lrSchedulerInfo = {
    "lr_scheduler":args.lr_scheduler,
    "lr":args.lr,
    "lr_min":args.lr_min,
    "lr_patience":args.lr_patience,
}
learing_rate = args.lr

### FYI: not used yet but batch size is n_steps * n_env where n_env is number of environment copies running in parallel
if args.model == "RecurrentPPO": 
    policy = "MlpLstmPolicy"
elif args.model == "PPO": 
    if args.policy == None: policy = "MlpPolicy"
    elif args.policy == "BNPolicy": policy = BNPolicy
    elif args.policy == "CNNPolicy": policy = CNNPolicy
    else: print("Invalid Policy Name"); exit(1)

net_arch = [int(x) for x in args.policy_param_s] + [{
        'pi':[int(x) for x in args.policy_param_p],
        'vf':[int(x) for x in args.policy_param_v]
    }]

if args.alpha is not None: args.alpha = float(args.alpha)
if args.no_holdtime: args.holdtime = 0.02

""" wandb init and wandb exit"""
if wandbUsed:     
    import wandb
    from wandb.integration.sb3 import WandbCallback

    run = wandb.init(
        project="SJ-4-directions-copilot", 
        entity="aaccjjt",
        config={"env_name": "SJ-4-directions",
                "policy_type":policy,
                "net_arch":net_arch,
                },
        sync_tensorboard=True,
        # monitor_gym=saveVideo,
        )
    wandb.config.update(args)
    wandb.config.update({"command":fullCommand})
    wandb.save(f"SJtools/copilot/env.py") # want to see store reward shaping details

    # for save when exiting
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        wandb.config.update({"Ctrl+C": True})
        wandb.save(myFiles.bestModelPath)
        
        # finish and clean up
        run.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    run_id = run.id
    wandb.config.update({"run_id": run_id})
else:
    run_id = ''


""" create directory for logging using fileOrganizer """
myFiles = fileOrganizer(wandbUsed, run_id, no_save=not args.save,fileName=args.fileName,currPath=args.filePath, devSave=args.devSave)

""" write full command one more time """
print('\033[91m' + '\033[1m' + fullCommand + '\033[0m')
print(f'\033[1mrun id: {myFiles.runId}\033[0m\n')
print("runid:"+myFiles.runId, file=sys.stderr) # printing runid to stderr (needed in this format exactly [look run.py])
myFiles.log(fullCommand)
    

""" ============== CREATE RL ENV ============== """

env = SJ4DirectionsEnv(wandbUsed=wandbUsed,render=args.renderTrain,showSoftmax=args.showSoftmax,showVelocity=args.showVelocity,showHeatmap=args.showHeatmap,softmax_type=args.softmax_type,reward_type=args.reward_type,setAlpha=args.alpha,CSvalue=args.CS,stillCS=args.stillCS,useTargetPredictor=args.target_predictor,target_predictor_input=args.target_predictor_input,cursor_target_obs=args.cursor_target_obs,holdtime=args.holdtime,curriculum_yaml=args.curriculum_yaml,extra_targets=args.extra_targets,extra_targets_yaml=args.extra_targets_yaml,obs=args.obs,action=args.action,historyDim=args.history,historyReset=args.historyReset,maskSoftmax=args.maskSoftmax,binaryAlpha=args.binaryAlpha,obs_heatmap=args.obs_heatmap,obs_heatmap_option=args.obs_heatmap_option,center_out_back=args.center_out_back,velReplaceSoftmax=args.velReplaceSoftmax,cardinalVelocityThres=args.cardinalVelocityThres,noSoftmax=args.noSoftmax,tools=args.tools,trial_per_episode=args.trial_per_episode,hideMass=not args.showMass,action_param=args.action_param)
eval_env = SJ4DirectionsEnv(isEval=True,wandbUsed=wandbUsed,render=args.renderEval,showSoftmax=args.showSoftmax,showVelocity=args.showVelocity,showHeatmap=args.showHeatmap,softmax_type=args.softmax_type,reward_type=args.reward_type,setAlpha=args.alpha,CSvalue=args.CS,stillCS=args.stillCS,target_predictor_input=args.target_predictor_input,useTargetPredictor=args.target_predictor,cursor_target_obs=args.cursor_target_obs,holdtime=args.holdtime,curriculum_yaml=args.curriculum_yaml,extra_targets=args.extra_targets,extra_targets_yaml=args.extra_targets_yaml,obs=args.obs,action=args.action,historyDim=args.history,historyReset=args.historyReset,maskSoftmax=args.maskSoftmax,binaryAlpha=args.binaryAlpha,obs_heatmap=args.obs_heatmap,obs_heatmap_option=args.obs_heatmap_option,center_out_back=args.center_out_back,velReplaceSoftmax=args.velReplaceSoftmax,cardinalVelocityThres=args.cardinalVelocityThres,noSoftmax=args.noSoftmax,tools=args.tools,trial_per_episode=args.trial_per_episode,hideMass=not args.showMass,action_param=args.action_param)
env = Monitor(env) # this monitor wrapping is necessary in sb3
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(eval_env, best_model_save_path=myFiles.best_model_save_path,
                            log_path=myFiles.evalCallbackPath, eval_freq=eval_freq,
                            n_eval_episodes=n_eval_episodes,
                            deterministic=True, render=False)

lr_callback = LearningRateCallback(eval_callback, n_update_per_log, lrSchedulerInfo, args.timesteps, wandbUsed)

tb_callback = TensorboardLoggerCallback(env=env,eval_env=eval_env,n_update_per_log=n_update_per_log, n_eval_episodes=n_eval_episodes)

""" ============== CREATE RL AGENT ============== """

policy_kwargs = {'net_arch':net_arch}

if args.model == "RecurrentPPO":
    model = RecurrentPPO(policy, env, n_steps=args.n_steps, learning_rate=learing_rate, batch_size=args.batch_size ,verbose=1, tensorboard_log=myFiles.tensorboard_log, device=args.device)
if args.model == "PPO":
    model = PPO(policy, env, n_steps=args.n_steps, learning_rate=learing_rate, batch_size=args.batch_size, verbose=1, tensorboard_log=myFiles.tensorboard_log, policy_kwargs=policy_kwargs, device=args.device)

""" save model yaml """
yamlcontent = {
        "copilot":{
            "model":args.model,
            "policy":args.policy,
            "net_arch":net_arch, # no need to save, but it's a good reference
        },
        "targets": {
            "extra_targets":args.extra_targets,
            "extra_targets_yaml":args.extra_targets_yaml,
        },
    }
yamlcontent.update(env.copilotYamlParam)
myFiles.saveYaml(yamlcontent)
myFiles.saveRewardYaml(env.rewardClass.fullYamlPath)
if wandbUsed:
    wandb.save(myFiles.modelYamlPath)

""" ============== TRAIN RL AGENT ============== """

if wandbUsed: callback = [eval_callback, lr_callback, tb_callback, WandbCallback()]
else: callback = [eval_callback, lr_callback, tb_callback]
model.learn(total_timesteps=args.timesteps,log_interval=args.log_interval, callback=callback)

""" saving model """
last_model = model
if args.save: last_model.save(myFiles.lastModelPath)

""" try model for 100 trials """
# best model / last model
print("FINAL EVALUATION")
best_model = None
if args.save and args.model == "RecurrentPPO":
    best_model = RecurrentPPO.load(myFiles.bestModelPath)
if args.save and args.model == "PPO":
    best_model = PPO.load(myFiles.bestModelPath)

models_text = ["last model","best model"]
models = [last_model, best_model]
data2log = {}
finalLog = "" # for run.py
if not args.save: models_text.pop(); models.pop() # best model is not saved
for model,txt in zip(models,models_text):
    for softmax_type in ["complex", "two_peak","simple","normal_target"]:
    # for softmax_type in [args.softmax_type]:

        total_trials = 100
        trial_i = 0
        env = SJ4DirectionsEnv(isEval=True,wandbUsed=wandbUsed,render=args.renderEval,showSoftmax=args.showSoftmax,showVelocity=args.showVelocity,showHeatmap=args.showHeatmap,softmax_type=softmax_type,reward_type=args.reward_type,setAlpha=args.alpha,CSvalue=args.CS,stillCS=args.stillCS,target_predictor_input=args.target_predictor_input,useTargetPredictor=args.target_predictor,cursor_target_obs=args.cursor_target_obs,extra_targets=args.extra_targets,extra_targets_yaml=args.extra_targets_yaml,obs=args.obs,action=args.action,historyDim=args.history,historyReset=args.historyReset,maskSoftmax=args.maskSoftmax,binaryAlpha=args.binaryAlpha,obs_heatmap=args.obs_heatmap,obs_heatmap_option=args.obs_heatmap_option,center_out_back=args.center_out_back,velReplaceSoftmax=args.velReplaceSoftmax,noSoftmax=args.noSoftmax,tools=args.tools,trial_per_episode=args.trial_per_episode,hideMass=not args.showMass,action_param=args.action_param)
        obs = env.reset()
        episode_start = True
        _states = None

        while trial_i < total_trials:
            action, _states = model.predict(obs, state=_states, deterministic=True, episode_start=episode_start)
            episode_start = False
            obs, reward, done, info = env.step(action)
            if done: 
                obs = env.reset()
                episode_start = True
                
                # skip still state
                if np.sum(np.isnan(info['target_pos'])) == 0:
                    trial_i += 1
        
        total_success = np.sum(np.array(env.trialResults) > 0)
        print(f"{txt} with {softmax_type} softmax: {total_success/total_trials}, {total_success}, {total_trials}")
        print(f"^ time to hit, avg extra distance travelled, avg distance travelled: {format(env.taskGame.PerformanceRecord.avgTimeToHit, '.2f')}, {format(env.taskGame.PerformanceRecord.avgExtraDistanceTravelled, '.2f')}, {format(env.taskGame.PerformanceRecord.avgDistanceTravelled, '.2f')}")
        myFiles.log(f"{txt} with {softmax_type} softmax: {total_success/total_trials}, {total_success}, {total_trials}")
        myFiles.log(f"^ time to hit, avg extra distance travelled, avg distance travelled: {format(env.taskGame.PerformanceRecord.avgTimeToHit, '.2f')}, {format(env.taskGame.PerformanceRecord.avgExtraDistanceTravelled, '.2f')}, {format(env.taskGame.PerformanceRecord.avgDistanceTravelled, '.2f')}")
        data2log[f"{txt} with {softmax_type} softmax"] = total_success

        # log to stderr 
        finalLog = finalLog + f"{txt} {softmax_type}: {total_success}\n"
        finalLog = finalLog + f"{txt} {softmax_type} time to hit: {format(env.taskGame.PerformanceRecord.avgTimeToHit, '.2f')}\n"
        finalLog = finalLog + f"{txt} {softmax_type} extra distance travelled: {format(env.taskGame.PerformanceRecord.avgExtraDistanceTravelled, '.2f')}\n"

print(finalLog, file=sys.stderr) # printing finalLog to stderr [look run.py]

if wandbUsed:
    wandb.log(data2log)
    
""" print one last time for good measure """
myFiles.log(fullCommand)
print('\033[91m' + '\033[1m' + fullCommand + '\033[0m')
print(f'\033[1mrun id: {myFiles.runId}\033[0m\n')

""" save wandb files """
if wandbUsed:
    """ wandb exit """
    wandb.save(myFiles.lastModelPath)
    wandb.save(myFiles.bestModelPath)
    
    # finish and clean up
    run.finish()

print("Done")

