
"""
some important keys in test:
s: proceeds the game when -pause flag is on (stop/start)
d: sets alpha to 0 (dead copilot)
f: makes cursor follow mouse when -followCursorAction flag is on (follow)
space: showNonCopilotCursor. shows non copilot cursor
"""


from SJtools.copilot.env import SJ4DirectionsEnv
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import time
import sys
import signal
import argparse
import numpy as np
import os.path
import yaml
import torch
from SJtools.copilot.trainUtil import isfloat, getDevice
import matplotlib.pyplot as plt
from SJtools.copilot.copilotUtils.animatedPlot import AnimatedReward
from SJtools.copilot.copilotUtils.rewards import RewardClass


# hyper parameters
parser = argparse.ArgumentParser(description="Specify Copilot Argument")
parser.add_argument("argv", metavar='N', nargs='+', help="usually needs file (+alpha)")
parser.add_argument("-alpha", default=None,help="float between -1 to 1. you can fix action[2] which is alpha to be a fixed number. action[2] controls convex combination of decoder + copilot")
parser.add_argument("-model",type=str,default="RecurrentPPO")
parser.add_argument("-softmax_type", default='two_peak', type=str,help="choose between (simple, complex, two_peak)")
parser.add_argument("-CS", default=0.7,type=float, help="percentage of correct softmax (does not apply for simple softmax)")
parser.add_argument("-stillCS", default=0.7, type=float, help="percentage of correct softmax for still state specifically (does not apply for simple softmax)")
parser.add_argument("-reward_type", default='exp_dist', type=str, help="choose between (lin_dist,exp_dist,lin_angle)")
parser.add_argument("-target_predictor", type=str, default=None, help='type "truth" for perfect target predictor')
parser.add_argument("-target_predictor_input", default="softmax", help="choose between (softmax,softmax_pos)")
parser.add_argument("-obs",default=[], nargs='+',help="can have several items: hold, vel, acc. these information is always added in alphabetical order")
parser.add_argument("-use_realtime", default=0, help="1 means real time, 2 means twice as fast, 0 means not using it")
parser.add_argument("-no_holdtime", default=False, action='store_true', help="if you want to have no hold time (hold_time_thres = 0.02 instead of 50ms), then set this flag")
parser.add_argument("-holdtime", default=0.5, type=float, help="uses this original holdtime of 500ms unless you change it")
parser.add_argument("-showSoftmax", default=False, action='store_true')
parser.add_argument("-showHeatmap", default=False, action='store_true')
parser.add_argument("-showVelocity", default=False, action='store_true')
parser.add_argument("-showReward", default=False, action='store_true',help="shows reward using pyplot")
parser.add_argument("-showRewards", default=False, action='store_true',help="shows multiple rewards using pyplot")
parser.add_argument("-showNonCopilotCursor", default=False, action='store_true', help='press space to see it')
parser.add_argument("-showMass", default=False, action='store_true')
parser.add_argument("-extra_targets",default={}, nargs='+',help='xy coordinate of new target (multipe of 2). i.e) 0.0 0.0 0.0 0.4')
parser.add_argument("-extra_targets_yaml",default=None, type=str, help='feed in new targets only by yaml file name i.e) grid-16.yaml. look inside SJtools/copilot/targets/')
parser.add_argument("-center_out_back", default=False, action='store_true', help="creates a game such that it is center out back")
parser.add_argument("-history",default=[1,0], nargs='+', help="[n, interval] n history with t interval ex) -history 3 100 looks at 3 history (including current) with 100 timestep interval")
parser.add_argument("-maskSoftmax", default=None, type=str, help='all or partial or None. all means removing all softmax history')
parser.add_argument("-binaryAlpha",default=False, action='store_true')
parser.add_argument("-obs_heatmap",default=None,type=int,help="put in one number (n) to create nxn heatmap to be fed as observation")
parser.add_argument("-obs_heatmap_option",default=[], nargs='+',help="should only be set up of args.obs_heatmap=int. may add other obs option such as (com = center of mass of heatmap (x,y), bcc= bincount of cursor (counter where cursor is located), only = no heatmap used in obs, df0.9 = decrement factor how much to decrement obs_heatmap) i.e [cf0.9, comonly]")
parser.add_argument("-softmaxStyle", default='cross', type=str,help="cross, bar, cursor")
parser.add_argument("-device",default="cpu", help='default will choose the best one, but can specify (cpu,cuda,mps // cpu is fastest on macbook)')
parser.add_argument("-from_readpy", default=False, action='store_true',help="setting it true means, this file is executed by read.py from runs/read.py. simply means id it passes should be run")
parser.add_argument("-pause", default=False, action='store_true',help="'press 'enter' key to move one step press 's' key to speedily move; allows user to control next step")
parser.add_argument("-followCursorAction", default=False, action='store_true',help="press 'f' key. then cursor explicitely (controlling action to move cursor)")
parser.add_argument("-followCursorSoftmax", default=False, action='store_true',help="softmax generated with keyboard (controlling softmax to move cursor)")
parser.add_argument("-eval_metric", default=[], nargs='+', help=' various metric of evaluation: [closeness (=per trial), avgTimeToHit (=across trials), avgDistance (=across trials), avgExtraDistance (=across trials)')
    
parser.add_argument('-tools',nargs='+', default=[],help='set this flag to use various tools provided by raspy i.e) seeTrajectory, seeCopilotContribution (red copilot, green softmax)')
parser.add_argument('-silence', default=False, action='store_true',help="silence action space")

args = parser.parse_args()

filePath = "SJtools/copilot/"
fileName = "SJtools/copilot/models/SJ-4-directions-2drtkeiz/SJ-4-directions-2drtkeiz"

if len(args.argv) > 0:
    fileName = filePath + sys.argv[1]
if len(args.argv) > 1 and isfloat(sys.argv[2]):
    args.alpha = float(sys.argv[2])

if len(args.history) == 2: args.history = [int(i) for i in args.history]
else: print("ERROR: args.history must be 2 value list"); exit(1)
args.use_realtime = False if args.use_realtime == 0 else float(args.use_realtime)

# read copilot yaml param
copilotYamlParam = fileName if os.path.exists(fileName+'.yaml') else None

# filePath from readpy
if args.from_readpy: 
    fileName = "SJtools/copilot/runs/" + sys.argv[1] + f"/{sys.argv[2]}_model"
    copilotYamlParam = "SJtools/copilot/runs/" + sys.argv[1] + "/model"

if type(copilotYamlParam)==str:
    copilotYamlParamFile = open(copilotYamlParam + ".yaml", 'r')
    copilotYamlParam = yaml.load(copilotYamlParamFile, Loader=yaml.Loader)

    # read targets if not specified by argsparse
    if len(args.extra_targets) == 0: args.extra_targets = copilotYamlParam['targets']['extra_targets']
    if args.extra_targets_yaml == None: args.extra_targets_yaml = copilotYamlParam['targets']['extra_targets_yaml']
    


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


# create env
env = SJ4DirectionsEnv(render=True,showSoftmax=args.showSoftmax,showVelocity=args.showVelocity,showHeatmap=args.showHeatmap,softmax_type=args.softmax_type,reward_type=args.reward_type,setAlpha=args.alpha,CSvalue=args.CS,stillCS=args.stillCS,useTargetPredictor=args.target_predictor,target_predictor_input=args.target_predictor_input,holdtime=args.holdtime,extra_targets=args.extra_targets,extra_targets_yaml=args.extra_targets_yaml,obs=args.obs,historyDim=args.history,maskSoftmax=args.maskSoftmax,binaryAlpha=args.binaryAlpha,copilotYamlParam=copilotYamlParam,obs_heatmap=args.obs_heatmap,obs_heatmap_option=args.obs_heatmap_option,showNonCopilotCursor=args.showNonCopilotCursor,softmaxStyle=args.softmaxStyle,hideMass=not args.showMass,center_out_back=args.center_out_back,tools=args.tools)

# show multiple Rewards
if args.showRewards:
    rewards = {
        'exp angle': RewardClass(env,yamlPath='pastExpAngle.yaml'),
        'exp dist': RewardClass(env,yamlPath='pastExpDist.yaml'),
        'lin angle': RewardClass(env,yamlPath='pastLinAngle.yaml'),
        'lin dist': RewardClass(env,yamlPath='pastLinDist.yaml'),
        # 'new exp angle': RewardClass(env,yamlPath='expAngle.yaml'),
    }

# extract info from copilot yaml if there is some:
if env.taskGame.copilotInfo is not None: 
    args.model = env.taskGame.copilotInfo.get('model',args.model)

# use appropriate model
if args.model == "RecurrentPPO":
    model = RecurrentPPO.load(fileName,device=args.device)
if args.model == "PPO":
    model = PPO.load(fileName,device=args.device)

# print(model.policy)
# print(model.policy.device)
# print("is model on device: ",next(model.policy.mlp_extractor.policy_net.parameters()).device)

if args.showReward or args.showRewards: 
    dim = 1 + len(rewards) if args.showRewards else 1
    labels = ['reward']+list(rewards.keys()) if args.showRewards else None
    rewardPlot = AnimatedReward(dim=dim,ymax=[110,1000],labels=labels,n=70)

obs = env.reset()
episode_start = True
reward = 0
_states = None

key_up_once = True
key_down_once = False
past = time.time()

totalScore = 0.0
initialDist = None
def evaluationByDistance(env,info,done=False):
    # (Time it stayed near target) x (1-distance from the target) x (trialTime)
    global totalScore
    global initialDist

    totalTickN = env.taskGame.activeTickLength
    targetPos = info["target_pos"]
    cursorPos = info["cursor_pos"]
    timeGoing = 1 - env.taskGame.TaskCopilotObservation.timeRemain
    distance = np.linalg.norm(targetPos-cursorPos)
    if initialDist is None: initialDist = distance
    distScore = (initialDist-distance) / initialDist # 2 is maximum distance in this game
    totalScore += distScore * timeGoing
    # print("{:0.2f}".format(distScore), timeGoing)
    if done:
        finalScore = totalScore
        totalScore = 0.0
        initialDist = None
        return finalScore / totalTickN
    else:
        return None

while True:
    action, _states = model.predict(obs, state=_states, deterministic=True, episode_start=episode_start)
    episode_start = False

    # override model action by pressing f (follow)
    if args.followCursorAction and env.taskGame.key_pressed.get('f',False):
        v = env.taskGame.mousePos - env.taskGame.cursorPos
        vnorm = np.linalg.norm(v)
        vthres = 0.1
        if vnorm > vthres: action[:2] = v / vnorm
        else: action[:2] = v / vnorm * (vnorm / vthres)
        action[2] = 1
        print('override',action)
    else:
        if not args.silence: print(action,reward,obs)
    

    if env.taskGame.key_pressed.get('d',False): # do as without copilot
        env.taskGame.TaskCopilotAction.copilot_alpha = -1
    else:
        env.taskGame.TaskCopilotAction.copilot_alpha = env.taskGame.copilot_default_alpha

    # if not args.from_readpy: print(np.round(action, 2))
    obs, reward, done, info = env.step(action)


    # new evaluation metric by closeness to correct
    if 'closeness' in args.eval_metric:
        evalScore = evaluationByDistance(env,info,done)
        if evalScore is not None:
            print('\033[91m' + '\033[1m' + 'closeness score: ' + str(evalScore) + '\033[0m')

    # pause needed
    if args.pause: 
        while True:
            env.taskGame.check_pygame_event()
            enter_pressed = env.taskGame.key_pressed.get('return',False)
            s_pressed = env.taskGame.key_pressed.get('s',False)
            sincePressed = None
            if not enter_pressed and key_down_once:
                key_down_once = False
                key_up_once = True
            if enter_pressed and key_up_once or s_pressed:
                key_down_once = True
                key_up_once = False
                break

    # show reward
    if args.showReward or args.showRewards: 
        value = [reward]
        if args.showRewards: value += [v.getReward(info['result'],action) for _,v in rewards.items()]
        rewardPlot.valueUpdate(value)
        rewardPlot.animationUpdate()

    
    # real time
    if args.use_realtime > 0: 
        diff = 0.02 / args.use_realtime - (time.time()-past)
        if diff > 0: time.sleep(diff)
        past = time.time()

    
    # render = env.render()
    if done: 

        # show more metric
        if 'avgTimeToHit' in args.eval_metric: print('\033[91m' + '\033[1m' + 'avg time to hit score: ' + str(env.taskGame.PerformanceRecord.avgTimeToHit) + '\033[0m')
        if 'avgDistance' in args.eval_metric: print('\033[91m' + '\033[1m' + 'avg distance score: ' + str(env.taskGame.PerformanceRecord.avgDistanceTravelled) + '\033[0m')
        if 'avgExtraDistance' in args.eval_metric: print('\033[91m' + '\033[1m' + 'avg extra distance score: ' + str(env.taskGame.PerformanceRecord.avgExtraDistanceTravelled) + '\033[0m')

        obs = env.reset()
        episode_start = True
        if args.showRewards: [v.reset() for _,v in rewards.items()]

        