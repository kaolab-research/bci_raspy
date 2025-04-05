import re
import gym
from gym import spaces
import numpy as np
from modules.kf_4_directions_constructor import SJ_4_directions
import math
from SJtools.copilot.targetPredictor.model import LSTM,LSTM2,LSTMFCS,NN
from SJtools.copilot.copilotUtils.rewards import RewardClass
import torch
import yaml
import sys
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


""" ============== PREARE ENV FUNCTION =============="""

N_DISCRETE_ACTIONS = 3
N_STATE = 5
N_POS_DIM = 2
N_OBS = N_STATE + N_POS_DIM
STILL_STATE = N_STATE - 1

# def _getVelocity (env,arg):
#   """ generate action give softmax and action of RL """

#   # break arg into two
#   RLaction,softmax = arg
#   pos,alpha = RLaction[:2], (RLaction[2]+1) / 2

#   # decode softmax to vel (x,y)
#   decodedVel = env.decodedVel[np.argmax(softmax)]

#   # return convex combination
#   return alpha * pos + (1-alpha) * decodedVel


def getCorrectDirection(cursorPos, targetPos, targetSize, still_state, two=False):
  """ return two direction if two is set to True """

  # by target position
  state = None # either 0~4 LRUD
  dirs = targetPos - cursorPos
  posDirs = abs(dirs) 
  
  
  correct= [0,0]

  #x direction
  sign = dirs[0]
  if sign > 0: correct[0] = 1
  else: correct[0] = 0

  #y direction 
  sign = dirs[1]
  if sign > 0: correct[1] = 2
  else: correct[1] = 3
  
  xy = np.argmax(posDirs)
  state = correct[xy]

  inTarget = (posDirs[0] <= (targetSize[0]/2)) * (posDirs[1] <= (targetSize[1]/2))
  if inTarget: 
    state = still_state
    correct = [state,correct[xy]]

  if two: return correct[xy],correct[xy-1]
  else: return state


chosenDirectionTimer = 0
chosenDirection = 0
def complexSoftmax(env, arg, CSvalue=0.7, stillCS=0.7):
  global chosenDirection, chosenDirectionTimer
  """ returns synthetic softmax for creating game observation """
  """ CSvalue percentage of correct softmax (does not apply for simplesoftmax!) """

  [cursorPos, targetPos, targetSize, state_taskidx, game_state, detail] = arg
  still_state = env.stillId

  # 80% of time correct direction, 20% of time deteremine how long the wrong target will last
  if chosenDirectionTimer > 0:
    chosenDirectionTimer -= 1
  else:
    # find correct direction
    correctDirection = getCorrectDirection(cursorPos, targetPos, targetSize, still_state)

    # choose correct direction depending on stillCS or CSvalue
    if correctDirection == still_state:
      useCorrectDirection = np.random.random() <= stillCS
    else:
      useCorrectDirection = np.random.random() <= CSvalue

    # once decided, fix on a direction
    if useCorrectDirection: # percentage of correct softmax
      # correct target
      chosenDirection = correctDirection
      chosenDirectionTimer = np.random.randint(20,40)
    else:
      # wrong direction
      # wrong direction
      directions = list(range(N_STATE))
      directions.remove(correctDirection)
      chosenDirection = directions[np.random.randint(N_STATE-1)]
      chosenDirectionTimer = np.random.randint(20,40)
  
  # prepare one hot vector
  stateVector = np.zeros(N_STATE)
  stateVector[chosenDirection] = 1
  
  # 50 % of the time shave off top 40% and distribute it to others
  shaveOff = (np.random.random()*2-1) * 0.4
  shaveOff = 0 if shaveOff < 0 else shaveOff

  # distribute shaveOff semi equally to rest
  equal = np.random.rand(5)
  equal[chosenDirection] = 0
  equal = equal / np.sum(equal)
  equal *= shaveOff
  stateVector += equal
  stateVector[chosenDirection] -= shaveOff
  softmax = stateVector

  # print(state_taskidx,cursorPos,chosenDirection,softmax)
  # print(state_taskidx,np.argmax(softmax),softmax)

  return softmax

def simpleSoftmax(env,arg):
  # cannot handle extra_target more than cardianl directional targets!! will throw an error if you attept to do so

  [cursorPos, targetPos, targetSize, state_taskidx, game_state, detail] = arg
  still_state = env.stillId

  state_taskidx  = getCorrectDirection(cursorPos, targetPos, targetSize, still_state)
  target_encoding = np.zeros((N_STATE))
  target_encoding[state_taskidx] = 1
  target_encoding += 3*np.random.rand(N_STATE)
  target_encoding = np.exp(target_encoding) / np.sum(np.exp(target_encoding))
  return target_encoding

def halfPeakSoftmax(env,arg):
  [cursorPos, targetPos, targetSize, state_taskidx, game_state, detail] = arg
  still_state = env.stillId

  peak1,peak2  = getCorrectDirection(cursorPos, targetPos, targetSize, still_state, two=True)
  target_encoding = np.zeros((N_STATE))
  target_encoding[peak1] = 0.5
  target_encoding[peak2] = 0.5
  target_encoding += 3*np.random.rand(N_STATE)
  target_encoding = np.exp(target_encoding) / np.sum(np.exp(target_encoding))
  return target_encoding



previousDirection = 0
transitionTimer = 0
transitionTimerLength = 5
def twoPeakSoftmax(env,arg,CSvalue,stillCS=0.7):
  # there is always two peak with a tid bit noise
  # 70% of the time, correct peak
  # 30% wrong peak chosen
  # if two different peak chosen, then there is transition between the two 15-20 frame
  # shave off still applies but only to top 5% (to suggest noise, randomly distributed)

  global chosenDirection, chosenDirectionTimer, previousDirection, transitionTimer, transitionTimerLength
  """ returns synthetic softmax for creating game observation """
  """ CSvalue percentage of correct softmax (does not apply for simplesoftmax!) """

  [cursorPos, targetPos, targetSize, state_taskidx, game_state, detail] = arg
  still_state = env.stillId

  # 80% of time correct direction, 20% of time deteremine how long the wrong target will last
  if chosenDirectionTimer > 0:
    chosenDirectionTimer -= 1
  else:
    correctDirection = getCorrectDirection(cursorPos, targetPos, targetSize, still_state)

    # choose correct direction depending on stillCS or CSvalue
    if correctDirection == still_state:
      useCorrectDirection = np.random.random() <= stillCS
    else:
      useCorrectDirection = np.random.random() <= CSvalue

    if useCorrectDirection: # percentage of correct softmax
      # correct target
      chosenDirection = correctDirection
      chosenDirectionTimer = np.random.randint(20,40)
      transitionTimerLength = np.random.randint(15,20)
      transitionTimer = 0
    else:
      # wrong direction
      directions = list(range(N_STATE))
      directions.remove(correctDirection)
      chosenDirection = directions[np.random.randint(N_STATE-1)]
      
      chosenDirectionTimer = np.random.randint(20,40)
      transitionTimerLength = np.random.randint(15,20)
      transitionTimer = 0
  

  # transition
  if transitionTimer == transitionTimerLength:
    previousDirection = chosenDirection

  if transitionTimer < transitionTimerLength:
    transitionTimer += 1
    
  if transitionTimer < transitionTimerLength and chosenDirection != previousDirection:
    stateVector = np.zeros(N_STATE)
    stateVector[chosenDirection] = 1 * transitionTimer / transitionTimerLength
    stateVector[previousDirection] = 1 * (1- transitionTimer / transitionTimerLength)
  else:
    # prepare one hot vector
    stateVector = np.zeros(N_STATE)
    stateVector[chosenDirection] = 1
  
  # 50 % of the time shave off top 5% and distribute it to others
  shaveOff = (np.random.random()*2-1) * 0.05
  shaveOff = 0 if shaveOff < 0 else shaveOff

  # distribute shaveOff semi equally to rest
  equal = np.random.rand(5)
  equal[chosenDirection] = 0
  equal = equal / np.sum(equal)
  equal *= shaveOff
  stateVector += equal
  stateVector[chosenDirection] -= shaveOff
  softmax = stateVector

  return softmax

prevVel = np.zeros(2)
desiredVel = np.zeros(2)
transitionTimer = 0
transitionTimerLength = 5
chosenDirection2 = previousDirection2 = np.zeros(2)
def normalTargetSoftmax(env,arg):
  # there is always two vel. always heading toward new vel
  # the new vel will be chosen by finding optimal angle, and magnitude. both of which will encounter some additive noise
  # then as the velocity moves to the new vel. concentric noise will be added along the path
  # finally this velocity will be converted to softmax. please take a look at slide 2023/12/05

  global chosenDirection2, chosenDirectionTimer, previousDirection2, transitionTimer, transitionTimerLength
  [cursorPos, targetPos, targetSize, state_taskidx, game_state, detail] = arg

  # vel
  if chosenDirectionTimer > 0:
    chosenDirectionTimer -= 1
  else:
    # choose correct direction using angle, magnitude normal noise
    optimalDir = (targetPos - cursorPos)
    optimalAngle = np.arctan2(optimalDir[1],optimalDir[0])
    lessOptimalAngle = (optimalAngle + np.random.normal(0, np.pi/3))
    lessOptimalAngle = (lessOptimalAngle + np.pi) % (2 * np.pi) - np.pi # wrap
    selectedMag = np.random.normal(0.5, 0.1) # mean,var
    chosenDirection2 = np.array([np.cos(lessOptimalAngle),np.sin(lessOptimalAngle)]) * selectedMag
  
    # choose duration
    chosenDirectionTimer = np.random.randint(15,30)
    transitionTimerLength = np.random.randint(10,15)
    transitionTimer = 0

  # transition
  if transitionTimer == transitionTimerLength: previousDirection2 = chosenDirection2
  elif transitionTimer < transitionTimerLength: transitionTimer += 1
  if transitionTimer < transitionTimerLength:
    vel = chosenDirection2 * (transitionTimer / transitionTimerLength) + previousDirection2 * (1- transitionTimer / transitionTimerLength)
  else:
    vel = chosenDirection2

  # add noise to the velocity at the end
  vel += np.random.normal(0, 0.03, 2)

  # convert vel to softmax
  softmax = np.zeros(5)
  if vel[0] < 0: softmax[0] = -vel[0]
  elif vel[0] > 0: softmax[1] = vel[0]
  if vel[1] > 0: softmax[2] = vel[1]
  if vel[1] < 0: softmax[3] = -vel[1]

  return softmax

def _getDone(env,result):
  game_state = result[4]
  return game_state.isupper()

""" CircularQueue for history """
class CircularQueue():
  def __init__(self,dim,obs_size):
    self.n_history = dim[0]
    self.interval = dim[1]
    self.obs_size = obs_size

    self.data_dim = ((self.n_history-1) * self.interval + 1, obs_size)
    self.n = self.data_dim[0]
    self.reset()

  def add(self,v):
    self.i += 1
    if self.i == self.n: self.i = 0
    self.data[self.i] = v

  def get(self):
    # return n history with interval
    selected = self.i - np.arange(self.n_history) * self.interval 
    return self.data[selected]
  
  def add_get(self,v):
    self.add(v)
    return self.get()
  
  def reset(self):
    self.data = np.zeros(self.data_dim)
    self.i = 0

  def resetAllSoftmax(self):
    self.data[:,2:7] = 0




""" ============== GENERATE ENV ============== """

class SJ4DirectionsEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, isEval=False, wandbUsed=False, getVelocity=None,getSoftmax=None,getDone=None,render=False,showSoftmax=False,showVelocity=False,showHeatmap=False,softmax_type='complex',reward_type=None,setAlpha=None,CSvalue=0.7,stillCS=0.7,useTargetPredictor=None,target_predictor_input="softmax",cursor_target_obs=False,holdtime=None,randomInitCursorPosition=False,curriculum_yaml=None,extra_targets={},extra_targets_yaml=None,obs=[],action=['vx','vy','alpha'],historyDim=[1,0],historyReset='zero',maskSoftmax=None,binaryAlpha=False,copilotYamlParam=None,obs_heatmap=None,obs_heatmap_option=[],showNonCopilotCursor=False, gamify=False, gamifyOption=[],softmaxStyle='cross',hideMass=False,center_out_back=False,velReplaceSoftmax=False,cardinalVelocityThres=False,noSoftmax=False,tools=[],trial_per_episode=1,action_param=[]):
    super(SJ4DirectionsEnv, self).__init__()
    
    if getVelocity is not None: self.getVelocity = getVelocity
    if getSoftmax is not None: self.getSoftmax = getSoftmax
    # if getDone is not None: self.getDone = getDone # voided due to trial_per_epsiode. see def getDone

    # create and save all the copilot related yaml needed by the task (this will be exported when training finishes)
    if copilotYamlParam is None: self.copilotYamlParam = self.create_copilot_yaml_param(obs=obs,target_predictor=useTargetPredictor,target_predictor_input=target_predictor_input,cursor_target_obs=cursor_target_obs,history=historyDim,historyReset=historyReset,maskSoftmax=maskSoftmax,obs_heatmap=obs_heatmap,obs_heatmap_option=obs_heatmap_option,action=action,alpha=setAlpha,velReplaceSoftmax=velReplaceSoftmax,cardinalVelocityThres=cardinalVelocityThres,noSoftmax=noSoftmax,trial_per_episode=trial_per_episode,action_param=action_param)
    else: 
      if setAlpha is not None: copilotYamlParam['action_dim']['alpha'] = float(setAlpha)
      self.copilotYamlParam = copilotYamlParam
    self.taskGame = SJ_4_directions(render=render,useRealTime=False,showAllTarget=True,showSoftmax=(showSoftmax and render),showVelocity=(showVelocity and render),showHeatmap=(showHeatmap and render),randomInitCursorPosition=randomInitCursorPosition,copilotYamlParam=self.copilotYamlParam,showNonCopilotCursor=showNonCopilotCursor,gamify=gamify,gamifyOption=gamifyOption,softmaxStyle=softmaxStyle,hideMass=hideMass,taskTools=tools)

    # create reward


    # define obs dim from obs dim defined by SJ_4_directions
    self.obs_dim = self.taskGame.n_copilotObsDim
    self.observation_space = spaces.Box(low=-1, high=1, shape= (self.obs_dim,), dtype=np.float32)
    print(f"Observation Space: {self.obs_dim}")

    # obs and action space (3 value, x,y,alpha, where alpha is confidence about its action)
    # self.action_dim = len(self.taskGame.copilotActionTypes)
    self.action_dim = self.taskGame.n_copilotActionDim
    self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
    self.alphaIndex = self.taskGame.copilotActionTypes.index("alpha") if "alpha" in self.taskGame.copilotActionTypes else -1
    print(f"Action Space: {self.action_dim}")

    if extra_targets_yaml is not None:
      self.taskGame.useTargetYamlFile(extra_targets_yaml)
    self.taskGame.addTargets(extra_targets)

    # no need to train on still
    if 'still' in self.taskGame.desiredTargetList:  self.taskGame.desiredTargetList.remove('still')

    if center_out_back:
      self.taskGame.centerIn = True
      self.taskGame.resetCursorPos = False
      self.taskGame.inactiveLength = self.taskGame.inactiveTickLength = 0
      self.taskGame.delayedLength = self.taskGame.delayedTickLength = 0
      self.taskGame.episodeLength = self.taskGame.activeLength
      self.taskGame.episodeTickLength = self.taskGame.activeTickLength
      if 'still' in self.taskGame.desiredTargetList:  self.taskGame.desiredTargetList.remove('still')
      size = self.taskGame.targetsInfo[self.taskGame.desiredTargetList[0]][1] 
      self.taskGame.addTargets({'center':[np.array([0., 0.],),size]})
      self.taskGame.centerIn = True
      self.taskGame.ignoreWrongTarget = True
      self.taskGame.showAllTarget = False
      self.taskGame.desiredTargetList.remove('center')

    # redefine action space, for charge targets
    self.action_dim = self.taskGame.TaskCopilotAction.n_copilotActionDim
    self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
    print(f"RE: Action Space: {self.action_dim}")
    
      
      

    self.still = 'still'
    self.stillId = self.taskGame.target2state_task[self.still]

    if holdtime is not None: self.taskGame.holdTimeThres = self.taskGame.dwellTimeThres = holdtime
    self.decodedVel = self.taskGame.decodedVel
    self.softmax_type = softmax_type
    self.reward_type = reward_type
    self.isEval = isEval # is the env eval eval?
    self.wandbUsed = wandbUsed # wandb is used
    self.maskSoftmax = maskSoftmax
    self.binaryAlpha = binaryAlpha
    self.setAlpha = setAlpha
    self.CSvalue = CSvalue # percentage of correct softmax (does not apply for simple softmax)
    self.stillCS = stillCS # percentage of correct softmax for still state specfically(does not apply for simple softmax)
    self.gamify = gamify
    self.cursor_target_obs = cursor_target_obs
    self.curriculum_yaml = curriculum_yaml
    self.use_curriculum = self.curriculum_yaml is not None and not self.isEval
    
    self.reward_wrong = -10000 # wrong penalty
    self.reward_tolerated_wrong = -500 # tolerated wrong penalty

    # initial values
    self.needReset = False
    self.episodicTrialCounter = 0
    self.softmax = np.ones(N_STATE) / N_STATE

    # self.trial_per_episode
    if 'reset' in self.copilotYamlParam and 'trial_per_episode' in self.copilotYamlParam['reset']:
      self.trial_per_episode = self.copilotYamlParam['reset']['trial_per_episode']
    else:
      self.trial_per_episode = trial_per_episode
    
    # curriculum
    if self.use_curriculum:
      self.curriculum = self.init_curriculum_param_with_yaml(self.curriculum_yaml)
      self.taskGame.setAttributeFromCurriculum(self.curriculum)
      self.setAttributeFromCurriculum(self.curriculum)

    # print some notes to user
    print(f"Note: using {self.softmax_type} softmax")
    if self.softmax_type != 'simple': print("Note: using softmax with CS:",self.CSvalue)
    print("Note: using reward function:",self.reward_type)
    if self.setAlpha is not None: print("Note: using set alpha =",self.setAlpha)
    if useTargetPredictor is not None: print("Note: using target predictor")
    print()

    # initialize result
    self.result = [np.array([0,0]),np.array([np.NAN,np.NAN]),-1,0]  # [cursorPos, targetPos, state_taskidx, success]

    # initialize trial results
    self.trialResults = [] # should only count trials with actual targets
    self.currentTrialLength = 0

    # start by skipping! skip first stop sign
    if reward_type == "exp_dist": self.rewardClass = RewardClass(self,yamlPath='pastExpDist.yaml')
    elif reward_type == "lin_dist": self.rewardClass = RewardClass(self,yamlPath='pastLinDist.yaml')
    elif reward_type == "exp_angle": self.rewardClass = RewardClass(self,yamlPath='pastExpAngle.yaml')
    elif reward_type == "lin_angle": self.rewardClass = RewardClass(self,yamlPath='pastLinAngle.yaml')
    else: self.rewardClass = RewardClass(self,yamlPath=reward_type)
    print("Reward Type:",self.rewardClass.yamlPath)

    self.reset() 

  def create_copilot_yaml_param(self,obs=[],target_predictor=None,target_predictor_input="softmax",cursor_target_obs=False,history=[1,0],historyReset='zero',maskSoftmax=None,obs_heatmap=None,obs_heatmap_option=[],action=["vx","vy","alpha"],alpha=None,velReplaceSoftmax=False,cardinalVelocityThres=False,noSoftmax=False,trial_per_episode=1,action_param=[]):
    """create dictionary of copilot yaml and return it"""

    action_param_dict = {}
    for i in range(0,len(action_param),2):
      key = action_param[i]
      value = action_param[i+1]
      action_param_dict[key] = value

    param = {
      "obs_dim":{
          "obs":obs, # "can have several items: hold, vel, acc. these information is always added in alphabetical order"
          "target_predictor":target_predictor, # default None
          "target_predictor_input":target_predictor_input, # default softmax, choose between (softmax,softmax_pos) 
          "cursor_target_obs": cursor_target_obs, # default False set True if you only want 4 dim which is [cursor(2),target(2)]
          "history":history, # default [1,0], nargs='+', help="[n, interval] n history with t interval ex) -history 3 100 looks at 3 history (including current) with 100 timestep interval")
          "historyReset":historyReset,
          "maskSoftmax": maskSoftmax, # default None type=str, help='all or partial or None. all means removing all softmax history')
          "heatmap": obs_heatmap,
          "obs_heatmap_option": obs_heatmap_option,
          "velReplaceSoftmax": velReplaceSoftmax,
          "noSoftmax": noSoftmax,
      },
      "action_dim": {
          "alpha":alpha,
          "action": action,
          "action_param": action_param_dict,
      },
      "reset": {
        "trial_per_episode": trial_per_episode,
      }
    }

    # additional parameter
    if cardinalVelocityThres:
      param["obs_dim"]["cardinalVelocityThres"] = float(cardinalVelocityThres)
    
    return param


  # default generate softmax
  def getSoftmax(self,env,arg): 
    if self.softmax_type == 'complex':
      return complexSoftmax(env,arg,self.CSvalue,self.stillCS)
    elif self.softmax_type == 'normal_target':
      return normalTargetSoftmax(env,arg)
    elif self.softmax_type == 'two_peak':
      return twoPeakSoftmax(env,arg,self.CSvalue,self.stillCS)
    elif self.softmax_type == 'half_peak':
      return halfPeakSoftmax(env,arg)
    elif self.softmax_type == 'simple':
      return simpleSoftmax(env,arg)
    else: 
      print("getSoftmax error")
      exit(1)


  # default generate done
  def getDone(self,env,arg): 
    """
    return reset, done
    reset: bool that tells whether we must reset parameter or not (for new trial within episode)
    done: bool that tells whether episode is over.
    """
    done = _getDone(env,arg)
    if not done: return False, False
    else:
      self.episodicTrialCounter += 1
      if self.episodicTrialCounter < self.trial_per_episode:
        return True, False # reset must be made internally
      else:
        self.episodicTrialCounter = 0
        return False, True # reset will happen externally
      

  def step(self, action):
    # Execute one time step within the environment

    if self.needReset: # internal reset created by trial_per_episode > 1
      obs = self.reset()
      reward = 0
      done = False
      self.needReset = False
      return obs, reward, done, {}

    if self.binaryAlpha:
      action[2] = 1 if action[2] > 0 else -1

    # get softmax and vel
    observed_softmax = self.softmax

    # update task game
    if self.setAlpha is not None and self.alphaIndex != -1: action[self.alphaIndex] = self.setAlpha # this line maybe removed now
    self.action = action
    self.result = self.taskGame.update([observed_softmax,action]) # softmax, decoded_Vel (only send decoded_vel)

    # record success (AND change target size)
    self.recordSuccess(self.result)

    # get reward
    reward = self.rewardClass.getReward(self.result, action, self.pastObs)

    # get done
    self.needReset, done = self.getDone(self,self.result)

    # get obs
    softmax = self.getSoftmax(self,self.result)

    option = None
    if self.alphaIndex != -1 and action[self.alphaIndex] > 0:
      option = "mask" # masking happens only if masking argument is passed to taskGame. but regardless, instruction to mask is sent to get_env_obs everytime alpha > 0
    
    self.pastObs = obs = self.taskGame.get_env_obs(softmax, action=option)

    # save softmax for next iteration
    self.softmax = softmax
    
    return obs, reward, done, {"target_pos":self.result[1],"task_id":self.result[3],"softmax":softmax,"cursor_pos":self.result[0],"result":self.result}

  def reset(self):
    
    skipped = False
    while True:
      if self.taskGame.activeTrialHasBegun: # result[3] != -1: #state_taskidx  

        # store this for reward shaping
        self.rewardClass.reset()
        
        # get softmax and obs
        softmax = self.softmax
        self.pastObs = obs = self.taskGame.get_env_obs(softmax, action="reset")
        self.softmax = softmax
        
        return obs
      
      else: 
        # Reset the state of the environment to an initial state
        defaultSoftmax = np.ones(N_STATE) / N_STATE

        # create default action
        defaultAction = np.zeros(self.action_dim)
        if self.alphaIndex != -1: defaultAction[self.alphaIndex] = 1

        result = self.taskGame.update([defaultSoftmax, defaultAction])
        skipped = True

  def render(self, mode='human', close=False):
    # Render the environment to the screen

    return np.ones((16,16),dtype=np.uint8)*255
  
  successChain = 0 # for fun / view not for training
  def recordSuccess(self,result):
    # record success inside trialResults (-1 timeout, -2 means wrong hit, pos number means how many timesteps it took)
    game_state = result[4]
    if game_state.islower():
      # game unfinished
      self.currentTrialLength += 1
    else:
      # game finished
      if game_state == 'H':
        self.trialResults.append(self.currentTrialLength)

        self.successChain += 1
        if self.successChain > 1: print("success chain:",self.successChain)
        # if it last trial was successful, see if curriculum needs to be updated
        if self.use_curriculum:
          self.update_curriculum(self.trialResults,self.curriculum)

      else: # timeout, wrong hit
        targetPos = result[1]
        if np.isnan(targetPos).sum() == 0: # skip nan targets like still state
          result = -2 if game_state == 'W' else -1
          self.trialResults.append(result)
        self.successChain = 0
      self.currentTrialLength = 0
  
  def init_curriculum_param_with_yaml(self,curriculum_yaml_name):
    yamlPath = f"SJtools/copilot/curriculum/{curriculum_yaml_name}"
    with open(yamlPath) as yaml_file:
        yamlData = yaml.load(yaml_file, Loader=Loader)
    
    # include current variable that keeps track of current state. this variable maybe refreneced outside by callbacks
    for name in yamlData:
      yamlData[name]["current"] = yamlData[name]["start"]
      yamlData[name]["last_update"] = 0

    return yamlData
  
  def update_curriculum(self,trialResults,curriculum):
    # update taskGame according to curriculum given all trialResults
    # specificially, updates attribute (i.e target_size) bu rules
    # trialResults = [T,F] hit (positive numbders = hit) and misses (negative numbers)

    curriculum_updated = False # flag that helps it know if taskGame needs to be updated

    for attribute, rule in curriculum.items():

      # see if copilot run enough trials to begin curriculum update
      req_trials = rule["req_trials"]
      if len(trialResults[rule["last_update"]:]) < req_trials: continue

      correctOnes = (np.array(trialResults[rule["last_update"]:][-req_trials:]) > 0).sum()
      passingScores = rule["req_trials"] * rule["req_success"]
      
      # if you have reached the end, no need to update
      if round(abs(rule["end"] - rule["current"]) / abs(rule["step"])) == 0: continue

      # if there is enough success and needs a curriculum update
      if correctOnes >= passingScores:
        if (rule["end"] - rule["start"]) > 0: # if attribute needs to increase
          if rule["end"] > rule["current"]: 
            rule["current"] += abs(rule["step"])
            rule["last_update"] = len(trialResults)
            curriculum_updated = True
            print(f"Curriculum [{attribute}] Updated to {rule['current']} @ {rule['last_update']}")
            print(f"curriculum: {attribute} "+"{:.2f}".format(rule['current']), file=sys.stderr) # printing finalLog to stderr [look run.py]
        elif (rule["end"] - rule["start"]) < 0: # if attribute needs to decreases:
          if rule["end"] < rule["current"]: 
            rule["current"] -= abs(rule["step"])
            rule["last_update"] = len(trialResults)
            curriculum_updated = True
            print(f"Curriculum [{attribute}] Updated to {rule['current']} @ {rule['last_update']}")
            print(f"curriculum: {attribute} "+"{:.2f}".format(rule['current']), file=sys.stderr) # printing finalLog to stderr [look run.py]
        
      
    if curriculum_updated:
      self.taskGame.setAttributeFromCurriculum(curriculum)
      self.setAttributeFromCurriculum(curriculum)

    # if the trial results gets too big empty it time to time
    if len(trialResults) > 100000:
      for attribute, rule in curriculum.items(): rule["last_update"] = 0
      trialResults.clear()
      

  def setAttributeFromCurriculum(self,curriculum):
    if 'wrong_penalty' in curriculum: self.reward_wrong = curriculum['wrong_penalty']['current']
    if 'tolerated_wrong_penalty' in curriculum: self.reward_tolerated_wrong = curriculum['tolerated_wrong_penalty']['current']
    if 'CS' in curriculum: self.CSvalue = curriculum['CS']['current']