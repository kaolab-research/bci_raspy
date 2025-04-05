import typing
import SJtools.copilot.env
import numpy as np
import math
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from scipy import spatial

class RewardClass:
  """
  defines all possible rewards in a standardized method
  """

  def __init__(self, env, yamlPath=None):
    """ initialize """
    # extract all the constants needed for reward here
    self.env = env
    self.reward_wrong = env.reward_wrong
    self.reward_tolerated_wrong = env.reward_tolerated_wrong
    self.yamlPath = yamlPath
    self.fullYamlPath = f"SJtools/copilot/reward/{yamlPath}"
    self.activeT = self.env.taskGame.activeTickLength

    # constant parameters
    self.setParameterFromYaml()

    # private variable
    self.prev_pos = np.zeros(2)
    self.t = 0

  def reset(self):
    self.prev_pos = np.zeros(2)
    self.t = 0
    self.currentDecay = 1

  def setParameterFromYaml(self):
    """ set parameter according to yaml"""
    
    if self.yamlPath == 'custom':
      self.getReward = self.getCustomReward
      self.reset = self.customReset
      return

    # opening yaml
    with open(self.fullYamlPath) as yaml_file:
        yamlData = yaml.load(yaml_file, Loader=Loader)

    # set game state reward
    self.REWARD_GAME_STATE = {}
    self.REWARD_BASE = yamlData['game_state'].get('lower', -1)# h,w,n
    self.REWARD_GAME_STATE['t'] = yamlData['game_state'].get('t', -500) # tolerated  hi
    self.REWARD_GAME_STATE['T'] = yamlData['game_state'].get('T', -1) # Time out
    self.REWARD_GAME_STATE['W'] = yamlData['game_state'].get('W', -1000) # Incorrect hit
    self.REWARD_GAME_STATE['H'] = yamlData['game_state'].get('H', 1000)# Hit

    # set bonus reward parameter
    self.BONUS_STALL_IN_TARGET = yamlData['bonus_param'].get('stall_in_target', None)
    self.BONUS_CLOSER_TO_CENTER_CLIP = yamlData['bonus_param'].get('closer_to_center_clip',None)
    self.BONUS_CLOSER_TO_CENTER = yamlData['bonus_param'].get('closer_to_center',None) # having *10 makes the reward x10. having nothing makes the reward x100
    self.BONUS_SUBTRACTED_HOLD = yamlData['bonus_param'].get('subtracted_hold',None)
    self.TIME_DECAY = yamlData.get('time_decay', 1)
    self.currentDecay = 1 # updated back to 1 every episode

    # reward weighing (which reward weighs more?)
    self.weight_GAME_STATE = yamlData['reward_weight'].get('game_state',0)
    self.weight_STALL_IN_TARGET = yamlData['reward_weight'].get('stall_in_target',0)
    self.weight_CLOSER_TO_CENTER = yamlData['reward_weight'].get('closer_to_center',0)
    self.weight_SUBTRACTED_HOLD = yamlData['reward_weight'].get('subtracted_hold',0)
    self.weight_COPILOT_CONTRIBUTION = yamlData['reward_weight'].get('copilot_contribution',0)
    self.weight_SHAPING = yamlData['reward_weight'].get('shaping',0)

    # set True if legacy is used
    self.useLegacy = yamlData.get('legacy',False)
    if self.useLegacy: 
      self.getReward = self.getReward__legacy
      if yamlData['shaping'] == 'exp_dist': self.getShapingReward = self._getExpDistReward__legacy
      elif yamlData['shaping'] == 'lin_dist': self.getShapingReward = self._getLinDistReward__legacy
      elif yamlData['shaping'] == 'exp_angle': self.getShapingReward = self._getExpAngleReward__legacy
      elif yamlData['shaping'] == 'lin_angle': self.getShapingReward = self._getLinAngleReward__legacy
      else: print("unrecognized shaping parameter:",yamlData['shaping']); exit(1)

    # choose shaping
    if yamlData['shaping'] == 'exp_dist': self.getShapingReward = self._getExpDistReward
    elif yamlData['shaping'] == 'lin_dist': self.getShapingReward = self._getLinDistReward
    elif yamlData['shaping'] == 'exp_angle': self.getShapingReward = self._getExpAngleReward
    elif yamlData['shaping'] == 'lin_angle': self.getShapingReward = self._getLinAngleReward
    elif yamlData['shaping'] == 'lin_angle_travelled': self.getShapingReward = self._getLinAngleTravelledReward
    elif yamlData['shaping'] == 'mon_dist': self.getShapingReward = self._getMonotonicDistanceReward
    else: print("unrecognized shaping parameter:",yamlData['shaping']); exit(1)



  def getReward(self, result, action, obs):
    
    decoded_pos   = result[0]
    target_pos    = result[1]
    target_size   = result[2]
    game_state    = result[4]
    
    # base reward
    baseReward = self._getGameStateReward(game_state)

    # bonus reward 
    bonusReward_CTC = self._getBonusCloserToCenter(decoded_pos,target_pos,target_size,action)
    bonusReward_SIT = self._getBonusStallInTarget(decoded_pos,target_pos,target_size,action)
    bonusReward_SH = self._getBonusSubtractedHold(decoded_pos,target_pos,target_size,action)
    bonusReward_COP = self._getBonusCopilotContribution(decoded_pos,target_pos,target_size,action)

    # shaping reward
    shapingReward = self.getShapingReward(target_pos,decoded_pos,action)

    # combining all
    reward = baseReward * self.weight_GAME_STATE
    reward += bonusReward_CTC * self.weight_CLOSER_TO_CENTER
    reward += bonusReward_SIT * self.weight_STALL_IN_TARGET
    reward += bonusReward_SH * self.weight_SUBTRACTED_HOLD
    reward += bonusReward_COP * self.weight_COPILOT_CONTRIBUTION
    reward += shapingReward * self.weight_SHAPING

    # print(baseReward)
    # print(bonusReward_CTC)
    # print(bonusReward_SIT)
    # print(bonusReward_SH)
    # print(shapingReward)
    # print(reward)
    # print()

    if self.TIME_DECAY != 1:
      reward *= self.currentDecay

      # update time decay
      self.currentDecay *= self.TIME_DECAY
      self.currentDecay = np.clip(self.currentDecay,0.1,1)

      if game_state.isupper():
        self.currentDecay = 1.0

    return reward



  def _getLinDistReward(self,target_pos,decoded_pos,action):
    """ 
    Linear Distance Reward Shaping 
    Every step returns reward between 1 to -1
    """


    if np.sum(np.isnan(target_pos)) == 0:
      # RewardShaping: if target is nan don't worry about it
      distCurrent = math.dist(target_pos,decoded_pos)
      distPast = math.dist(target_pos,self.prev_pos)
      rewardShaping = (distPast-distCurrent) * (200/3)
      self.prev_pos = np.copy(decoded_pos)
  
      return rewardShaping
  
    return 0
  
  def _getExpDistReward(self,target_pos,decoded_pos,action):
    """ 
    Exponential Distance Reward Shaping 
    Every step outside the target should be between 1 to -1
    maximum reward gained inside target should be around 150 
    """
    sigma = 0.002 # to prevent division by zero
    factor = 1

    if np.sum(np.isnan(target_pos)) == 0:
      # RewardShaping: if target is nan don't worry about it
      distCurrent = math.dist(target_pos,decoded_pos) * factor
      distPast = math.dist(target_pos,self.prev_pos) * factor
      rewardShaping = (1/(distCurrent+sigma) - 1/(distPast+sigma)) 
      self.prev_pos = np.copy(decoded_pos)
  
      return rewardShaping
  
    return 0


  def _getLinAngleReward(self,target_pos,decoded_pos,action):
    """ 
    Linear Angle Reward Shaping:
    Every step returns reward between 1 to -1
    """
    factor = 2/np.pi

    if np.linalg.norm(action[:2]) == 0: return 0 # if not moving, reward is zero
    if np.sum(np.isnan(target_pos)) != 0: return 0 # target is nan (still state) #TODO: need to get rid of still state when training
    
    correctDir = (target_pos-decoded_pos)
    normCorrectDir = np.linalg.norm(correctDir)
    if normCorrectDir == 0: return 1 # if perfect, returns highest reward
    correctDir = correctDir / normCorrectDir

    # no longer uses action[:2] because action can control mass or acceleration
    # cursorDirNorm = np.linalg.norm(action[:2])
    # cursorDir = action[:2] / cursorDirNorm 
    movement = decoded_pos - self.prev_pos
    self.prev_pos = np.copy(decoded_pos)

    cursorDirNorm = np.linalg.norm(movement)
    if cursorDirNorm == 0: return 0
    cursorDir = movement / cursorDirNorm
    

    angle = np.arccos(np.clip(np.dot(correctDir,cursorDir),-1,1)) # find angle between two unit vector: pi = opposite dir, 0 = same dir
    rewardShaping = np.pi/2 - angle
    rewardShaping *= factor
  
    return rewardShaping
  
  
  def _getExpAngleReward(self,target_pos,decoded_pos,action):
    """ 
    Exponential Angle Reward Shaping 
    generate reward shaping based on whether cursor moved toward correct direction or not (depends on 1/dist from target)
    Every step returns reward between 1 to -1
    distance between 
    """
    if np.sum(np.isnan(target_pos)) != 0: return 0 # target is nan (still state) #TODO: need to get rid of still state when training

    sigma = 0.001 # to prevent division by zero
    factor = 0.1

    # get linAngleReward
    linAngleReward = self._getLinAngleReward(target_pos,decoded_pos,action)

    # exp distance factor
    dist = math.dist(target_pos,decoded_pos)
    rewardShaping = linAngleReward * factor/(dist+sigma)
  
    return rewardShaping
  
  def _getLinAngleTravelledReward(self,target_pos,decoded_pos,action):
    """ 
    Lin angle reward * distance travelled
    generate reward shaping based on whether cursor moved toward correct direction or not (but also depends on distance travelled)
    Every step returns reward roughly between 1 to -1
    assuming max speed is 1 (it isn't; it's actually rad(2) because action can currently be [vx=1,vy=1]), the max reward that can be gained is 1 
    true max reward gained is rad(2) (when cursor travelles diagonally at speed vx=1, vy=1)
    """

    # get linAngleReward
    linAngleReward = self._getLinAngleReward(target_pos,decoded_pos,action)

    # multiply it by how much cursor travelled in that direction
    travelled = np.linalg.norm(action[:2])
    rewardShaping = linAngleReward * travelled
  
    return rewardShaping
  
  def _getMonotonicDistanceReward(self,target_pos,decoded_pos,action):
     
     # if target is still, return 0. TODO: still target should be gone. is this messing it up?
     if np.sum(np.isnan(target_pos)) != 0: return 0

     time = self.t / self.activeT
     initDist = math.dist(target_pos,np.zeros(2))
     cursorDist = math.dist(target_pos,decoded_pos)
     scaledDist = (initDist-cursorDist) / initDist
     return time * scaledDist


  def _getGameStateReward(self,game_state):
    # get reward purely from game state (H,W,T,h,w,t,n)

    if game_state == 't': reward = self.REWARD_GAME_STATE['t'] # tolerated hit
    elif game_state.islower(): reward = self.REWARD_BASE
    elif game_state == 'T': reward = self.REWARD_GAME_STATE['T'] # time out
    elif game_state == 'H': reward = self.REWARD_GAME_STATE['H'] # hit
    elif game_state == 'W': reward = self.REWARD_GAME_STATE['W'] # wrong hit
    else: 
      print("unrecognized game state:",game_state)
      exit(1)

    return reward



  def isCursorInsideTarget(self,decoded_pos,target_pos,target_size):
    return np.sum(abs(decoded_pos-target_pos) <= target_size/2) == 2

  def _getBonusSubtractedHold(self,decoded_pos,target_pos,target_size,action):
    """ 
     additional linear award where agent gets most award for being at dead center, and little less for being around edge
     1 = dead center
     0 = at edge
    """
    if self.weight_SUBTRACTED_HOLD != 0:
      if self.isCursorInsideTarget(decoded_pos,target_pos,target_size):
        reward = 1 - math.dist(target_pos,decoded_pos) / np.linalg.norm(target_size/2)
        return reward
    return 0
      
  def _getBonusStallInTarget(self,decoded_pos,target_pos,target_size,action):
    """ 
    additional award for having small magnitude for action 
    1 max reward (staying still)
    0 min reward (moving at max speed)
    """
    if self.weight_STALL_IN_TARGET != 0:
      if self.isCursorInsideTarget(decoded_pos,target_pos,target_size):
        reward = 1 - np.linalg.norm(action[:2]) / np.linalg.norm([1,1])
        return reward
    return 0

  def _getBonusCloserToCenter(self,decoded_pos,target_pos,target_size,action):
    """
    exponential reward for coming closer to the center
    """
    sigma = 0.001
    factor = 1

    if self.weight_CLOSER_TO_CENTER != 0:
      if self.isCursorInsideTarget(decoded_pos,target_pos,target_size):
        dist = math.dist(target_pos,decoded_pos) * factor # having *10 makes the reward x10. having nothing makes the reward x100
        reward = 1/(dist+sigma)
        # dist = max(math.dist(target_pos,decoded_pos),0.0001) * 10 # voided after May 15 2023 # expAngle
        # dist = min(math.dist(target_pos,decoded_pos),0.01) # voided after May 15 2023 # LinDist
        return reward
    return 0

  def _getBonusCopilotContribution(self,decoded_pos,target_pos,target_size,action):
    """
    reward given for soley copilot's contribution
    project action to desired direction. award the magntiude amount
    """

    desired = target_pos - decoded_pos 
    mag = np.dot(action[:2],desired)

    return mag







  """ legacy code can be called by setting yaml with legacy flag """

  def getReward__legacy(self,result,action, obs=None):
    
    decoded_pos   = result[0]
    target_pos    = result[1]
    target_size   = result[2]
    game_state    = result[4]
    
    # base reward
    reward = self._getGameStateReward(game_state)

    # bonus reward 
    bonusReward, discard = self._getInsideTargetBonusReward__legacy(decoded_pos,target_pos,target_size,action)
    if discard: return bonusReward
    reward += bonusReward
    
    # shaping reward
    reward += self.getShapingReward(target_pos,decoded_pos,action)
    
    return reward
  
  def _getInsideTargetBonusReward__legacy(self,decoded_pos,target_pos,target_size,action):
    """ 
    This is how I used to do things. very illogical (look at reward = for instance)
    This is kept here as legacy in case comparison is needed. but we won't use this in the future
    if the cursor is inside, target it may merit additional reward 
    return 
        reward: additional reward
        discard (T/F): 
            True = return only use this bonus reward (discard all other reward including gamestate reward)
            False = add this bonus reward to gamestate (base) reward
    """

    # if it is inside target, don't deduct point! give it point to encourage staying (more point if get closer to reward)
    if np.sum(abs(decoded_pos-target_pos) <= target_size/2) == 2: 
          
      reward = 0
      
      # simple award where agent gets most award for being at dead center, and little less for being around edge
      if self.weight_SUBTRACTED_HOLD != 0:
        reward = self.BONUS_SUBTRACTED_HOLD - math.dist(target_pos,decoded_pos)
        return reward, True
      
      # additional award for having small magnitude for action
      if self.weight_STALL_IN_TARGET != 0:
        mag = 1 - np.linalg.norm(action[:2]) / np.linalg.norm([1,1])
        reward = self.BONUS_STALL_IN_TARGET * mag
  
      # exponential reward for coming closer to the center
      if self.weight_CLOSER_TO_CENTER != 0:
        dist = max(math.dist(target_pos,decoded_pos),self.BONUS_CLOSER_TO_CENTER_CLIP) * self.BONUS_CLOSER_TO_CENTER # having *10 makes the reward x10. having nothing makes the reward x100
        reward += 1/dist
        # dist = max(math.dist(target_pos,decoded_pos),0.0001) * 10 # voided after May 15 2023 # expAngle
        # dist = min(math.dist(target_pos,decoded_pos),0.01) # voided after May 15 2023 # LinDist
    

      return reward, True
    
    return 0, False


  def _getLinDistReward__legacy(self,target_pos,decoded_pos,action):
    """ Linear Distance Reward Shaping """

    if np.sum(np.isnan(target_pos)) == 0:
      # RewardShaping: if target is nan don't worry about it
      distCurrent = math.dist(target_pos,decoded_pos)
      distPast = math.dist(target_pos,self.prev_pos)
      rewardShaping = (distPast-distCurrent) * 100
      self.prev_pos = np.copy(decoded_pos)
  
      return rewardShaping
  
    return 0
  
  def _getExpDistReward__legacy(self,target_pos,decoded_pos,action):
    """ Exponential Distance Reward Shaping """

    if np.sum(np.isnan(target_pos)) == 0:
      # RewardShaping: if target is nan don't worry about it
      distCurrent = math.dist(target_pos,decoded_pos)
      distPast = math.dist(target_pos,self.prev_pos)
      if distCurrent < 0.001 or distPast < 0.001: # avoid divide by zero
        rewardShaping = 0
      else:
        rewardShaping = (1/distCurrent - 1/distPast) * 100
        rewardShaping = np.clip(rewardShaping,-np.inf, 10000)
      self.prev_pos = np.copy(decoded_pos)
  
      return rewardShaping
  
    return 0


  def _getLinAngleReward__legacy(self,target_pos,decoded_pos,action):
    """ 
    Linear Angle Reward Shaping:
    generate reward shaping based on whether cursor moved toward correct direction or not
    """

    correctDir = (target_pos-decoded_pos)
    correctDir = correctDir / np.linalg.norm(correctDir)
    cursorDir = action[:2] / np.linalg.norm(action[:2])
    angle = np.arccos(np.clip(np.dot(correctDir,cursorDir),-1,1)) # find angle between two unit vector: pi = opposite dir, 0 = same dir
    rewardShaping = np.pi/2 - angle
  
    return rewardShaping
  
  
  def _getExpAngleReward__legacy(self,target_pos,decoded_pos,action):
    """ 
    Exponential Angle Reward Shaping 
    generate reward shaping based on whether cursor moved toward correct direction or not (depends on 1/dist from target)
    """
    
    correctDir = (target_pos-decoded_pos)
    correctDir = correctDir / np.linalg.norm(correctDir)
    cursorDir = action[:2] / np.linalg.norm(action[:2])
    angle = np.arccos(np.clip(np.dot(correctDir,cursorDir),-1,1)) # find angle between two unit vector: pi = opposite dir, 0 = same dir
    dist = np.clip(math.dist(target_pos,decoded_pos),0.0001,np.inf)
    rewardShaping = (np.pi/2 - angle) * 1/dist
  
    # print(reward)
    return rewardShaping
  

  """ DEV AREA. EXPERIMENTAL NOT PUBLISHABLE """


  def getCustomReward(self,result, action, obs):
        
    # result = [cursorPos, targetPos, targetSize, state_taskidx, self.game_state, self.detail]
    # action
    # prev obs

    decoded_pos   = result[0]
    # target_pos    = result[1]
    # target_size   = result[2]
    # game_state    = result[4]

    # targetReward
    reward = 0
    for name, (target_pos,target_size) in self.env.taskGame.targetsInfo.items():
      if np.sum(abs(decoded_pos-target_pos) <= target_size/2) == 2: 
        # cursorMovement = (decoded_pos - self.prev_pos) / self.env.taskGame.cursorVel[0]
        # correctMovement = target_pos - self.prev_pos
        # correctMovement /= np.linalg.norm(correctMovement)
        # targetReward = (1-spatial.distance.cosine(correctMovement,cursorMovement)) * np.linalg.norm(cursorMovement)
        # reward += targetReward * 10

        d = np.linalg.norm(decoded_pos-target_pos)
        max_d = np.linalg.norm(target_size/2)

        targetReward = (max_d-d)/max_d * 10 + 1
        reward += targetReward
        # print('target',name,targetReward)
        # reward += (action[2])*-10
        break
        
    else:
      # softmax reward
      L,R,U,D,S = obs[2:7]
      softmaxMovement = (R-L,U-D)
      cursorMovement = (decoded_pos - self.prev_pos) / self.env.taskGame.cursorVel[0]
      softmaxMovement = (R-L,U-D)
      softmaxReward = 1 - np.linalg.norm(cursorMovement-softmaxMovement)
      softmaxReward = np.clip(softmaxReward,0,1)
      reward += softmaxReward
      # print('softmax',reward)
    
    
    self.prev_pos = np.copy(decoded_pos)
    return reward
  
  def customReset(self):
    self.prev_pos = np.zeros(2)
    self.t = 0