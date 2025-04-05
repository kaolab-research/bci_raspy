
from modules.SJutil.Heatmap import Heatmap
from modules.SJutil.TaskTargetPredictor import TaskTargetPredictorClass
from modules.SJutil.DataStructure import CircularQueue
from pathlib import Path
import numpy as np
import math
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from modules.SJutil.TaskCopilotActionType.VXY import ActionVXY
from modules.SJutil.TaskCopilotActionType.FXY import ActionFXY
from modules.SJutil.TaskCopilotActionType.MXY import ActionMXY
from modules.SJutil.TaskCopilotActionType.CTarget import ActionCTarget


"""
handles loading copilot
needs to keep information about what observation is used
"""

class TaskCopilotAction:
    """
        all things copilot action related goes here
        such as interpreting copilot's action to create update_cursor
    """

    def __init__(self, copilotYamlParam):
        """ 
            read copilot yaml to determine action dimension 
            save these values in 
            self.copilotActionDim
        """
        # variable to be read by outside every timestep (for coloration)
        self.copilotDominatesCursorUpdate = False
        self.useMass = False
        
        self.initCopilotParamYaml(copilotYamlParam)
        self.reset_each_trial()

    def initCopilotParamYaml(self,copilotYamlParam):

        # if it is yaml extract the parameter
        if type(copilotYamlParam)==str:
            try: copilotYamlParamFile = open(copilotYamlParam + ".yaml", 'r')
            except: copilotYamlParamFile = open(Path(copilotYamlParam).parent / "model.yaml", 'r')    
            copilotYamlParam = yaml.load(copilotYamlParamFile, Loader=Loader)
            copilotYamlParamFile.close()
        
        self.copilotYamlParam = copilotYamlParam

        # action set it up
        self.n_copilotActionDim = len(copilotYamlParam["action_dim"]["action"])
        self.copilotActionTypes = copilotYamlParam["action_dim"]["action"]

        # alpha type
        if copilotYamlParam["action_dim"]["alpha"] is None:
            self.copilot_binary_alpha = False
            self.copilot_default_alpha = None
            self.copilot_alpha = None
        elif copilotYamlParam["action_dim"]["alpha"] == 'binary':
            self.copilot_binary_alpha = True
            self.copilot_default_alpha = None
            self.copilot_alpha = None
        else:
            self.copilot_default_alpha = copilotYamlParam["action_dim"]["alpha"] * 2 - 1
            self.copilot_alpha = self.copilot_default_alpha
            self.copilot_binary_alpha = False

        # click type:
        self.noHoldTimeByCopilot = 'click' in self.copilotActionTypes # this means copilot will not use holdtime (will use click instead)

        # action type
        if self.copilotActionTypes == ["vx", "vy", "alpha"]:
            self.ActionType = ActionVXY(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action
        elif self.copilotActionTypes == ["vx", "vy", "alpha", "click"]:
            self.ActionType = ActionVXY(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action
        elif self.copilotActionTypes == ["fx", "fy"]:
            self.ActionType = ActionFXY(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action
        elif self.copilotActionTypes == ["fx", "fy", "click"]:
            self.ActionType = ActionFXY(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action
        elif self.copilotActionTypes[:2] == ["mx", "my"]:
            self.ActionType = ActionMXY(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action
            self.useMass = self.copilotDominatesCursorUpdate = True
        elif self.copilotActionTypes[:2] == ["smx", "smy"]: # mass and softmax both has effect
            self.ActionType = ActionMXY(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action
            self.useMass = self.copilotDominatesCursorUpdate = True
        elif self.copilotActionTypes == ["chargeTargets"]:
            self.ActionType = ActionCTarget(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action
            self.useMass = self.copilotDominatesCursorUpdate = True
        elif self.copilotActionTypes == ["chargeTargets_kf_eff"]:
            self.ActionType = ActionCTarget(copilotYamlParam, self.copilotActionTypes)
            self.interpret = self.ActionType.copilot_action_kf_eff
            self.useMass = self.copilotDominatesCursorUpdate = True
        else:
            raise Exception("Undefined copilot action type")

    def reset_each_trial(self):
        """ reset any state variable that lingers. must be reset every trial """
        self.ActionType.reset_each_trial()
    
    @property
    def massPos(self): return self.ActionType.massPos if self.useMass else np.zeros(2)
    @property
    def saved_copilot_output(self): return self.ActionType.saved_copilot_output
    @property
    def copilotInControl(self): return self.ActionType.copilotInControl

    def updateTargetPos(self,targetsPos):
        """ to be called by outside this class to update targets pos. targetsPos is expected to be dictionary from SJ-4-directions """
        actionTargetPos = [pos for pos in list(targetsPos.values()) if isinstance(pos, np.ndarray) ]
        actionTargetPos.sort(key=lambda item: (item[0],item[1]))
        self.ActionType.targetsPos = actionTargetPos
        
        if self.copilotActionTypes == ["chargeTargets"]: 
            self.n_copilotActionDim = len(self.ActionType.targetsPos)
            




class TaskCopilotObservation:
    def __init__(self,copilotYamlParam):
        """ 
            read copilot yaml to determine obs dimension
            save these values in 
            self.copilotActionDim
        """

        # variables (set by tasks's save_to_obseration) and read by copilot
        self.softmax = None
        self.cursorPos = None
        self.correctTargetPos = None
        self.targetCursorOnPos = None
        self.massPos = None
        self.game_state = None
        self.detail = None
        self.timeRemain = None
        self.emptyData = None # whether I need to empty data or not
        self.holdTime = None
        self.holdTimeThres = None
        self.trajectoryIndex = [] # does not include present trajectory
        self.trajectoryValue = []

        # obs used (set once when initialized and treated as constants: True/False)
        self.useVelReplaceSoftmax = False
        self.useCardinalVelocity = False
        self.cardinalVelocityThres = 0
        self.useNoSoftmax = False
        self.useHold = False
        self.useHoldOn = False
        self.useVel = False
        self.useAcc = False
        self.useTime = False
        self.useTargetTime = False
        self.useTargetCenter = False
        self.useTrialEnd = False
        self.resetPosToLast = False # reset history so that it sets all history as trial's last seen cursor pos (for center out back)
        
        # constant defintion
        self.decodedVel = {0:np.array([-1, 0]),   #   0 : left,
                            1:np.array([ 1, 0]),   #   1 : right,
                            2:np.array([ 0, 1]),   #   2 : up,
                            3:np.array([ 0,-1]),   #   3 : down,
                            4:np.array([ 0, 0]),   #   4 : still,
                            }  #   5 : rest

        self.initCopilotParamYaml(copilotYamlParam)
    
    def initCopilotParamYaml(self,copilotYamlParam):

        """
        save which obs needs to be used and saved in this game
        """

        # copilotYamlParam = {
        #     "obs_dim":{
        #         "obs":[],
        #         "target_predictor":None,
        #         "target_predictor_input":"softmax",
        #         "cursor_target_obs": False, # set True if you only want 4 dim which is [cursor(2),target(2)]
        #         "history":[1,0], #, nargs='+', help="[n, interval] n history with t interval ex) -history 3 100 looks at 3 history (including current) with 100 timestep interval")
        #         "maskSoftmax": None, # type=str, help='all or partial or None. all means removing all softmax history')
        #     }
        # }
        
        # if it is yaml extract the parameter
        if type(copilotYamlParam)==str:
            try: copilotYamlParamFile = open(copilotYamlParam + ".yaml", 'r')
            except: copilotYamlParamFile = open(Path(copilotYamlParam).parent / "model.yaml", 'r') 
            copilotYamlParam = yaml.load(copilotYamlParamFile, Loader=Loader)
            copilotYamlParamFile.close()

        self.copilotObsDim = copilotYamlParam["obs_dim"]
        self.copilotInfo = copilotYamlParam["copilot"] if "copilot" in copilotYamlParam else None

        # if copilot uses target predictor set it up
        targetPredictorName = self.copilotObsDim["target_predictor"]
        if targetPredictorName is not None:
            self.TaskTargetPredictorClass = TaskTargetPredictorClass(targetPredictorName)
        else:
            self.TaskTargetPredictorClass = None
        
        # determine obs dimension size (n_copilotObsDim)
        N_STATE = 5
        N_POS_DIM = 2
        if self.copilotObsDim.get('noSoftmax',False): self.n_copilotObsDim = N_POS_DIM
        elif self.copilotObsDim.get('velReplaceSoftmax',False): self.n_copilotObsDim = N_POS_DIM + 2
        else: self.n_copilotObsDim = N_POS_DIM + N_STATE
        if 'hold' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 1
        if 'holdOn' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 1
        if 'vel' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 2
        if 'acc' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 2
        if 'time' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 1
        if 'targetTime' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 1
        if 'targetCenter' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 2
        if 'trialEnd' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 1
        if 'mass' in self.copilotObsDim["obs"]: self.n_copilotObsDim += 2

        # set 
        self.useNoSoftmax = self.copilotObsDim.get('noSoftmax',False)
        self.useVelReplaceSoftmax = self.copilotObsDim.get('velReplaceSoftmax',False)
        self.useCardinalVelocity = 'cardinalVelocityThres' in self.copilotObsDim
        if self.useCardinalVelocity: self.cardinalVelocityThres = self.copilotObsDim.get('cardinalVelocityThres')
        self.useHold = 'hold' in self.copilotObsDim["obs"]
        self.useHoldOn = 'holdOn' in self.copilotObsDim["obs"]
        self.useVel = 'vel' in self.copilotObsDim["obs"]
        self.useAcc = 'acc' in self.copilotObsDim["obs"]
        self.useTime = 'time' in self.copilotObsDim["obs"]
        self.useTargetTime = 'targetTime' in self.copilotObsDim["obs"]
        self.useTargetCenter = 'targetCenter' in self.copilotObsDim["obs"]
        self.useTrialEnd = 'trialEnd' in self.copilotObsDim["obs"]
        self.useMassPos = 'mass' in self.copilotObsDim["obs"]
        self.resetPosToLast = self.copilotObsDim.get('historyReset','zero') == 'last' # two option, last or zero

        # history dimension [n,interval]. 1 means no history, just current
        if targetPredictorName is not None: self.n_copilotObsDim += 2
        if self.copilotObsDim["cursor_target_obs"]: self.n_copilotObsDim = 4
        if self.copilotObsDim["cursor_target_obs"] and targetPredictorName is None:
            print("ERROR: If you set cursor_target_obs flag, you must set useTargetPredictor flag as well!")
            exit(1)

        # if copilot uses history set it up: newest stuff + old history
        self.historyDim = self.copilotObsDim["history"]
        if len(self.historyDim) == 2: 
            historyObsDim = self.n_copilotObsDim
            self.trajectoryIndex = self.n_copilotObsDim + np.array([(i*historyObsDim,i*historyObsDim+1) for i in range(self.historyDim[0]-1)]).flatten()
        else: 
            historyObsDim = 0
            if 'pos' in self.historyDim: 
                historyObsDim += 2
                self.trajectoryIndex = self.n_copilotObsDim + np.arange((2 * (self.historyDim[0]-1)))

                

        self.copilotHistoryCircularQueue = CircularQueue(self.historyDim,historyObsDim)
        self.n_copilotObsDim = self.n_copilotObsDim + (historyObsDim * (self.historyDim[0]-1))
        
        # if heat map set it up
        if "heatmap" in self.copilotObsDim and self.copilotObsDim["heatmap"] is not None:
            n = self.copilotObsDim["heatmap"]
            heatmapOption = self.copilotObsDim.get("obs_heatmap_option",[])
            self.heatmap = Heatmap(n, heatmapOption)
            self.n_copilotObsDim += self.heatmap.copilot_dim
        else:
            self.copilotObsDim["heatmap"] = None
            self.heatmap = None


    def get_env_obs(self, action=None):
        """
        based on intialization, select observation data to create obs
        observation data is freshly updated at the end of each timestep
        action = None | "mask" | "reset"
        """
        # self.copilotObsDim

        softmax = self.softmax
        cursorPos = self.cursorPos
        correctTargetPos = self.correctTargetPos
        targetCursorOnPos = self.targetCursorOnPos
        massPos = self.massPos
        game_state = self.game_state
        detail = self.detail
        timeRemain = self.timeRemain
        needReset = self.emptyData or (action=="reset")
        holdTime = self.holdTime
        holdtimeThres = self.holdTimeThres
        try:
            targetTime = (-1 if holdTime is None else holdTime / holdtimeThres) # -1 if not on target, holdtime normalized if on target
        except ZeroDivisionError:
            targetTime = (-1 if holdTime is None else 1)
        etc = None # filled if there is etc to be returned to SJtask

        
        obs = [cursorPos,softmax]
        if self.useVelReplaceSoftmax: 
            softmaxVel = self.softmax_to_vel(softmax)
            obs = [cursorPos,softmaxVel]
        if self.useNoSoftmax:
            obs = [cursorPos]
        if self.useHold: obs.append([int(game_state=='h' or game_state=='H' or game_state=='w' or game_state=='W')])
        if self.useHoldOn: obs.append([self.holdOn(cursorPos)]) # if it is on a target(regardless of game_state)
        if self.useVel: obs.append(detail['vel'])
        if self.useAcc: obs.append(detail['acc'])
        if self.useTime: obs.append([timeRemain])
        if self.useTargetTime: obs.append([targetTime])
        if self.useTargetCenter: obs.append(targetCursorOnPos)
        if self.useTrialEnd: obs.append([game_state.isupper()])
        if self.useMassPos: obs.append(massPos)
        obs = np.concatenate(obs)

        # predict target pos if it is used
        if self.copilotObsDim["target_predictor"] is not None:
            pred_target_pos = self.TaskTargetPredictorClass.predict(softmax, cursorPos, correctTargetPos, needReset)

            etc = ["pred_target", pred_target_pos]
            obs = np.concatenate([obs,pred_target_pos])
            if self.copilotObsDim["cursor_target_obs"]: obs = np.concatenate([cursorPos,pred_target_pos])

        # masksoftmax and history
        if needReset:
            if self.resetPosToLast: self.copilotHistoryCircularQueue.reset(cursorPos) # set history to have pos as last pos seen in trial (For center out back)
            else: self.copilotHistoryCircularQueue.reset()
            # self.copilotHistoryCircularQueue.reset()
        elif action == "mask":
            if self.copilotObsDim["maskSoftmax"] is not None: # partial
                obs[2:7] = 0
                if self.copilotObsDim["maskSoftmax"] == "all":
                    self.copilotHistoryCircularQueue.resetAllSoftmax()

        # get history
        if len(self.historyDim) == 2: 
            history_input = obs
        else: 
            history_input = []
            if 'pos' in self.historyDim: history_input += obs[:2].tolist()

        history_obs = self.copilotHistoryCircularQueue.add_get(history_input).flatten()
        # history obs has order of current, past1, past2...pastn
        # print(history_obs[len(history_input):])

        # append newest obs with history
        obs = np.concatenate([obs,history_obs[len(history_input):]])
        
        # add heat map
        if self.heatmap is not None:
            if self.heatmap.useCom:
                obs = np.concatenate([obs,self.heatmap.com])
            if self.heatmap.useBcc:
                obs = np.concatenate([obs,[self.heatmap.bcc]])
            if not self.heatmap.useOnly:
                obs = np.concatenate([obs,self.heatmap.map.flatten()])

        obs = np.float32(obs) # box is set to be float32 in env

        # save trajectory
        self.trajectoryValue = obs[self.trajectoryIndex.tolist()]
        
        # return observation
        return obs, etc
    


    def softmax_to_vel(self,softmax):
        """ return velocity generated from softmax"""
        if self.useCardinalVelocity:
            if max(softmax) > self.cardinalVelocityThres: # 0.8
                directionIdx = np.argmax(softmax)
                direction = self.decodedVel[directionIdx]
            else:
                direction = self.decodedVel[4]
            return np.copy(direction)
        
        else:
            vx = softmax[1]-softmax[0]
            vy = softmax[2]-softmax[3]
            vel = np.array([vx,vy])
            vel = np.clip(vel,-1,1)
            return vel
    
    def updateTargetsInfo(self,targetsInfo):
        """ to be called by outside this class to update targets pos. targetsPos is expected to be dictionary from SJ-4-directions """
        self.targetsTopLeft = []
        self.targetsBottomRight = []
        for i,v in targetsInfo.items():
            if type(v) is str: 
                continue
            else: 
                self.targetsTopLeft.append(v[0]-v[1]/2)
                self.targetsBottomRight.append(v[0]+v[1]/2)
        self.targetsTopLeft = np.array(self.targetsTopLeft)
        self.targetsBottomRight = np.array(self.targetsBottomRight)

    def holdOn(self,cursorPos):
        withinLeftBound = self.targetsTopLeft <= cursorPos
        withinRightBound = self.targetsBottomRight >= cursorPos
        withinBound = withinLeftBound * withinRightBound
        withinBound = withinBound.sum(1) == 2
        withinTarget = sum(withinBound)
        return withinTarget
    