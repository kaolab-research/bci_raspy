
# in:
#   - state_task
#   - decoder_output
#   - target_hit
# out:
#   - decoded_pos
#   - target_hit
#   - scores


import pygame
import pygame.gfxdraw
import numpy as np
import time
import random
import yaml
import torch
import math
import queue
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
# import tkinter as tk
from modules.SJutil.DataStructure import CircularQueue
from modules.SJutil.Heatmap import Heatmap
from modules.SJutil.Assist import AssistClass
from modules.SJutil.Gamify import GamifyClass
from modules.SJutil.SyntheticSoftmax import SyntheticSoftmax
from modules.SJutil.TaskTargetPredictor import TaskTargetPredictorClass
from modules.SJutil.TaskCopilot import TaskCopilotAction, TaskCopilotObservation
from modules.SJutil.TaskTools import TaskTools
from modules.SJutil.PeformanceRecord import PerformanceRecord
import pyautogui # to prpoperly use this, goto system preference, security, accessibility, check terminal
from scipy import ndimage
from scipy.signal import resample
import scipy.special
import pathlib # used in destructor
import os 

class TimeProfiler():
    def __init__(self):
        self._t = time.time()
        return
    def profile(self):
        new_t = time.time()
        dt = new_t - self._t
        self._t = new_t
        return dt


class SJ_4_directions:

    def __init__(self,params=None,render=True,useRealTime=True,showAllTarget=False,showSoftmax=False,showVelocity=False,randomInitCursorPosition=False,copilotYamlParam=None,showHeatmap=False,showNonCopilotCursor=False,replayData=None,gamify=False,gamifyOption={},softmaxStyle='cross',hideMass=True,taskTools=[]):
        
        self.params = params
        __slots__ = ('render', 'useRealTime', 'sessionLength', 'screenSize', 'objScale', 'cursorRadius', 'styleCursorRadius', 'cursorVel', 'showAllTarget', 'ignoreWrongTarget', 'skipFirstNtrials', 'useRandomTargetPos', 'randomTargetPosRadius', 'centerIn', 'resetCursorPos', 'targetWrongColor', 'targetDesiredColor', 'targetHoldColor', 'targetBorderL', 'targetDesiredBorderL', 'defaultCursorColor', 'secondCursorColor', 'cursorColor', 'cursorColors', 'targetsInfo', 'nBit', 'defaultTargetSize', 'desiredTargetList', 'targetsPos', 'decodedVel', 'target2state_task', 'state_task2target', 'holdTimeThres', 'dwellTimeThres', 'graceTimeThres', 'softmaxThres', 'assistValue', 'assistMode', 'inactiveLength', 'delayedLength', 'activeLength', 'showTextMetrics', 'showCountDown', 'showSoftmax', 'showHeatmap', 'showColorCursor', 'showNonCopilotCursor', 'randomInitCursorPosition', 'styleChange', 'styleChangeBallSize', 'styleChangeCursorSize', 'softmaxStyle', 'useCmBoard', 'cmBoardDetail', 'hideAllTargets', 'gamify', 'fullScreen', 'wrongTolerance', 'episodeLength', 'tickTimer', 'tickLength', 'inactiveTickLength', 'delayedTickLength', 'activeTickLength', 'episodeTickLength', 'graceTickLength', 'etc', 'heatmap', 'AssistClass', 'GamifyClass', 'game_state', 'TaskCopilotAction', 'TaskCopilotObservation', 'copilotInfo', 'copilotObsDim', 'n_copilotObsDim', 'copilot_default_alpha', 'copilot_alpha', 'noHoldTimeByCopilot', 'copilotActionTypes', 'noCopilot', 'hasCopilot', 'cursorPos', 'mousePos', 'secondCursorPos', 'copilotCursorPos', 'detail', 'heldTimeBegin', 'startTime', 'hitRate', 'missRate', 'reset', 'nextTargetBucket', 'pygameKeyPressed', 'K_a_pressed', 'key_pressed', 'target', 'pastCursorPos', 'pastCursorVel', 'pastTargetPos', 'pastTargetSize', 'randomTargetPos', 'trialCount', 'generatedImage', 'cumulativeActiveTime', 'useSmoothVelocity', 'useClickState', 'useNonLinearVel', 'copilotClicked', 'manualClick', 'wrongToleranceCounter', 'variable4CursorColor', 'PRESSED_ONCE', 'L', 'windowSize', 'windowPadding', 'cursorBias', 'cursorMagnitude', 'targetsImgPos', 'stopSize', 'stopImgSize', 'stopImgPos', 'fontSize', 'fontImgSize', 'textImgPos', 'barPos', 'softmaxBarSize', 'softmaxBarImgSize', 'softmaxBarImgSizei', 'softmaxBarImgPosC', 'softmaxBarImgPosA', 'softmaxBarImgPosB', 'softmaxBarImgPosD', 'softmaxBarImgPosE', 'softmaxBarColor', 'softmaxCrossImgPosL', 'softmaxCrossImgPosR', 'softmaxCrossImgPosU', 'softmaxCrossImgPosD', 'softmaxCrossImgPosS', 'nextTarget', 'currCursorVel', '__module__', '__init__', 'init_variables', 'update_window_size_constants', 'insertRandomTargetPos', 'initParamWithYaml', 'getScreenInfo', 'init_copilot_param_with_yaml', 'initPygameImage', 'update', 'replayTimestep', 'fastForwardSpeed', 'replay', 'check_pygame_event', 'reset_and_determine_target', 'resetYamlParam', 'resetMousePos', 'update_cursor', 'skip_cursor_movement', 'smooth_velocity', 'decode_direction', 'decoder_direction', 'copilot_direction', 'nonlinear_velocity', 'hit_target', 'determine_game_state', 'determine_reset', 'determine_cursor_color', 'draw_state', 'determine_taskidx', 'create_output', 'addTargets', 'useTargetYamlFile', 'getCurrentCursorPos', 'getTruthPredictorTargeterPos', 'changeTargetSize', 'setAttributeFromCurriculum', 'get_env_obs', 'save_to_observation', 'add_etc', 'reset_etc', '__dict__', '__weakref__', '__doc__', '__repr__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__')

        self.yaml_data = None

        # replay parameters
        if replayData is not None: 
            self.replayData = replayData[0]
            self.trialRangeReplayData = replayData[1]
            self.eegReplayData = replayData[2]
            # decoder part
            self.hasReplayDecoder = self.eegReplayData is not None
            if self.hasReplayDecoder:
                self.eegData, self.eegDecoder, self.eegDownsampled_length, self.eegInput_length = replayData[2]
            self.replayTimestepMax = len(self.replayData["state_task"])
            print("NOTE: THIS IS A REPLAY")

        # tunable constants
        self.render = render # uses to render image or not
        self.useRealTime = useRealTime # uses time.time package for real time hold time and game ending signal
        # ^ if false uses tickLength for holdTime
        self.sessionLength = 300 # seconds
        self.syncLength = 10000000 # seconds
        self.waitIntegerBlocks = False
        self.numCompletedBlocks = -100 # NULL value
        self.enableKfSyncInternal = -1 # NULL value
        self.numInitialEvaluationBlocks = 0
        self.screenSize = np.array([700,700]) # w,h
        self.objScale = 1/1000
        self.squareScale = 1.0
        self.cursorRadius = 10
        self.styleCursorRadius = 30
        self.cursorVel = (0.015,0.015) # max is 1
        self.copilotVel = (0.02,0.02)
        self.driftCompensation = None
        self.showAllTarget = showAllTarget
        self.centerCorrectTarget = False
        self.hideCursor = False
        self.promptArrow = False
        self.drawFixationCross = False
        self.deltaPosScale = 1.0
        self.ignoreWrongTarget = False
        self.ignoreCorrectTarget = False
        self.skipFirstNtrials = 0
        self.cursorMoveInCorretDirectionOnly = False
        self.constrain1D = False
        self.restrictHalfPlane = False
        self.correctVsOthers = False
        self.useRandomTargetPos = False
        self.randomTargetPosRadius = -1
        self.randomTargetMinRadius = 0
        self.centerIn = False
        self.outTime = 0 # cumulative time to reach outer targets.
        self.enforceCenter = 'reset'
        self.resetNextCursorPos = False # used with self.enforceCenter
        self.resetCursorPos = True
        self.targetWrongColor = (0,255, 200)
        self.targetDesiredColor = (235,235,235)
        self.targetHoldColor = (255,205,0)
        self.targetBorderL = 1
        self.targetDesiredBorderL = 2
        self.defaultCursorColor = (200, 200, 200)
        self.secondCursorColor = (0,225,100) # (50, 50, 50)
        self.cursorColor = self.defaultCursorColor
        self.cursorColors = {}
        self.cursorColors['r'] = (255,0,0)
        self.cursorColors['o'] = (255,128,0)
        self.cursorColors['y'] = (255,255,0)
        self.cursorColors['g'] = (0,225,100)
        self.cursorColors['b'] = (0,0,255)
        self.cursorColors['i'] = (106,0,255)
        self.cursorColors['lr'] = (240,10,10)
        self.cursorColors['w'] = (255, 255, 255)
        self.targetsInfo = { # max is 1 [pos, size]
            'left' :[np.array([-0.85, 0  ]), np.array([0.2,0.2])],   #   0 : left,
            'right':[np.array([ 0.85, 0  ]), np.array([0.2,0.2])],   #   1 : right,
            'up'   :[np.array([ 0  , 0.85]), np.array([0.2,0.2])],   #   2 : up,
            'down' :[np.array([ 0  ,-0.85]), np.array([0.2,0.2])],   #   3 : down,
            'still':'still',
        }
        self.targetsWord = {}
        self.nBit = math.log(len(self.targetsInfo),2)
        self.defaultTargetSize = np.array([0.2,0.2]) # just choose one for now
        self.desiredTargetList = [k for k in self.targetsInfo]
        if self.centerIn: self.targetsInfo['center'] = [np.array([ 0  , 0]), self.defaultTargetSize] # center target information is needed
        self.targetsPos = {}
        self.useCircleTargets = False
        self.usePolygonTargets = False
        for k,v in self.targetsInfo.items():
            self.targetsPos[k] = v if type(v) is str else v[0]

        # other variables (decoder label to velocity)
        self.decodedVel = {0:np.array([-1, 0]),   #   0 : left,
                            1:np.array([ 1, 0]),   #   1 : right,
                            2:np.array([ 0, 1]),   #   2 : up,
                            3:np.array([ 0,-1]),   #   3 : down,
                            4:np.array([ 0, 0]),   #   4 : still,
                            5:np.array([ 0, 0]),}  #   5 : rest

        # targets converted to decoder's label for training step
        self.target2state_task = {
                               "left":  0,
                               "right": 1,
                               "up":    2,
                               "down":  3,
                               "still": 4,
                               None:    -1, # when there is no target (i.e stop)
                               }
        self.target2state_task["random"] = ord('r') # 114 means random
        self.target2state_task["center"] = ord('c') # center is special target. =99.
        self.state_task2target = {v:k for k,v in self.target2state_task.items()} # only used for replay
        self.pizza = None
        self.usePizza = False
        if self.pizza is not None:
            self.set_pizza_param(self.pizza)

        self.holdTimeThres = 0.5 # seconds # how long to hold until registered as click for correct target
        self.dwellTimeThres = self.holdTimeThres # seconds # how long to hold until registered as click for incorrect target
        self.graceTimeThres = 0.0 # seconds
        self.softmaxThres = 0.0
        self.assistValue = 0.0
        self.assistMode = 'e' # efficient vs natural

        self.inactiveLength = 2 # seconds
        self.delayedLength = 0 # seconds
        self.activeLength = 10 # seconds
        self.calibrationLength = self.activeLength
        self.showTextMetrics = None
        self.showCountDown = False
        self.controlSoftmax = False
        self.showSoftmax = showSoftmax
        self.showVelocity = showVelocity
        self.showColorSoftmax = False
        self.showStats = False
        self.showHeatmap = showHeatmap
        self.showColorCursor = True
        self.hideMass = hideMass
        self.hideCom = False
        self.showNonCopilotCursor = showNonCopilotCursor
        self.randomInitCursorPosition = randomInitCursorPosition
        self.styleChange = False
        self.styleChangeBallSize = 0.5
        self.styleChangeCursorSize = np.array([0.2, 0.2])
        self.generatedImage = {} # cached pygame image
        self.softmaxStyle = softmaxStyle
        self.useCmBoard = False
        self.overrideSyntheticSoftmax = False
        self.cmBoardDetail = {}
        self.hideAllTargets = False
        self.gamify = gamify
        self.fullScreen = False
        self.backgroundMoves = False
        self.useColorBackground = False
        self.wrongTolerance = 0 # zero tolerance means wrong is wrong. 1 tolerance is can make 1 mistake

        self.renderAngles = [0.0] # possible render angles (in degrees)
        self.render_angle = 0 # in degrees. Used to render the cursor and target at different angles.
        self.renderAnglesShuffled = {} # one list for each target. populated in reset_and_determine_target.

        # default kf parameters
        self.kfCopilotAlpha = 1 # 1=kf only 0=copilot only
        self.enableKfSync = False
        self.enableKfAdapt = False
        self.enableKfAdaptDelay = 1 # 1 seconds

        # dependent constants (should not be tuned!)
        self.episodeLength = self.inactiveLength + self.delayedLength + self.activeLength
        self.tickTimer = 0
        self.tickLength = 0.02 # each tick is assumed to be 20ms (value imported from timer.py)
        self.inactiveTickLength = self.inactiveLength / self.tickLength
        self.delayedTickLength = self.delayedLength / self.tickLength
        self.activeTickLength = self.activeLength / self.tickLength
        self.calibrationTickLength = self.calibrationLength / self.tickLength
        self.episodeTickLength = self.episodeLength / self.tickLength
        self.graceTickLength = self.graceTimeThres / self.tickLength
        self.KfSyncDelayTickLength = self.inactiveTickLength + self.delayedTickLength + self.enableKfAdaptDelay / self.tickLength
        self.etc = {}
        self.heatmap = None
        self.isReady = False # set to true if eeg starts to flow in, False if no eeg seen

        # other modules
        self.AssistClass = AssistClass(assistMode=self.assistMode,assistValue=self.assistValue,cursorSpeed=self.cursorVel[0],tickLength=self.tickLength)
        self.GamifyClass = GamifyClass(gamify)
        
        targetColor = {'wrong':self.targetWrongColor,'desired':self.targetDesiredColor,'hold':self.targetHoldColor}
        self.GamifyClass.initGamifyOption(gamifyOption,targetColor)
        self.targetWrongColor = targetColor['wrong']
        self.targetDesiredColor = targetColor['desired']
        self.targetHoldColor = targetColor['hold']
        
        self.overrideStateTask = False # Used to load state_task externally. Overrides random factor.

        # default value
        self.game_state = 'n' #stand for neutral

        # set copilot yaml
        if copilotYamlParam is not None:
            self.init_copilot_param_with_yaml(copilotYamlParam,hideMass)
            self.TaskTools = None if taskTools == [] else TaskTools(self.TaskCopilotObservation,self.TaskCopilotAction,tools=taskTools)
        self.noCopilot = copilotYamlParam is None
        self.hasCopilot = not self.noCopilot

        """ use yaml's param if there is any """
        if params is not None:
            self.initParamWithYaml(params)
        """        #################         """

        # init pygame
        if self.render:
            pygame.font.init()
            if self.fullScreen:
                self.screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
            else:    
                self.screen = pygame.display.set_mode(self.screenSize, pygame.RESIZABLE)

        # init variables
        self.init_variables()

        # set dependent variable constants
        self.update_window_size_constants()

        

    def init_variables(self):
        # this function runs once to initialize variable
        # maybe run again when 'r' key is pressed but otherwise, runs exactly once

        self.cursorPos = np.array([0., 0.])
        self.mousePos = np.array([0., 0.])
        self.secondCursorPos = np.array([0., 0.]) # to show non copilot cursor
        self.copilotCursorPos = np.array([0., 0.])
        self.detail = {'vel':np.array([0., 0.]), 'acc':np.array([0., 0.])} 
        self.heldTimeBegin = None
        self.startTime = time.time()
        self.activeTimeRecordingStart = None
        self.activeTimeRecordingEnd = None
        self.hitRate = 0
        self.missRate = 0
        self.timeoutRate = 0
        self.outHitRate = 0
        self.outMissRate = 0
        self.outTimeoutRate = 0
        self.reset = True
        self.nextTargetBucket = []
        self.pygameKeyPressed = None
        self.K_a_pressed = False
        self.key_pressed = {} # indicates if any key is pressed (stored as dictionary)
        self.target = None # initialy None
        self.nextTarget = None
        self.pastCursorPos = np.zeros(2)
        self.pastCursorVel = np.zeros(2)
        self.pastTargetPos = np.zeros(2)
        self.pastTargetSize = np.zeros(2)
        self.randomTargetPos = np.zeros(2)
        self.trialCount = 0
        self.cumulativeActiveTime = 0 # needed to calculate bitrate
        self.useSmoothVelocity = False
        self.useClickState = False # referring to automatic clicking after still for 10 consecutive timesteps
        self.useNonLinearVel = False
        self.copilotClicked = None
        self.manualClick = False # click recorded by user clicking with space bar
        self.noHoldTimeByCopilot = self.noHoldTimeByCopilot if hasattr(self,"noHoldTimeByCopilot") else False # referring to holdtime disabled for copilot that has click as action space
        self.wrongToleranceCounter = 0
        self.tempSynSoft = SyntheticSoftmax()
        self.PerformanceRecord = PerformanceRecord()
        self.stats = {
            'CorrectSoftmax':0, # counting everytime model had correct Softmax (in trial)
            'NonStillTTimer':0, # value used to calculate soft correct softmax (ignore still mistakes)
            'Streak':np.zeros(5,dtype=int), # number of chosen softmax in a row (in trial)
            'CorrectAction':np.zeros(5,dtype=int), # correct softmax rate in a trial (in trial)
            'TotalAction': np.zeros(5,dtype=int), # number of Still state in a trial
            'TrialTime':0,
        }
        self.allow_kf_sync = False # boolean controller for [allow_kf_adapt, allow_kf_sync] 
        self.allow_kf_adapt = False
        # self.consecutiveStillRequired = 15

        # these are states saved every tick for only the cursor color purposes
        self.variable4CursorColor = {'copilotControl':False,
                                     'usingNonLinearVel':False,
                                     'clicked':False,
                                     'delayedPhase': False
                                     } 
        self.PRESSED_ONCE = False # for denoisign multiclick for pygame key pressed. useful sometimes

        self.save_to_observation(
            softmax = np.ones(5)*0.2, 
            cursorPos=self.cursorPos, 
            holdTime = 0,
            target = None,
            targetHit = None,
        )

        # text for targets
        self.multiply_equation = ''
        self.subtract_equation = ''
        self.word_association = ''
        with open('./asset/text/random_word.txt', "r") as my_file:
            self.word_association_list = my_file.read().split()

        # welford drift average
        self.driftTotal = np.zeros(2)
        self.driftN = 1
        self.driftTotals = [] # list of totals (used to calculate driftTotal after all target reach has been done)
        self.driftNs = [] # list of Ns (used to calculate driftN)

    def update_window_size_constants(self,humanInduced=False):

        if self.useCmBoard:
            # use cmboard size and resize the pygame
            self.L = self.cmBoardDetail["gameBoardSize"][0] * self.screenInfo["width_cm2px"]
            
            if self.cmBoardDetail["fullSizeScreen"]:
                self.screenSize = np.array((self.screenInfo["width_px"],self.screenInfo["height_px"]))
            else:
                self.screenSize = (int(self.L / 0.9),)*2
                
            if humanInduced:
                self.screenSize = np.array(self.screen.get_size())
            else:
                pygame.display.set_mode(self.screenSize, pygame.RESIZABLE)
        
        else:
            # use human adjustable size
            if self.render: self.screenSize = np.array(self.screen.get_size())
            self.L = min(self.screenSize) * 0.9

        self.windowSize = np.array([self.L,self.L])
        windowPaddingX = (self.screenSize[0]-self.L)/2
        windowPaddingY = (self.screenSize[1]-self.L)/2
        self.windowPadding = np.array((windowPaddingX,windowPaddingY))
        self.statWindoPadding = np.array((windowPaddingX*1.1+self.L, windowPaddingY))
        if self.render: 
            self.window = pygame.Surface(self.windowSize)
            if self.backgroundMoves:
                self.windowDup = pygame.Surface(self.windowSize)
            if self.useColorBackground:
                self.generatedImage['colorBackground'] = self.getImage(f'background{self.useColorBackground}',self.windowSize*2)
        if self.render:
            self.renderStatWindow = self.showStats and (windowPaddingX > windowPaddingY * 5)
            if self.renderStatWindow:
                self.statWindow = pygame.Surface(np.array((windowPaddingX * 0.8,self.L)))

        # reset cursor constants
        self.cursorBias = self.windowSize / 2
        self.cursorMagnitude = self.windowSize / 2

        # name: [leftTop(x,y), bottomRight(x,y), size(w,h)]
        self.targetsImgPos = {} # depends on targetsPos / used to determine target
        for k,info in self.targetsInfo.items():
            if type(info) is not str: 
                pos, size = info
                realTargetSize = self.L * size / 2
                pos = np.array([pos[0],-pos[1]]) # need to flip y axis in image
                topLeft = self.cursorBias + self.cursorMagnitude * pos - realTargetSize / 2
                bottomRight = self.cursorBias + self.cursorMagnitude * pos + realTargetSize / 2
                self.targetsImgPos[k] = (topLeft,bottomRight,realTargetSize)

        # subtract position
        pos, size = np.array((0, -0.9)),np.array((0.2,0.2))
        realTargetSize = self.L * size / 2
        pos = np.array([pos[0],-pos[1]]) # need to flip y axis in image
        topLeft = self.cursorBias + self.cursorMagnitude * pos - realTargetSize / 2
        bottomRight = self.cursorBias + self.cursorMagnitude * pos + realTargetSize / 2
        self.subtractImgPos = (topLeft,bottomRight,realTargetSize)

        if self.styleChange:
            self.imgTargetsImgPos = {k:(TL+(size*(1-self.styleChangeBallSize)/2),BR-(size*(1-self.styleChangeBallSize)/2),size*self.styleChangeBallSize) for k,(TL,BR,size) in self.targetsImgPos.items()}
            self.imgDefaultTargetImgSize = self.L * self.defaultTargetSize / 2 * self.styleChangeBallSize # image of target, it's image size not actual size
            corrospondence = {
                "cursor":"greyball",
                "target":["greyball","greenball","blueball"],
                }
            #print(self.windowSize, self.imgTargetsImgPos)
            self.cursorImgSize = self.L * self.styleChangeCursorSize / 2
            self.pygameImage = self.initPygameImage(corrospondence,self.imgTargetsImgPos,self.imgDefaultTargetImgSize,self.cursorImgSize)

        # stop sign
        self.stopSize = np.array([200,200])
        self.stopImgSize = self.L * self.objScale * self.stopSize
        self.stopImgPos = self.windowSize / 2 - self.stopImgSize / 2

        # font size
        self.fontSize = 60
        self.fontImgSize = int(self.L * self.objScale * self.fontSize)
        statFontImgSize = int(windowPaddingX * self.objScale * 90)
        self.statImgPos = np.ones(2) * int(statFontImgSize * 0.3)
        self.statNewlineImgPos = np.array((0,statFontImgSize))
        if self.render: 
            self.pygameFont = pygame.font.SysFont(None, self.fontImgSize)
            self.statFont = pygame.font.SysFont(None, statFontImgSize)

        # action font size
        self.actionTextFontSize = 60 #48
        self.actionFontImgSize = int(self.L * self.objScale * self.actionTextFontSize)
        if self.render: self.pygameActionFont = pygame.font.SysFont(None, self.actionFontImgSize)


        textImgPosY = windowPaddingY-self.L * 0.05
        # if windowPaddingY-windowPaddingX <= 0: textImgPosY = 1

        self.textImgPos = (windowPaddingX,textImgPosY)

        # bar pos
        # barPosY = self.screenSize[1] - windowPaddingY + windowPaddingY / 3
        barPosY = self.windowSize[1] + windowPaddingY + self.L * 0.05 / 2
        barSizeX = self.windowSize[0]
        self.barPos = np.array((windowPaddingX,barPosY,barSizeX, self.L * 0.05 / 7))

        # softmax pos
        self.softmaxBarSize = np.array([60,400])
        softmaxBarPadding = np.array([25,400])
        self.softmaxBarImgSize = self.L * self.objScale * self.softmaxBarSize
        self.softmaxBarImgSizei = np.array((self.softmaxBarImgSize[1],self.softmaxBarImgSize[0]))
        softmaxBarImgsoftmaxBarPadding = self.L * self.objScale * softmaxBarPadding
        self.softmaxBarImgPosC = self.windowSize / 2 - self.softmaxBarImgSize / 2
        self.softmaxBarImgPosA = self.softmaxBarImgPosC - (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([2.0,0])
        self.softmaxBarImgPosB = self.softmaxBarImgPosC - (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([1.0,0])
        self.softmaxBarImgPosD = self.softmaxBarImgPosC + (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([1.0,0])
        self.softmaxBarImgPosE = self.softmaxBarImgPosC + (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([2.0,0])
        self.softmaxBarColor = np.array((40,40,40))
        self.softmaxBarArgColor = np.array((80,40,40))
        self.softmaxBarCorrColor = np.array((40,80,40))

        self.softmaxCrossImgPosL = self.windowSize / 2 - self.softmaxBarImgSizei * np.array((1,0.5)) - np.array((0.5,0)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosR = self.windowSize / 2 - self.softmaxBarImgSizei * np.array((0,0.5)) + np.array((0.5,0)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosU = self.windowSize / 2 - self.softmaxBarImgSize * np.array((0.5,1)) - np.array((0,0.5)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosD = self.windowSize / 2 - self.softmaxBarImgSize * np.array((0.5,0)) + np.array((0,0.5)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosS = self.windowSize / 2


        # heatmap pos & size
        if self.heatmap is not None:
            self.heatmap.update_window_size_constants(self.L)

        # modules affected
        self.GamifyClass.windowSettingChanged({'L':self.L,'windowPadding':self.windowPadding,'screenSize':self.screenSize})
            
        # at the very end, if copilot exists, update target pos for it
        if self.hasCopilot:
            self.TaskCopilotAction.updateTargetPos(self.targetsPos)
            self.TaskCopilotObservation.updateTargetsInfo(self.targetsInfo)

    def insertRandomTargetPos(self,pos,size):
        # put information in targetsInfo
        self.targetsInfo.update({"random" : [pos,size]})
        self.targetsPos.update({"random" : pos})

        # put information in targetsImgPos
        realTargetSize = self.L * size / 2
        pos = np.array([pos[0],-pos[1]]) # need to flip y axis in image
        topLeft = self.cursorBias + self.cursorMagnitude * pos - realTargetSize / 2
        bottomRight = self.cursorBias + self.cursorMagnitude * pos + realTargetSize / 2
        self.targetsImgPos = {"random": (topLeft,bottomRight, realTargetSize)}
        if self.styleChange:
            self.imgTargetsImgPos = {k:(TL+(size*(1-self.styleChangeBallSize)/2),BR-(size*(1-self.styleChangeBallSize)/2),size) for k,(TL,BR,size) in self.targetsImgPos.items()}
        self.pygameImage["random"] = self.pygameImage["default"]

    def initParamWithYaml(self,params):
        if 'sessionLength' in params:
            self.sessionLength = params['sessionLength'] 
        elif 'trialLength' in params:
            self.sessionLength = params['trialLength'] # legacy code
        self.syncLength = params.get('syncLength', 10000000)
        self.waitIntegerBlocks = params.get('waitIntegerBlocks', False)
        self.numInitialEvaluationBlocks = params.get('numInitialEvaluationBlocks', 0)
        self.screenSize = np.array(params['screenSize']) # w,h
        self.objScale = params['objScale']
        self.squareScale = params.get('squareScale', self.squareScale)
        self.cursorRadius = params['cursorRadius']
        self.cursorVel = np.array(params['cursorVel']) #*(self.params['dt']/20000)
        self.driftCompensation = params.get('driftCompensation',None)
        if type(self.driftCompensation) is list: self.driftCompensation = np.array(self.driftCompensation)
        self.showAllTarget = params['showAllTarget']
        self.centerCorrectTarget = params.get('centerCorrectTarget', False)
        self.hideCursor = params.get('hideCursor', False)
        self.promptArrow = params.get('promptArrow', False)
        self.drawFixationCross = params.get('drawFixationCross', False)
        self.deltaPosScale = params.get('deltaPosScale', 1.0)
        self.targetDesiredBorderL = params.get('targetDesiredBorderL', 2)
        self.ignoreWrongTarget = params['ignoreWrongTarget']
        self.ignoreCorrectTarget = params.get('ignoreCorrectTarget', False)
        self.skipFirstNtrials = params['skipFirstNtrials']
        self.cursorMoveInCorretDirectionOnly = params.get('cursorMoveInCorretDirectionOnly', self.cursorMoveInCorretDirectionOnly)
        self.constrain1D = params.get('constrain1D', self.constrain1D)
        self.restrictHalfPlane = params.get('restrictHalfPlane', self.restrictHalfPlane)
        self.correctVsOthers = params.get('correctVsOthers', self.correctVsOthers)
        self.useRandomTargetPos = params['useRandomTargetPos']
        self.randomTargetPosRadius = params['randomTargetPosRadius']
        self.randomTargetMinRadius = params.get('randomTargetMinRadius', 0)
        if 'centerIn' in params: self.centerIn = params['centerIn']
        self.enforceCenter = params.get('enforceCenter', 'reset')
        self.resetCursorPos = params['resetCursorPos']

        if 'targetWrongColor' in params: self.targetWrongColor =  params['targetWrongColor']
        if 'targetDesiredColor' in params: self.targetDesiredColor = params['targetDesiredColor']
        if 'targetHoldColor' in params: self.targetHoldColor = params['targetHoldColor']

        self.targetsWord = params.get('targetsWord',{})
        self.useCircleTargets = params.get('useCircleTargets', False)
        self.usePolygonTargets = params.get('usePolygonTargets', False)
        self.targetsInfo = {}
        for k,v in params['targetsInfo'].items():
            if type(v) is str: 
                self.targetsInfo[k] = v
                self.targetsPos[k] = v
            else: 
                self.targetsInfo[k] = [np.array(v[0]),np.array(v[1])]
                self.targetsPos[k] = np.array(v[0])
        if 'defaultTargetSize' in params:
            self.defaultTargetSize = np.array(params['defaultTargetSize'])
        else:
            self.defaultTargetSize = np.array(list(self.targetsInfo.items())[0][1][1])
        self.desiredTargetList = [k for k in self.targetsInfo]
        if self.centerIn: self.targetsInfo['center'] = [np.array([ 0  , 0]), self.defaultTargetSize] # center target information is needed
        self.targetsPos = {}
        for k,v in self.targetsInfo.items():
            self.targetsPos[k] = v if type(v) is str else v[0]
        self.nBit = math.log(len(self.targetsInfo),2)

        self.renderAngles = params.get('renderAngles', self.renderAngles) # used to rotate cursor and target rendering during 1D.

        self.decodedVel = {}
        for k,v in params['decodedVel'].items():
            self.decodedVel[k] = np.array(v)
        self.target2state_task = params['target2state_task']
        self.target2state_task["random"] = ord('r') # 114 means random
        self.target2state_task["center"] = ord('c') # center is special target
        self.state_task2target = {v:k for k,v in self.target2state_task.items()}

        self.pizza = params.get('pizza', self.pizza)
        if self.pizza is not None:
            self.set_pizza_param(self.pizza)

        self.holdTimeThres = params['holdTimeThres']
        self.dwellTimeThres = params['dwellTimeThres'] if 'dwellTimeThres' in params else self.holdTimeThres

        if 'graceTimeThres' in params:
            self.graceTimeThres = params['graceTimeThres']
        self.softmaxThres = params['softmaxThres']
        self.assistValue = params['assist']
        self.assistMode = params['assistMode']

        self.activeLength = params['activeLength']
        self.inactiveLength = params['inactiveLength']
        if 'calibrationLength' in params: self.calibrationLength = params['calibrationLength']
        if 'delayedLength' in params: self.delayedLength = params['delayedLength']
        self.yamlName = params['yamlName']
        self.yamlLastModified = os.path.getmtime(self.yamlName)
        self.showTextMetrics = params['showTextMetrics'] if 'showTextMetrics' in params else None
        self.showCountDown = params['showCountDown'] if 'showCountDown' in params else False
        self.showSoftmax = params['showSoftmax']
        self.showVelocity = params.get('showVelocity',False)
        self.controlSoftmax = params.get('controlSoftmax',False)
        self.showColorSoftmax = params.get('showColorSoftmax',False)
        self.showStats = params.get('showStats',False)
        self.showHeatmap = params['showHeatmap'] if 'showHeatmap' in params else self.showHeatmap
        self.showHeatmapNumber = params['showHeatmapNumber'] if 'showHeatmapNumber' in params else None

        # additional heatmap setup (if forced by the yaml file). requires setting self.showHeatmapNumber value
        # it turns heatmap on even if copilot doens't inherently have one
        if type(self.showHeatmapNumber) is int:
            if hasattr(self, 'copilotObsDim'): 
                if self.copilotObsDim["heatmap"] is not None:
                    raise Exception("READ THIS ERROR: copilot already has its own heatmap dimension. you cannot set 'showHeatmapNumber' in yaml file")
            n = self.showHeatmapNumber
            self.heatmap = Heatmap(n)

        self.showPredictedTarget = params['showPredictedTarget']
        self.showColorCursor = params['showColorCursor'] if 'showColorCursor' in params else False
        self.hideCom = params.get("hideCom",self.hideCom)
        self.hideMass = params.get("hideMass",self.hideMass)
        if self.hasCopilot:
            self.TaskCopilotAction.ActionType.hideMass = self.hideMass
        if 'copilot_alpha' in params:
            self.copilot_alpha = params['copilot_alpha'] * 2 - 1
        else: 
            if hasattr(self, 'copilot_default_alpha'):
                self.copilot_alpha = self.copilot_default_alpha
        self.copilotVel = params.get('copilotVel', (0.02,0.02))        
        self.kfCopilotAlpha = params['kfCopilotAlpha']
        self.enableKfSync = params.get('enableKfSync', self.enableKfSync)
        self.enableKfAdapt = params.get('enableKfAdapt', self.enableKfAdapt)
        self.enableKfAdaptDelay = params.get('enableKfAdaptDelay', self.enableKfAdaptDelay)
        self.randomInitCursorPosition = params['randomInitCursorPosition'] if 'randomInitCursorPosition' in params else self.randomInitCursorPosition
        self.styleChange = params['styleChange']
        if 'styleChangeBallSize' in params:
            self.styleChangeBallSize = params['styleChangeBallSize']
        if 'styleChangeCursorSize' in params:
            self.styleChangeCursorSize =np.array(params['styleChangeCursorSize'])
        if 'softmaxStyle' in params:
            self.softmaxStyle = params['softmaxStyle']

        # dependent constants (should not be tuned!)
        self.episodeLength = self.inactiveLength + self.delayedLength + self.activeLength
        self.tickTimer = 0
        self.tickLength = params['dt']/1e6 if 'dt' in params else 0.020 # each tick is assumed to be 20ms by default, else use params['dt']/1e6 seconds
        self.calibrationTickLength = self.calibrationLength / self.tickLength
        self.activeTickLength = self.activeLength / self.tickLength
        self.inactiveTickLength = self.inactiveLength / self.tickLength
        self.delayedTickLength = self.delayedLength / self.tickLength
        self.episodeTickLength = self.episodeLength / self.tickLength
        self.graceTickLength = self.graceTimeThres / self.tickLength
        self.KfSyncDelayTickLength = self.inactiveTickLength + self.delayedTickLength + self.enableKfAdaptDelay / self.tickLength

        if 'stillStateParam' in params:
            self.stillStateParam = params['stillStateParam']
            self.useClickState = (len(self.stillStateParam) == 2)
            self.targetHitHistory = []
        else:
            self.useClickState = False
        
        if 'nonLinearVel' in params:
            self.useNonLinearVel = True
            self.nonLinearVelParam = params['nonLinearVel']
        else:
            self.useNonLinearVel = False

        if 'smoothVelocity' in params and params['smoothVelocity'] > 1:
            self.useSmoothVelocity = True
            self.smoothVelocityBuffer = [np.array([0,0])]*params['smoothVelocity']
        else:
            self.useSmoothVelocity = False

        # using cm gameBoard
        self.useCmBoard = 'useCmBoard' in params and params["useCmBoard"]
        self.cmBoardDetail = params["cmBoardDetail"] if self.useCmBoard else {} 
        if self.useCmBoard:
            self.screenInfo = self.getScreenInfo()
        self.overrideSyntheticSoftmax = params.get('overrideSyntheticSoftmax',False)

        # parameter for gamifying the task
        self.gamify = params['gamify'] if 'gamify' in params else False
        self.gamifyOption = params['gamifyOption'] if 'gamifyOption' in params else {}
        if self.gamify: 
            targetColor = {'wrong':self.targetWrongColor,'desired':self.targetDesiredColor,'hold':self.targetHoldColor}
            self.GamifyClass.initGamifyOption(self.gamifyOption,targetColor)
            self.targetWrongColor = targetColor['wrong']
            self.targetDesiredColor = targetColor['desired']
            self.targetHoldColor = targetColor['hold']
        
        self.overrideStateTask = params.get('overrideStateTask', False)

        self.fullScreen = params['fullScreen'] if 'fullScreen' in params else False
        self.backgroundMoves = params.get('backgroundMoves',False)
        self.useColorBackground = params.get('useColorBackground',False)

        # copilot task tools
        taskTools = params.get('copilot_tools',[])
        self.TaskTools = None if taskTools == [] else TaskTools(self.TaskCopilotObservation,self.TaskCopilotAction,tools=taskTools)

        # other modules
        self.AssistClass.yamlSettingChanged(assistMode=self.assistMode,assistValue=self.assistValue,cursorSpeed=self.cursorVel[0],tickLength=self.tickLength)


    def getScreenInfo(self):
        """ very awful way to get screen info regarding pixel and cm but this is the best I have currently. 
        it will cause the Tk inter window to appear for split second at the beginning of trial """
        root = tk.Tk()
        info = {}
        info["width_px"] = root.winfo_screenwidth()
        info["height_px"] = root.winfo_screenheight()
        info["width_cm"] = self.cmBoardDetail["actualScreenSize"][0] #root.winfo_screenmmwidth() / 10
        info["height_cm"] = self.cmBoardDetail["actualScreenSize"][1] #root.winfo_screenmmheight() / 10
        info["width_cm2px"] = info["width_px"] / info["width_cm"]
        info["height_cm2px"] = info["height_px"] / info["height_cm"]
        root.after(0, root.destroy)
        root.mainloop()
        return info
    
    def init_copilot_param_with_yaml(self,copilotYamlParam, hideMass):
        """ 
            read copilot yaml to determine obs dimension and action dimension 
            save these values in 
            self.copilotObsDim
            self.copilotActionDim
        """
        self.TaskCopilotAction = TaskCopilotAction(copilotYamlParam,)
        self.TaskCopilotObservation = TaskCopilotObservation(copilotYamlParam)
        self.copilotInfo = self.TaskCopilotAction.copilotYamlParam.get("copilot",None)

        # add additional parameter, information, function
        self.TaskCopilotAction.ActionType.add_etc = self.add_etc
        self.TaskCopilotAction.ActionType.hideMass = hideMass
        self.TaskCopilotAction.updateTargetPos(self.targetsPos)
        self.TaskCopilotObservation.updateTargetsInfo(self.targetsInfo)


        self.copilotObsDim = self.TaskCopilotObservation.copilotObsDim
        self.n_copilotObsDim = self.TaskCopilotObservation.n_copilotObsDim
        self.heatmap = self.TaskCopilotObservation.heatmap

        self.copilot_default_alpha = self.TaskCopilotAction.copilot_default_alpha
        self.copilot_alpha = self.TaskCopilotAction.copilot_alpha
        self.noHoldTimeByCopilot = self.TaskCopilotAction.noHoldTimeByCopilot        
        self.copilotActionTypes = self.TaskCopilotAction.copilotActionTypes
        self.n_copilotActionDim = self.TaskCopilotAction.n_copilotActionDim


    def getImage(self,imageName,size):
        fileName = 'asset/img/' + imageName + '.png'
        imgScreen = pygame.image.load(fileName)
        imgScreen = pygame.transform.smoothscale(imgScreen, size.astype(int))
        return imgScreen

    def initPygameImage(self,corrospondence,imgTargetsImgPos,defaultSize,cursorSize):
        """ want to return dictionary with targetNames:image
        i.e) image[cursor] = pygameimage of cursor
        image[left] = [pygameimage of default target, pygameimage of hold target]
        image[default] = # default one based on default targetSize
        """
        
        def getImage(imageName,size):
            
            key = imageName+str(size) 
            if key in self.generatedImage: # use cached image if it exists
                return self.generatedImage[key]

            fileName = 'asset/img/' + imageName + '.png'
            imgScreen = pygame.image.load(fileName)
            imgScreen = pygame.transform.smoothscale(imgScreen, size.astype(int)) 
            self.generatedImage[key] = imgScreen # cache
            return imgScreen

        # set up targets
        name2image = {}
        for k,(TL,BR,size) in imgTargetsImgPos.items():
            name2image[k] = [getImage(imageName,size) for imageName in corrospondence["target"]]

        # set up default target
        name2image["default"] = [getImage(imageName,defaultSize) for imageName in corrospondence["target"]]

        # set up cursor
        name2image["cursor"] = getImage(corrospondence["cursor"],cursorSize)

        return name2image

    def ready(self, eegIndex:int):
        """
        returns boolean
        once eeg is ready, return True to indicate that the task can officially start
        this is only used by 4_directions outer loop to begin using task.update()
        """
        if self.isReady: return True
        else:
            if hasattr(self, 'eegIndex'):
                if self.eegIndex != eegIndex:
                    self.isReady = True
                    self.complete_restart()
                    return True
            else:
                self.eegIndex = eegIndex
            return False

    def update_kf(self, param):

        kf_state_ref = param[3]
        # extract parameters

        decoder_output = param[0] # decoder_output uses MSEs, but the argmax is still valid
        softmax = decoder_output
        copilot_output = param[1]

        # check pygame event
        if self.render:
            self.check_pygame_event()

        reset_cursorPos_after_update = False
        if self.reset:
            # Make sure the position in kf_state-->kf_state_ref, which is used to update the cursor
            #   is appropriately reset.
            # Note that in self.update_cursor_kf_copilot, self.cursorPos is updated again based on kf_state_ref
            if self.resetCursorPos:
                kf_state_ref[0:2] = [0.0, 0]
                reset_cursorPos_after_update = True
        if self.resetNextCursorPos:
            kf_state_ref[0:2] = [0.0, 0]
            # IMPORTANT: if you pass kf_state as param[3],
            #   then this line `kf_state_ref[2:6] = 0.0`
            #   will modify the shared memory variable kf_state.
            #   This is intended behavior as of 01/19/2024.
            kf_state_ref[2:6] = 0.0
            reset_cursorPos_after_update = True
        self.resetNextCursorPos = False # must occur before self.reset_and_determine_target and after checking self.resetNextCursorPos
        
        # determine target
        target, beginTrial, isActive = self.reset_and_determine_target(self.reset)
        self.target = target
        if beginTrial: self.PerformanceRecord.trial_start(self.cursorPos, self.targetsPos[target], self.activeTimeRecordingStart, self.tickTimer)
        
        # update cursor position. This also updates self.cursorPos.
        cursorPos, click = self.update_cursor_kf_copilot(kf_state_ref, copilot_output, decoder_output, target)
        # reset self.cursorPos (again, non-redundant) if necessary.
        if reset_cursorPos_after_update:
            cursorPos[:] = 0.0
            self.cursorPos = cursorPos.copy()
        #cursorPos = self.cursorPos # sync cursorPos with self.cursorPos

        # drift compensation
        if self.driftCompensation is not None:
            cursorPos += self.getDriftCompensation()
            self.cursorPos = cursorPos
        
        # check if hit Target
        targetHit, holdTime = self.hit_target(cursorPos, target)
        
        # deteremine game state (n,H,h,W,w)
        self.game_state = self.determine_game_state(holdTime, target, targetHit, decoder_output, click)
        
        # determine cursor color
        if self.showColorCursor:
            self.cursorColor = self.determine_cursor_color(targetHit)

        # update heat map
        if self.heatmap is not None and self.game_state.islower():
            self.heatmap.update(cursorPos)
            if self.heatmap.useCom and not self.hideCom: self.add_etc("com", self.heatmap.com)

        # use tools
        if self.TaskTools is not None:
            self.TaskTools.mark_trajectory(self.etc)
            self.TaskTools.show_copilot_contribution(self.etc)

        # record all the observation seen
        if isActive: self.PerformanceRecord.record_step(cursorPos)

        # draw on pygame
        if self.render:
            secondCursorPos = self.secondCursorPos if self.showNonCopilotCursor else None
            self.draw_state(cursorPos, targetHit, target, softmax, self.etc, secondCursorPos)

        # create values to be returned by update function (targetpos, targetsize etc)
        state_taskidx = self.determine_taskidx(target,cursorPos)
        targetPos, targetSize, self.detail = self.create_output(target, softmax)

        # end by incresing ticktimer (internal timer on unit of 20ms)
        self.tickTimer += 1

        # determine if reset is necessary. This should always be at the end
        self.reset = self.determine_reset(self.game_state) 

        # marks the end of trial
        if self.reset: self.PerformanceRecord.trial_end(self.game_state, cursorPos, self.activeTimeRecordingEnd,self.tickTimer)

        # reset some variables
        if len(self.etc) > 0: self.reset_etc()

        # save everything to observation
        self.save_to_observation(softmax, cursorPos, holdTime, target, targetHit)

        # update Stats
        self.update_stats(softmax, cursorPos, targetPos, targetSize)

        # update kf details
        self.kfDetail = self.get_kf_detail(self.reset)
        

        return [cursorPos, targetPos, targetSize, state_taskidx, self.game_state, self.detail, self.kfDetail]

    def update(self, param):

        # extract parameters
        decoder_output = param[0]
        copilot_output = param[1]
        softmax = decoder_output

        # if True: # and self.copilot_action_type == 'copilot_cursor':
        #     copilot_output = (np.random.rand(2)-0.5) * 0.05 # np.array([0.05,0.05])
        #     self.copilotCursorPos += copilot_output[:2]
        #     self.copilotCursorPos = np.clip(self.copilotCursorPos,-1,1)
        #     self.etc['copilot_cursor'] = [self.copilotCursorPos,10,(255,255,100)]
        #     copilot_output = None

        # if synthetic softmax rewrite decoder_output
        if self.gamify:
            softmax = self.GamifyClass.getSyntheticSoftmax(self.pastCursorPos, self.pastTargetPos, self.pastTargetSize)
            decoder_output = softmax
        
        if self.overrideSyntheticSoftmax:
            if self.target is not None:
                _targetPos,_targetSize = self.targetsInfo[self.target]
            else:
                _targetPos = self.getTruthPredictorTargeterPos(self.target)
                _targetSize = self.defaultTargetSize
            decoder_output = softmax = self.tempSynSoft.twoPeakSoftmax(self.cursorPos, _targetPos, _targetSize, CSvalue=1.0, stillCS=1.0)
            

        # extract etc if there is any
        etc = param[2] if len(param) > 2 else {}
        
        # check pygame event
        if self.render:
            self.check_pygame_event()

        if self.resetNextCursorPos:
            self.cursorPos = np.zeros(2)
            self.resetNextCursorPos = False

        # determine target
        target, beginTrial, isActive = self.reset_and_determine_target(self.reset)
        self.target = target
        if beginTrial: self.PerformanceRecord.trial_start(self.cursorPos, self.targetsPos[target], self.activeTimeRecordingStart, self.tickTimer)

        # update cursor position
        cursorPos, click = self.update_cursor(decoder_output,copilot_output,target)

        # drift compensation
        if self.driftCompensation is not None:
            cursorPos += self.getDriftCompensation()
            self.cursorPos = cursorPos

        # check if hit Target
        targetHit, holdTime = self.hit_target(cursorPos,target)
        
        # deteremine game state (n,H,h,W,w)
        self.game_state = self.determine_game_state(holdTime,target,targetHit,decoder_output,click)

        # determine cursor color
        if self.showColorCursor:
            self.cursorColor = self.determine_cursor_color(targetHit)

        # update heat map
        if self.heatmap is not None and self.game_state.islower():
            self.heatmap.update(cursorPos)
            if self.heatmap.useCom and not self.hideCom: self.add_etc("com",self.heatmap.com)

        # use tools
        if self.TaskTools is not None:
            self.TaskTools.mark_trajectory(self.etc)
            self.TaskTools.show_copilot_contribution(self.etc)

        # record all the observation seen
        if isActive: self.PerformanceRecord.record_step(cursorPos)

        # draw on pygame
        if self.render:
            secondCursorPos = self.secondCursorPos if self.showNonCopilotCursor else None
            self.draw_state(cursorPos, targetHit, target, softmax, self.etc, secondCursorPos)

        # create values to be returned by update function (targetpos, targetsize etc)
        state_taskidx = self.determine_taskidx(target,cursorPos)
        targetPos, targetSize, self.detail = self.create_output(target, softmax)

        # end by incresing ticktimer (internal timer on unit of 20ms)
        self.tickTimer += 1

        # determine if reset is necessary. This should always be at the end
        self.reset = self.determine_reset(self.game_state) 

        # marks the end of trial
        if self.reset: self.PerformanceRecord.trial_end(self.game_state, cursorPos, self.activeTimeRecordingEnd,self.tickTimer)

        # reset some variables
        if len(self.etc) > 0: self.reset_etc()

        # save everything to observation
        self.save_to_observation(softmax, cursorPos, holdTime, target, targetHit)

        # update Stats
        self.update_stats(softmax,cursorPos,targetPos,targetSize)

        return [cursorPos, targetPos, targetSize, state_taskidx, self.game_state, self.detail]

    replayTimestep = 0
    fastForwardSpeed = 2
    stopOn = False
    def replay(self):
        """ this is used instead of update in SJ_4_directions_replay to replay an experiment """
    
        # controlling speed
        if self.pygameKeyPressed==pygame.K_LEFT: speedx = -self.fastForwardSpeed
        elif self.pygameKeyPressed==pygame.K_RIGHT: speedx = self.fastForwardSpeed 
        if self.stopOn: speedx = 0
        else: speedx = 1
        self.replayTimestep += speedx


        # fetch data
        state_task = self.replayData['state_task'][self.replayTimestep][0]
        softmax = self.replayData['decoder_output'][self.replayTimestep]
        cursorPos = self.replayData['decoded_pos'][self.replayTimestep]
        target_pos = self.replayData['target_pos'][self.replayTimestep]
        targetSize = self.replayData['target_size'][self.replayTimestep]
        game_state = self.replayData['game_state'][self.replayTimestep][0]
        eeg_step = self.replayData['eeg_step'][self.replayTimestep]
        target = self.state_task2target[state_task]

        if 'copilot_kf_state' in self.params:
            kf_state = self.replayData['kf_state'][self.replayTimestep,[3,2,4,5]]
            softmax = np.append(kf_state, np.zeros(1))
            softmax = np.clip(softmax,0,1)

        # fetch EEGData
        if self.hasReplayDecoder:
            eegSlice = self.fetch_eeg_slice(eeg_step)
            decoderSoftmax = np.zeros(5)
            c = self.eegDecoder(eegSlice)
            output = self.eegDecoder(eegSlice).flatten().detach().numpy()
            decoderSoftmax[:len(output)] = output
            v,_,_ = self.decode_direction(decoderSoftmax, None, target)
            if state_task == -1:
                self.secondCursorPos = np.zeros(2)
            else:
                self.secondCursorPos += v * self.cursorVel
                self.secondCursorPos = np.clip(self.secondCursorPos,-1,1)
            
            secondCursorPos = self.secondCursorPos
            
            # hit?
            # if holdTime is not None and holdTime > self.holdTimeThres:
            #     secondCursorHitRate += 1
            secondCursorHitRate = 999
        else:
            secondCursorHitRate = 0
            decoderSoftmax=None
            secondCursorPos=None

        # update Stats
        self.update_stats(softmax,cursorPos,target_pos,targetSize,secondCursorHitRate)
        
        # fetch target
        target = self.state_task2target[state_task]
        if state_task == ord('r'):
            self.insertRandomTargetPos(target_pos,self.defaultTargetSize)
            
        # game state and score keeping
        game_state = chr(int(game_state))


        # controlling trial / timer speed
        if self.pygameKeyPressed is not None and not self.PRESSED_ONCE: # denoised clicker
            self.PRESSED_ONCE = True
            if self.pygameKeyPressed==ord('s'): 
                self.fastForwardSpeed = max(1,self.fastForwardSpeed-1)
                print("fastforward speed magnitude",self.fastForwardSpeed)
            elif self.pygameKeyPressed==ord('d'): 
                self.fastForwardSpeed += 1
                print("fastforward speed magnitude",self.fastForwardSpeed)
            elif self.pygameKeyPressed==pygame.K_SPACE:
                self.stopOn = not self.stopOn 
            elif self.pygameKeyPressed==pygame.K_UP: 
                self.secondCursorPos = np.zeros(2)
                self.trialCount = max(self.trialCount-1,0)
                t,n,h = self.trialRangeReplayData[self.trialCount]
                self.replayTimestep = t
                self.hitRate = h
                _,nActuallySkippedTrials,_ = self.trialRangeReplayData[self.skipFirstNtrials]
                self.missRate = n - h - (nActuallySkippedTrials)
                self.tickTimer = self.inactiveTickLength
            elif self.pygameKeyPressed==pygame.K_DOWN: 
                self.secondCursorPos = np.zeros(2)
                self.trialCount = min(self.trialCount+1,len(self.trialRangeReplayData)-1)
                t,n,h = self.trialRangeReplayData[self.trialCount]
                self.replayTimestep = t
                self.hitRate = h
                _,nActuallySkippedTrials,_ = self.trialRangeReplayData[self.skipFirstNtrials]
                self.missRate = n - h - (nActuallySkippedTrials)
                self.tickTimer = self.inactiveTickLength
        self.replayTimestep = np.clip(self.replayTimestep,0,self.replayTimestepMax-1)
        


        # calculate hit rate
        if speedx == 1:
            if game_state.isupper():
                self.trialCount += 1
                self.tickTimer = 0
                if target != 'still':
                    if game_state == 'H': self.hitRate += 1 
                    elif game_state == 'W': self.missRate += 1
                    elif game_state == 'T':
                        self.missRate += 1
                        self.timeoutRate += 1

        elif speedx > 1:
            for game_state in self.replayData['game_state'][self.replayTimestep-(speedx-1):self.replayTimestep+1]:
                game_state = chr(int(game_state))
                if game_state.isupper():
                    self.trialCount += 1
                    self.tickTimer = 0
                    if target != 'still':
                        if game_state == 'H': self.hitRate += 1
                        elif game_state == 'W': self.missRate += 1
                        elif game_state == 'T': self.missRate += 1

        elif speedx < 0:
            for game_state in self.replayData['game_state'][self.replayTimestep:self.replayTimestep-(speedx)]:
                game_state = chr(int(game_state))
                if game_state.isupper():
                    self.trialCount -= 1
                    self.tickTimer = self.trialRangeReplayData[self.trialCount+1][0] - self.trialRangeReplayData[self.trialCount][0]
                    if target != 'still':
                        if game_state == 'H': self.hitRate -= 1
                        elif game_state == 'W': self.missRate -= 1
                        elif game_state == 'T': self.missRate -= 1

        print('trial:',self.trialCount, self.replayTimestep,cursorPos,state_task)

        if self.trialCount <= self.skipFirstNtrials:
            self.hitRate = 0
            self.missRate = 0
            self.timeoutRate = 0

        # check pygame event
        if self.render:
            self.check_pygame_event()

        # check if hit Target
        targetHit, holdTime = self.hit_target(cursorPos,target)

        # handle timer
        if self.replayTimestep == self.replayTimestepMax-1:
            print("Replay: you've reached the end")
        else:
            self.tickTimer += speedx

        # # draw
        if self.render:
            replayTimeElapsed = self.replayTimestep * self.tickLength
            self.draw_state(cursorPos, targetHit, target, softmax, self.etc, secondCursorPos=secondCursorPos, secondSoftmax=decoderSoftmax,replayTimeElapsed=replayTimeElapsed)
            
        # reset some variables
        if len(self.etc) > 0: self.reset_etc()

    def fetch_eeg_slice(self,eeg_step):

        # downsample
        if eeg_step < self.eegInput_length: return np.zeros((self.eegDownsampled_length,self.eegData.shape[1]))
        else: data = self.eegData[eeg_step-self.eegInput_length: eeg_step]
        data = resample(data, self.eegDownsampled_length, axis=0)

        return data

    def update_stats(self,softmax,cursorPos,targetPos,targetSize,secondCursorHitRate=0):
        """
        updates stats dictionary with appropriate values
        """

        if not self.showStats: return

        # resets every stop
        if np.isnan(targetPos[0]):
            self.stats['Streak'] *= 0
            self.stats['CorrectAction'] *= 0
            self.stats['TotalAction'] *= 0
            self.stats['CorrectSoftmax'] = 0
            self.stats['NonStillTTimer'] = 0
            self.stats['TrialTime'] = 0
            self.stats['Second Hitrate'] = secondCursorHitRate
            return

        self.stats['TrialTime'] += 1

        # create vector with maximum = 1 all other as 0
        v = np.zeros(len(softmax), dtype=int)
        v[np.where(softmax==np.max(softmax))] = 1 

        # update streak for 5 softmax value
        streak = self.stats['Streak']
        streak *= v
        streak += v

        # update total sstreak
        self.stats['TotalAction'] += v

        # update correctSoftmax
        if not np.isnan(targetPos[0]):
            correct = self.is_correct_softmax(softmax,cursorPos,targetPos,targetSize)
            self.stats['CorrectSoftmax'] += correct
            self.stats['CorrectAction'][np.argmax(softmax)] += correct
            
            if correct or v[4] != 1:
                self.stats['NonStillTTimer'] += 1
    
    def is_correct_softmax(self, softmax, cursor, target, targetSize):
        """
        returns true or false depending whether softmax is pointing in the correct direction
        """
        correctDirection = target-cursor
        chosen = np.argmax(softmax)

        # still
        if chosen == 4: 
            # print(cursor,target,np.max(np.abs(correctDirection)) < targetSize[0],np.max(np.abs(correctDirection)),targetSize[0] / 2,)
            return np.max(np.abs(correctDirection)) < targetSize[0] / 2
        if correctDirection[0] > 0 and chosen == 0: return False
        if correctDirection[0] < 0 and chosen == 1: return False
        if correctDirection[1] > 0 and chosen == 3: return False
        if correctDirection[1] < 0 and chosen == 2: return False
        return True



    def check_pygame_event(self):
        """ event handler for pygame intrinsic """
        
        # check for end 
        if self.useRealTime:
            if time.time() - self.startTime > self.sessionLength + 1:
                if (not self.waitIntegerBlocks) or (self.integerBlocks):
                    # self.waitIntegerBlocks: False -> exit after this time
                    # else: self.integerBlocks: True -> exit only when integerBlocks is satisfied.
                    raise KeyboardInterrupt
                    exit(1)

        # check for event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
                exit(1)
            if event.type == pygame.VIDEORESIZE:
                # print("#######################\n#######################\n#######################\n#######################\n#######################\n#######################\n#######################\n")
                self.update_window_size_constants(humanInduced=True)
                

            # special pygame keypress recorder
            if event.type == pygame.KEYDOWN:
                # store key that was pressed
                self.key_pressed[pygame.key.name(event.key)] = True

                self.pygameKeyPressed = event.key
                if event.key == pygame.K_a: self.K_a_pressed = True

                # reset the game if r is pressed
                if self.pygameKeyPressed == pygame.K_r:
                    self.complete_restart()

                if self.pygameKeyPressed == pygame.K_SPACE:
                    self.secondCursorPos = self.cursorPos

            if event.type == pygame.KEYUP:
                # store key that was unpressed
                self.key_pressed[pygame.key.name(event.key)] = False

                if event.key == pygame.K_a:  self.K_a_pressed = False
                self.pygameKeyPressed = None
                self.PRESSED_ONCE = False

            if event.type == pygame.MOUSEMOTION:
                self.mousePos = (np.array(event.pos) - self.windowPadding)
                self.mousePos = (np.clip(self.mousePos,0,self.L) - self.L/2) / (self.L/2)
                self.mousePos[1] *= -1

            # pygame
            if self.gamify: self.GamifyClass.pygameEvent(event)


    def reset_and_determine_target(self, reset):
        """ 
        determine target based on reset flag 
        returns targetName, beginTrial, active
        targetName = targetName or None (indicating stop sign)
        beginTrial = bool. True if start of trail (after 2s inactive peiord) or False otherwise
        active = bool. True if active period begins, False when not in active period
        """

        # if reset, run stop for 2 seconds
        if self.reset:
            
            # update drift compensation
            if self.driftCompensation == 'average':
                tickLength = self.tickTimer - self.inactiveTickLength - self.delayedTickLength
                if tickLength > 0:
                    self.driftTotals.append(np.copy(self.cursorPos))
                    self.driftNs.append(tickLength)

            self.trialCount += 1
            self.reset = False
            self.tickTimer = 0
            self.heldTimeBegin = holdTime = None
            self.wrongToleranceCounter = 0
            if self.resetCursorPos: 
                self.cursorPos = np.array([0.,0.])
                self.secondCursorPos = np.array([0.,0.])
                
            # use calibration length if it is in calibration phase
            if self.trialCount < self.skipFirstNtrials:
                self.activeLength = self.calibrationLength
                self.episodeLength = self.inactiveLength + self.delayedLength + self.activeLength
                self.activeTickLength = self.calibrationTickLength
                self.episodeTickLength = self.episodeLength / self.tickLength

            if self.trialCount == self.skipFirstNtrials + 1:
                if hasattr(self, 'yamlName'):
                    yaml_file = open(self.yamlName, 'r')
                    yaml_data = yaml.load(yaml_file, Loader=Loader)
                    yaml_file.close()
                    self.yaml_data = yaml_data # used for auto README
                    print('warning: updated self.yaml_data')
                    self.initParamWithYaml(yaml_data["modules"]["SJ_4_directions"]["params"])


            # class reset
            self.AssistClass.resetEveryTrial()

            # reset mouse position reset if needed
            self.resetYamlParam()

            # pseudo randomize target choice
            if len(self.nextTargetBucket) == 0:
                ## uncomment to have each group of n targets shared the same render_angle
                # try:
                #     self.render_angle = self.renderAnglesShuffled.pop()
                # except:
                #     self.renderAnglesShuffled = random.sample(self.renderAngles, k=len(self.renderAngles))
                #     self.render_angle = self.renderAnglesShuffled.pop()

                # for calibration phase
                if self.trialCount < self.skipFirstNtrials:
                    choices = list(['left','right','up','down'])
                    choices = choices[:self.skipFirstNtrials]
                    random.shuffle(choices)

                else:
                    # randomize target choices
                    choices = list(self.desiredTargetList)
                    random.shuffle(choices)
                    
                    if self.centerIn:
                        # interleave centerIn
                        interLeavedChoices = []
                        for item in choices: 
                            if item == "still": interLeavedChoices.append(item)
                            else: interLeavedChoices += ['center',item]
                        choices = interLeavedChoices
                        # print(choices)

                # add target choices
                self.nextTargetBucket += choices

                # take care of drift if there is some
                # print("the result",self.driftTotals,self.driftNs) 
                self.driftTotal += sum(self.driftTotals)
                self.driftN += sum(self.driftNs)
                # print("the result",self.driftTotal,self.driftN)
                self.driftTotals = []
                self.driftNs = []
            
            # pseudo randomize target action text
            self.multiply_equation = f"{random.randint(20,40)}x{random.randint(6,50)}=?"
            primeList = [3,4,5,6,7,8,9,11] # [3,5,7,11,13,17,19,23]
            self.subtract_equation = f"{random.randint(100,300)}  -  {random.choice(primeList)}"
            self.word_association = self.generate_next_word_association()

            # pop next target
            if self.centerIn:
                if self.nextTarget != 'center':
                    try:
                        outTimeTrial = time.time() - self.outTimeRef
                        if self.trialCount > self.skipFirstNtrials + 1:
                            self.outTime += outTimeTrial
                    except:
                        pass
                if self.game_state == 'T' and self.nextTarget == 'center':
                    if self.enforceCenter is True:
                        # repeat center target if enforceCenter is true and the trial has a timeout
                        pass
                    else:
                        # don't repeat center target, but instead reset the next cursor position to center.
                        if self.enforceCenter == 'reset':
                            self.resetNextCursorPos = True
                        self.nextTarget = self.nextTargetBucket.pop()
                else:
                    self.nextTarget = self.nextTargetBucket.pop()
                if self.nextTarget != 'center':
                    self.outTimeRef = time.time()
            else:
                self.nextTarget = self.nextTargetBucket.pop()
            
            if self.overrideStateTask:
                # Pull nextTarget from state_task rather than randomly generating it.
                # Somewhat useful for replay, however, due to delta time execution at runtime, the replay will not be exact.
                global state_task
                self.nextTarget = self.state_task2target[state_task[0]]
                
            try:
                self.render_angle = self.renderAnglesShuffled[self.nextTarget].pop()
            except:
                self.renderAnglesShuffled[self.nextTarget] = random.sample(self.renderAngles, k=len(self.renderAngles))
                self.render_angle = self.renderAnglesShuffled[self.nextTarget].pop()

            # override with random target if random target pos used (as in pinball)
            if self.useRandomTargetPos:
                if self.trialCount > self.skipFirstNtrials:
                    if self.nextTarget != 'still': 
                        # delcare random target
                        self.nextTarget = "random"
                        # create random target
                        self.randomTargetSize = self.defaultTargetSize
                        # truly random
                        boundary = (np.array([1,1]) - self.randomTargetSize)
                        if self.randomTargetPosRadius == -1:
                            # truly random
                            self.randomTargetPos = (np.random.rand(2) * 2 - 1) * boundary
                        else:
                            # random with vicinity of 3
                            # see https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
                            pastPos = self.randomTargetPos
                            for i in range(101):
                                # if it can't for some reason stay
                                if i == 100:
                                    print('warning: using default random target of [0.0, 0.0]')
                                    self.randomTargetPos = np.zeros(2)
                                    break
                                
                                mag = self.randomTargetPosRadius*np.sqrt(np.random.random())
                                if mag < self.randomTargetMinRadius:
                                    continue
                                dir = np.random.random() * 2 * np.pi
                                self.randomTargetPos = self.cursorPos + np.array([np.cos(dir), np.sin(dir)]) * mag

                                # if it satisifes condition leave
                                if (np.abs(self.randomTargetPos) <= boundary).all():
                                    #if np.linalg.norm(self.cursorPos - self.randomTargetPos) >= self.randomTargetMinRadius:
                                    #    break
                                    break

                        self.insertRandomTargetPos(self.randomTargetPos,self.randomTargetSize)

            # reset other modules that needs reset
            if self.hasCopilot:
                self.TaskCopilotAction.reset_each_trial()
            
            # print('hits:', self.hitRate, 'completed trials:', self.trialCount - 1) # print hits, trials

        # if not reset
        if self.tickTimer < self.inactiveTickLength: # 2 seconds
            return None, False, False # none means not a target (i.e red stop sign)
        elif self.tickTimer == self.inactiveTickLength:
            # if game requires starting from random cursor position this is where we do it

            # START ACTIVE TIMER (only useful for calcualting bitrate)
            if self.showTextMetrics == 'b':
                self.activeTimeRecordingStart = time.time()

            if self.randomInitCursorPosition: 
                self.cursorPos = np.random.rand(2) * 2 -1

            # restart heat map if needed be
            if self.heatmap is not None:
                self.heatmap.reset()
 
            return self.nextTarget, True, True
        else:   # 10 seconds
            return self.nextTarget, False, True

    @property
    def activeTrialHasBegun(self): # marks when epsiode starts. used by copilot's env AND kf_4_directions for copilot only. Not for task at all
        if self.nextTarget == 'still': return False
        if (self.inactiveTickLength + self.delayedTickLength) == 0: return self.reset
        else: return self.tickTimer == (self.inactiveTickLength + self.delayedTickLength)

    def resetYamlParam(self):
        if hasattr(self, 'yamlName'):

            # skip if no change has been made
            lastModified = os.path.getmtime(self.yamlName)
            if self.yamlLastModified == lastModified:
                return
            self.yamlLastModified = lastModified

            yaml_file = open(self.yamlName, 'r')
            yaml_data = yaml.load(yaml_file, Loader=Loader)
            yaml_file.close()
            
            # maintain screenSize but other than that reset everything
            yaml_data["modules"]["SJ_4_directions"]["params"]["screenSize"] = self.screenSize
            self.initParamWithYaml(yaml_data["modules"]["SJ_4_directions"]["params"])

            self.update_window_size_constants()
            self.yaml_data = yaml_data # used for auto README????
            
    def resetMousePos(self):
        center = np.array(pyautogui.size())//2
        pyautogui.moveTo(center[0], center[1])
        self.mousePos = np.zeros(2)

    def update_cursor_kf_copilot(self, kf_state_ref, copilot_output, decoder_output, target):
        
        # override
        if not self.controlSoftmax:
            if self.pygameKeyPressed is not None:
                manualDirection = False
                if self.pygameKeyPressed == pygame.K_LEFT:
                    print("manual input: left")
                    direction = np.array([-1.,0])
                    manualDirection = True
                if self.pygameKeyPressed == pygame.K_RIGHT:
                    print("manual input: right")
                    direction = np.array([1.,0])
                    manualDirection = True
                if self.pygameKeyPressed == pygame.K_DOWN:
                    print("manual input: down")
                    direction = np.array([0.,-1])
                    manualDirection = True
                if self.pygameKeyPressed == pygame.K_UP:
                    print("manual input: up")
                    direction = np.array([0.,1])
                    manualDirection = True
                if manualDirection:
                    self.cursorPos += direction * self.cursorVel
                    return self.cursorPos, None

        if self.skip_cursor_movement(target, decoder_output): 
            return self.cursorPos, False

        # safe keep
        currentPos = np.copy(self.cursorPos)

        # kf part
        # Updates self.cursorPos, which is overwritten later in this function.
        cursorPos, click = self.update_cursor_kf(kf_state_ref, target)
        kf_effective_vel = cursorPos - currentPos

        if self.hasCopilot:
            # get decoder_direction
            if self.TaskCopilotObservation.useVelReplaceSoftmax:
                decoder_direction = self.TaskCopilotObservation.softmax_to_vel(decoder_output)
            else:
                decoder_output = np.clip(decoder_output,0,1)
                decoder_direction = self.decoder_direction(decoder_output)

            # copilot take action
            if self.TaskCopilotAction.copilotDominatesCursorUpdate:
                args = [currentPos, self.cursorVel, self.copilotVel, kf_effective_vel]
                self.cursorPos, click = self.TaskCopilotAction.interpret(copilot_output, decoder_direction, args)
                try:
                    self.delta_pos_warning
                except:
                    print('warning: bypassing deltapos')
                    self.delta_pos_warning = None
                return self.cursorPos, click
            else:
                direction, click = self.copilot_direction(copilot_output, decoder_direction)
                cursorPos = currentPos + direction * self.cursorVel
                cursorPos = np.clip(cursorPos, -1, 1)
                copilot_effective_vel = cursorPos - currentPos
        else:
            copilot_effective_vel = kf_effective_vel

                
        # merget kf and copilot part
        alpha = self.kfCopilotAlpha
        deltaPos = alpha * kf_effective_vel + (1-alpha) * copilot_effective_vel



        # constrained movement
        if self.constrain1D:
            if target in ['up', 'down', 'up2', 'down2']:
                deltaPos[0] *= 0.0
            elif target in ['left', 'right', 'left2', 'right2']:
                deltaPos[1] *= 0.0

        newPos = currentPos + self.deltaPosScale*deltaPos # separate just in case we want to further process the position.
        newPos = np.clip(newPos, -1, 1)

        if self.restrictHalfPlane:
            if target in ['up', 'up2']:
                newPos[1] = np.clip(newPos[1],  0.0, 1.0)
            elif target in ['down', 'down2']:
                newPos[1] = np.clip(newPos[1], -1.0, 0.0)
            elif target in ['left', 'left2']:
                newPos[0] = np.clip(newPos[0], -1.0, 0.0)
            elif target in ['right', 'right2']:
                newPos[0] = np.clip(newPos[0],  0.0, 1.0)
        self.cursorPos = newPos

        return self.cursorPos, click
        

    def update_cursor_kf(self, kf_state_ref, target):
        ## skip cursor movement if under condition to skip
        #if self.skip_cursor_movement(target, decoder_output): return self.cursorPos, False

        ## get decoded direction np(+-1,+-1) from decoder output / copilot
        #direction, nonCopilotDirection, click = self.decode_direction(decoder_output, copilot_output, target)
        if self.correctVsOthers:
            v = kf_state_ref[2:6]
            if target in ['right', 'right2']:
                v_order = [0, 1, 2, 3]
            elif target in ['left', 'left2']:
                v_order = [1, 0, 3, 2]
            elif target in ['up', 'up2']:
                v_order = [2, 3, 1, 0]
            elif target in ['down', 'down2']:
                v_order = [3, 2, 0, 1]
            v_correct = v[v_order[0]]
            v_others = v[v_order[1:4]]
            v_other = np.max(v_others)
            dr = self.tickLength*self.correctVsOthers*(v_correct - v_other)
            if target in ['right', 'right2']:
                self.cursorPos = self.cursorPos + [dr, 0]
            elif target in ['left', 'left2']:
                self.cursorPos = self.cursorPos + [-dr, 0]
            elif target in ['up', 'up2']:
                self.cursorPos = self.cursorPos + [0, dr]
            elif target in ['down', 'down2']:
                self.cursorPos = self.cursorPos + [0, -dr]
            
            self.secondCursorPos = None
            return self.cursorPos, False
        self.cursorPos = kf_state_ref[0:2]
        self.cursorPos = np.clip(self.cursorPos, -1, 1)
        self.secondCursorPos = None
        click = False
        return self.cursorPos, click
    
    def update_cursor(self,decoder_output,copilot_output,target):
        """ 
            target: target name
            update cursor position: np(+-1,+-1) 
            returns (new cursor position, click state)
        """

        # skip cursor movement if under condition to skip
        if self.skip_cursor_movement(target, decoder_output): return self.cursorPos, False


        # override
        # copilot_output = np.array([0,0,0])
        # if self.pygameKeyPressed is not None:
        #     direction = np.array([0,0])
        #     if self.pygameKeyPressed == pygame.K_LEFT:
        #         print("manual input: left")
        #         direction = np.array([-1.,0])
        #     if self.pygameKeyPressed == pygame.K_RIGHT:
        #         print("manual input: right")
        #         direction = np.array([1.,0])
        #     if self.pygameKeyPressed == pygame.K_DOWN:
        #         print("manual input: down")
        #         direction = np.array([0.,-1])
        #     if self.pygameKeyPressed == pygame.K_UP:
        #         print("manual input: up")
        #         direction = np.array([0.,1])
            
        #     self.cursorPos += direction * 0.02
        #     return self.cursorPos, False

        if self.hasCopilot and self.TaskCopilotAction.copilotDominatesCursorUpdate:
            args = [self.cursorPos, self.cursorVel, self.copilotVel]
            decoder_dir = self.decoder_direction(decoder_output)
            if self.TaskCopilotObservation.useVelReplaceSoftmax:
                decoder_dir = self.TaskCopilotObservation.softmax_to_vel(decoder_output)
            self.cursorPos, click = self.TaskCopilotAction.interpret(copilot_output, decoder_dir, args)
            
            return self.cursorPos, click

        # get decoded direction np(+-1,+-1) from decoder output / copilot
        direction, nonCopilotDirection, click = self.decode_direction(decoder_output, copilot_output, target)
        
        # smooth cursor velocity
        if self.useSmoothVelocity: direction = self.smooth_velocity(direction)

        # non linear velocity
        if self.useNonLinearVel:
            nonlinearGain = self.nonlinear_velocity(direction)
            gain = self.cursorVel * nonlinearGain
        else:
            gain = self.cursorVel
        
        # update current position
        self.cursorPos += direction * gain
        self.cursorPos = np.clip(self.cursorPos, -1, 1)

        # update second cursorpos
        if self.pygameKeyPressed == pygame.K_SPACE:
            self.secondCursorPos += nonCopilotDirection * self.cursorVel
            self.secondCursorPos = np.clip(self.secondCursorPos, -1, 1)
        else:
            self.secondCursorPos = None
        
        return self.cursorPos, click
        
    
    def skip_cursor_movement(self,target, decoder_output):
        """ determine if we need to skip cursor movement """

        # don't update cursor position if your target is None or still
        if target is None or target == "still":
            return True

        # don't update cursor position if you're in skipTrials
        if self.trialCount <= self.skipFirstNtrials:
            return True
        
        # dont' update if cursorMoveInCorretDirectionOnly is on and you are moving in wrong direction
        if self.cursorMoveInCorretDirectionOnly:
            targetPos = self.targetsPos[target]
            cursorPos = self.cursorPos
            if not self.isHeadedInCorrectDirection(decoder_output,cursorPos,targetPos):
                print("not headed in correct ")
                return True

        # don't update cursor position if you're in delayed phase
        if self.tickTimer < self.inactiveTickLength + self.delayedTickLength:
            self.variable4CursorColor['delayedPhase'] = True
            return True
        
        self.variable4CursorColor['delayedPhase'] = False
        return False
    
    def smooth_velocity(self,direction):

        self.smoothVelocityBuffer.append(direction)
        self.smoothVelocityBuffer.pop(0)

        direction = self.smoothVelocityBuffer[-1] * (1.0 - (len(self.smoothVelocityBuffer)-1) * 0.1)
        direction += sum(self.smoothVelocityBuffer[:-1]) * 0.1
        return direction

    def decode_direction(self, decoder_output, copilot_output, target):
        """ find corrosponding direction np(+-1,+-1) from decoder_output"""

        # decoder direction
        decoder_direction = self.decoder_direction(decoder_output)
        if self.hasCopilot and self.TaskCopilotObservation.useVelReplaceSoftmax:
            decoder_direction = self.TaskCopilotObservation.softmax_to_vel(decoder_output)
        
        # copilot
        direction, click = self.copilot_direction(copilot_output, decoder_direction)

        # gamify
        if self.gamify: 
            # reset mouse if needed to
            if self.GamifyClass.gamifyOnMouse and self.tickTimer == self.inactiveTickLength: self.resetMousePos()
            # set direction, cursor pos from gamifed direction
            direction, self.cursorPos = self.GamifyClass.gamified_direction(direction, self.cursorPos, self.mousePos, self.K_a_pressed)

        # assist
        if target in self.targetsImgPos: # must be in targetsImgPos, otherwise it means target doesn't have position
            targetPos = self.targetsPos[target] # get the position
            cursorPos = self.cursorPos
            direction = self.AssistClass.assist(targetPos, cursorPos, direction)
            
        # override
        if self.pygameKeyPressed is not None:
            if self.pygameKeyPressed == pygame.K_LEFT:
                print("manual input: left")
                direction = np.array([-1.,0])
            if self.pygameKeyPressed == pygame.K_RIGHT:
                print("manual input: right")
                direction = np.array([1.,0])
            if self.pygameKeyPressed == pygame.K_DOWN:
                print("manual input: down")
                direction = np.array([0.,-1])
            if self.pygameKeyPressed == pygame.K_UP:
                print("manual input: up")
                direction = np.array([0.,1])

        return direction, decoder_direction, click
    
    def decoder_direction(self, decoder_output):
        """ direction decoder would take in vx, vy [{-1,0,1},{-1,0,1}]"""

        if max(decoder_output) > self.softmaxThres: # 0.8
            directionIdx = np.argmax(decoder_output)
            direction = self.decodedVel[directionIdx]
        else:
            direction = np.array([0,0])
        
        return np.copy(direction)

    def copilot_direction(self, copilot_output, decoder_dir):
        """ generate action give softmax and action of RL """

        # if there is no copilot, don't bother
        if copilot_output is None: 
            return decoder_dir, False

        # get new cursor position, and click state
        cursorPos, click = self.TaskCopilotAction.interpret(copilot_output, decoder_dir)

        # color cursor if alpha is big enough
        self.variable4CursorColor['copilotControl'] = self.TaskCopilotAction.copilotInControl

        return cursorPos, click


    def nonlinear_velocity(self,direction):
        # get relu style velocity

        gain = 1
        mag = np.linalg.norm(direction)

        self.variable4CursorColor['usingNonLinearVel'] = False
        for i,[thres,gain_] in enumerate(self.nonLinearVelParam):
            if mag > thres: 
                gain = gain_
                # turn on color if using last partition of nonlinear vel
                if i+1 == len(self.nonLinearVelParam):
                    self.variable4CursorColor['usingNonLinearVel'] = True
        
        return gain

    def hit_target(self,cursorPos, target):
        """ 
        report whether target was hit for how long (doesn't matter if it hit the correct one or not) 
        returns targetName (None = (nothing was held) or targetName), holdTime (float >= 0)
        """
        
        # add stop sign, refresh everything
        if target == None:
            self.heldTimeBegin = holdTime = None

        # find cursorPos in img coordinate
        cursorImagePos = np.array([cursorPos[0],-cursorPos[1]])
        cursorImagePos = cursorImagePos * self.cursorMagnitude + self.cursorBias 

        # check if cursorImagePos lies in any of the target
        targetHit = None
        if self.usePizza:
            if np.linalg.norm(cursorPos) > self.pizza['radius']:

                # cursor Angle
                sliceAngle = 2*np.pi / self.pizza['slices']

                cursorAngle = np.arctan2(cursorPos[1],cursorPos[0]) - self.pizza['phase']
                cursorAngle = (cursorAngle + np.pi * 2) % (np.pi * 2)
                chosenSlice = int((cursorAngle) // sliceAngle)
                
                targetHit = 'p'+str(chosenSlice)
            

        else:
            for name, (leftTop, rightBottom, _) in self.targetsImgPos.items():
                # circle case
                if self.useCircleTargets:
                    center, target_size = self.targetsInfo[name]
                    center = np.array(center)
                    diameter = target_size[0]
                    if np.linalg.norm(cursorPos - center) <= 0.5*diameter:
                        targetHit = name
                elif self.usePolygonTargets:
                    center, target_verts = self.targetsInfo[name]
                    target_verts = np.array(target_verts)
                    target_next_verts = np.roll(target_verts, -1, axis=0)
                    # see https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html
                    winding_number = 0
                    #ray_origin = cursorPos
                    #ray_direction = np.array([1.0, 0.0])
                    #ray_perp = np.array([-ray_direction[1], ray_direction[0]])
                    for (vert1, vert2) in zip(target_verts, target_next_verts):
                        if vert1[1] == vert2[1]:
                            continue
                        prop = (cursorPos[1] - vert1[1])/(vert2[1] - vert1[1])
                        if prop >= 0 and prop <= 1:
                            if vert1[0] + prop*(vert2[0] - vert1[0]) >= cursorPos[0]:
                                if (vert1[1] >= cursorPos[1] and vert2[1] < cursorPos[1]):
                                    winding_number -= 1
                                if (vert1[1] < cursorPos[1] and vert2[1] >= cursorPos[1]):
                                    winding_number += 1
                    if winding_number % 2 == 1:
                        targetHit = name
                # square case
                else:
                    center, target_size = self.targetsInfo[name]
                    center, target_size = np.array(center), np.array(target_size)
                    if (abs(cursorPos - center) <= 0.5*target_size).all():
                        targetHit = name
            

        # if ignoreWrongTarget, then ignore it
        if self.ignoreWrongTarget:
            if targetHit != target:
                return None, 0 
        if self.ignoreCorrectTarget:
            return None, 0

        if self.tickTimer < self.inactiveTickLength + self.delayedTickLength + self.graceTickLength:
            self.heldTimeBegin = holdTime = None
            return targetHit, 0

        # calculate hold time (increases as cursor stays in box can be None or float)        
        if targetHit is not None:
            if self.heldTimeBegin is None:
                if self.useRealTime:
                    # uses realTime to measure heldtime
                    self.heldTimeBegin = time.time()
                    holdTime = 0
                else:
                    # does not use realTime to measure heldtime
                    holdTime = self.heldTimeBegin = 0
            else:
                if self.useRealTime:
                    holdTime = time.time() - self.heldTimeBegin
                else:
                    self.heldTimeBegin += self.tickLength
                    holdTime = self.heldTimeBegin
        else:
            self.heldTimeBegin = holdTime = None

        return targetHit, holdTime

    def determine_game_state(self, holdTime, target, targetHit,softmax, click):
        """ 
            determine whether we hit target, missed target etc using argument information
            argument: 
                holdTime = None or >= 0 value (0 means just started to hold)
                target = None or string: name of desired target
                targetHit = None or string: name of hit target
                softmax = used for still click state
            return ascii
                T = time out
                h = hold / H = Hit (Acquired)
                w = wrong hold / W = wrong hit
                n = neutral (not hit)
                note trial end condition uses capital letter
        """

        # click state 
        if self.useClickState:

            if self.tickTimer >= self.inactiveTickLength + self.delayedTickLength + self.graceTickLength:
                
                # keep history of target hit
                if len(self.targetHitHistory) == self.stillStateParam[0]: self.targetHitHistory.pop(0)
                self.targetHitHistory.append(np.argmax(softmax) == 4)
                
                # click state when avg > stillStateThres
                avg = sum(self.targetHitHistory) / len(self.targetHitHistory)
                self.variable4CursorColor['clicked'] = False
                if avg >= self.stillStateParam[1]:
                    self.variable4CursorColor['clicked'] = True

                    # to count as click
                    holdTime = max(self.holdTimeThres,self.dwellTimeThres)
                    print("CLICK")
        
        if self.gamify:
            self.manualClick = False  # to be used by logger
            
            if not self.PRESSED_ONCE:
                if self.pygameKeyPressed == pygame.K_SPACE:
                    # set correct flag
                    self.manualClick = True # to be used by logger
                    self.PRESSED_ONCE = True

                    # to count as click
                    holdTime = max(self.holdTimeThres,self.dwellTimeThres)
                    print("CLICK")
                    

        # copilot may click from copilot_nonstandard_action
        if self.noHoldTimeByCopilot:
            holdTime = 0

        if click:
            # to count as click
            holdTime = max(self.holdTimeThres,self.dwellTimeThres)
        self.variable4CursorColor['clicked'] = click


        # if target is a still, then no need to score it
        noTargetState = target not in self.targetsImgPos # still state
        game_state = None
        # check timer is beyond episode length
        if self.tickTimer >= self.episodeTickLength-1:
            if not noTargetState: 
                if self.trialCount > self.skipFirstNtrials:
                    self.missRate += 1
                    self.timeoutRate += 1
                    if target != 'center':
                        self.outMissRate += 1
                        self.outTimeoutRate += 1
            return 'T'  # t = time out 
        if targetHit is None:
            return 'n' # n = not holding target OR netural 
        elif targetHit == target: 
            if holdTime < self.holdTimeThres: return 'h' # h = holding
            else: 
                if self.trialCount > self.skipFirstNtrials:
                    self.hitRate += 1
                    if target != 'center':
                        self.outHitRate += 1
                return 'H' # H = acquired (capitalized holding)
        else: # hitting wrong target
            if self.usePizza:
                pass
            else:
                if not self.showAllTarget:
                    return 'n' # n = not holding target OR netural 
                if noTargetState:
                    return 'n' # if the game says you hit something but you are in still state
            if holdTime < self.dwellTimeThres: 
                return 'w' # w = wrong holding
            else:
                if self.trialCount > self.skipFirstNtrials:

                    # wrong click tolerance
                    if self.wrongToleranceCounter < self.wrongTolerance:
                        self.wrongToleranceCounter += 1
                        self.heldTimeBegin = None # this resets hold time to zero
                        return 't' # means it is tolerated once
                    
                    self.missRate += 1
                    if target != 'center':
                        self.outMissRate += 1
                return 'W' # W = Wrong acquired (capitalized w)

    def determine_reset(self, game_state):
        """ determine whether we need a reset or not """
        
        # currently if game_state is capitalized character it means trial end condition 
        # notably T=timeout H=hit W=wrongHit

        # update cumulative active time if the task involves calculating bitrate
        if self.showTextMetrics == 'b':
            if game_state.isupper():
                self.activeTimeRecordingEnd = time.time()
                self.activeTimeRecordingLength = self.activeTimeRecordingEnd - self.activeTimeRecordingStart
                if self.trialCount > self.skipFirstNtrials:
                    self.cumulativeActiveTime += self.activeTimeRecordingLength
            

        return game_state.isupper()

    def determine_cursor_color(self, targetHit):
        
        if self.gamify and targetHit is not None and self.GamifyClass.hideAllTargets:
            return self.cursorColors['g']
        if self.variable4CursorColor['delayedPhase']:
            return self.cursorColors['lr']
        elif self.variable4CursorColor['clicked']:
            return self.cursorColors['b'] # (136,233,100) blue
        elif self.variable4CursorColor['copilotControl']:
            if self.tickTimer < self.inactiveTickLength + self.delayedTickLength:
                return self.cursorColors['r']
            return self.cursorColors['w']
        elif self.variable4CursorColor['usingNonLinearVel']:
            return self.cursorColors['o'] # (242,167,71) orange     
        return self.defaultCursorColor

    def draw_arrow(self, surf, color, start, end, width, filled=False):
        d = end - start
        if (d == 0).all():
            return
        u = d/np.sqrt(np.sum(d**2))
        u90 = np.array([-u[1], u[0]])
        verts = []
        verts.append(0.5*width*u90 + start)
        verts.append(0.5*width*u90 + start + 0.7*d)
        verts.append(0.5*width*u90 + start + 0.7*d + 0.5*width*u90)
        verts.append(end)
        verts.append(-0.5*width*u90 + start + 0.7*d - 0.5*width*u90)
        verts.append(-0.5*width*u90 + start + 0.7*d)
        verts.append(-0.5*width*u90 + start)
        if filled:
            pygame.draw.polygon(surf, color, verts, 0) # target arrow
        else:
            pygame.draw.polygon(surf, color, verts, self.targetBorderL) # target arrow
        # pygame.draw.line(self.window, (0, 0, 0), start + 0.5*d + width*0.5*u, start + 0.5*d - width*0.5*u)
        # pygame.draw.line(self.window, (0, 0, 0), start + 0.5*d + width*0.5*u90, start + 0.5*d - width*0.5*u90)
        # pygame.draw.line(self.window, (255, 0, 0), start + 0.5*d + width*0.5*np.array([1.0, 0.0]), start + 0.5*d - width*0.5*np.array([1.0, 0]), 3)
        # pygame.draw.line(self.window, (255, 0, 0), start + 0.5*d + width*0.5*np.array([0.0, 1.0]), start + 0.5*d - width*0.5*np.array([0.0, 1.0]), 3)
        return

    #def draw_square()

    def draw_state(self, cursorPos, targetHit, target, softmax, etc={}, secondCursorPos=None, secondSoftmax=None, replayTimeElapsed=None):
        # etc is any thing that needs to be drawn on the board. normally it is an empty {}
        
        # reset image
        self.screen.fill((51,51,51))

        if self.useColorBackground:
            self.window.blit(self.generatedImage['colorBackground'], -self.windowSize*0.5)
        else:
            self.window.fill((0,0,0))
        
        bounding_box_points = np.array([[0.0, 0.0], [2*self.cursorBias[0] - 2, 0], [2*self.cursorBias[0] - 2, 2*self.cursorBias[1] - 2], [0, 2*self.cursorBias[1] - 2]])
        bounding_box_points = [self.squareScale*(point - self.cursorBias) + self.cursorBias for point in bounding_box_points]
        pygame.draw.lines(self.window, (90, 90, 90), True, bounding_box_points, 2*self.targetBorderL) # bounding box rectangle

        # calculate rotation and antirotation matrix based on self.render_angle (degrees)
        rotation_matrix = np.array([[np.cos(self.render_angle/180*np.pi), -np.sin(self.render_angle/180*np.pi)], [np.sin(self.render_angle/180*np.pi), np.cos(self.render_angle/180*np.pi)]])
        antirotation_matrix = np.array([[np.cos(self.render_angle/180*np.pi), np.sin(self.render_angle/180*np.pi)], [-np.sin(self.render_angle/180*np.pi), np.cos(self.render_angle/180*np.pi)]])

        # draw heatmap
        if self.showHeatmap and self.heatmap is not None:
            self.heatmap.draw_state(self.window)
             
        # draw softmax bar
        if self.showVelocity:
            if self.hasCopilot:
                vel = self.TaskCopilotObservation.softmax_to_vel(softmax) * np.array((1,-1)) # * 0.5
            else:
                vel = self.softmax_to_vel(softmax) * np.array((1,-1))
            velLine = np.array((np.zeros(2),vel)) * self.L // 4 + self.L // 2
            pygame.draw.lines(self.window, (20,100,20), True, velLine, 2*self.targetBorderL) # green. softmax

            if 'copilot_contribution' in self.etc:
                
                
                vel_cop = self.etc['copilot_contribution'] * np.array((1,-1))
                if self.TaskCopilotAction.copilot_alpha is None: alpha = self.etc['copilot_output_alpha'][0]
                else: alpha = self.TaskCopilotAction.copilot_alpha
                
                a = alpha/2+0.5 #copilot
                b = 1-a #decoder

                # white = copilot + decoder
                line = np.array((np.zeros(2),(vel*b+vel_cop*a))) * self.L // 4 + self.L // 2
                lineColor = (200,200,200) 
                pygame.draw.lines(self.window, lineColor, True, line, 2*self.targetBorderL) # target rectangle
                
                # green = decoder
                velLine = np.array((np.zeros(2),vel*b)) * self.L // 4 + self.L // 2
                pygame.draw.lines(self.window, (20,200,20), True, velLine, 2*self.targetBorderL) # green. softmax

                # red = copilot
                line = np.array((vel*b,vel*b+vel_cop*a)) * self.L // 4 + self.L // 2
                lineColor = (200,20,20) # red. copilot
                pygame.draw.lines(self.window, lineColor, True, line, 2*self.targetBorderL) # target rectangle



        elif self.showSoftmax:
            softmaxBarColor = np.ones((5,3)) * 40
            chosen = np.argmax(softmax)
            if self.showColorSoftmax: 
                softmaxBarColor[chosen] = self.softmaxBarArgColor
                if target is not None: 
                    targetPos,targetSize = self.targetsInfo[target]
                    if self.is_correct_softmax(softmax,cursorPos,targetPos,targetSize):
                        softmaxBarColor[chosen] = self.softmaxBarCorrColor
            
            # first softmax
            if (self.gamify or self.softmaxStyle == "cross") and len(softmax) == 5:
                
                pygame.draw.rect(self.window, softmaxBarColor[0],(self.softmaxCrossImgPosL + self.softmaxBarImgSizei * np.array([1-softmax[0],0]),self.softmaxBarImgSizei * np.array([softmax[0],1])),0)
                pygame.draw.rect(self.window, softmaxBarColor[1],(self.softmaxCrossImgPosR,self.softmaxBarImgSizei * np.array([softmax[1],1])),0) 
                pygame.draw.rect(self.window, softmaxBarColor[2],(self.softmaxCrossImgPosU + self.softmaxBarImgSize * np.array([0,1-softmax[2]]),self.softmaxBarImgSize * np.array([1,softmax[2]])),0)
                pygame.draw.rect(self.window, softmaxBarColor[3],(self.softmaxCrossImgPosD,self.softmaxBarImgSize * np.array([1,softmax[3]])),0)
                pygame.draw.circle(self.window, softmaxBarColor[4], self.softmaxCrossImgPosS, self.L * 0.1 * softmax[4], 0)
                
            else:
                if len(softmax) == 5:
                    pygame.draw.rect(self.window, softmaxBarColor[0],(self.softmaxBarImgPosA + self.softmaxBarImgSize * np.array([0,1-softmax[0]]),self.softmaxBarImgSize * np.array([1,softmax[0]])),0)
                    pygame.draw.rect(self.window, softmaxBarColor[1],(self.softmaxBarImgPosB + self.softmaxBarImgSize * np.array([0,1-softmax[1]]),self.softmaxBarImgSize * np.array([1,softmax[1]])),0)
                    pygame.draw.rect(self.window, softmaxBarColor[2],(self.softmaxBarImgPosC + self.softmaxBarImgSize * np.array([0,1-softmax[2]]),self.softmaxBarImgSize * np.array([1,softmax[2]])),0)
                    pygame.draw.rect(self.window, softmaxBarColor[3],(self.softmaxBarImgPosD + self.softmaxBarImgSize * np.array([0,1-softmax[3]]),self.softmaxBarImgSize * np.array([1,softmax[3]])),0)
                    pygame.draw.rect(self.window, softmaxBarColor[4],(self.softmaxBarImgPosE + self.softmaxBarImgSize * np.array([0,1-softmax[4]]),self.softmaxBarImgSize * np.array([1,softmax[4]])),0)
                elif len(softmax) == 2:
                    pygame.draw.rect(self.window, softmaxBarColor[0],(self.softmaxBarImgPosB + self.softmaxBarImgSize * np.array([0,1-softmax[0]]),self.softmaxBarImgSize * np.array([1,softmax[0]])),0)
                    pygame.draw.rect(self.window, softmaxBarColor[1],(self.softmaxBarImgPosD + self.softmaxBarImgSize * np.array([0,1-softmax[1]]),self.softmaxBarImgSize * np.array([1,softmax[1]])),0)
        
            # second softmax
            if secondSoftmax is not None:
                softmaxBarColor = np.ones((5,3)) * 60
                chosen = np.argmax(secondSoftmax)
                if self.showColorSoftmax: 
                    softmaxBarColor[chosen] = self.softmaxBarArgColor * 1.5
                    if target is not None: 
                        targetPos,targetSize = self.targetsInfo[target]
                        if self.is_correct_softmax(softmax,cursorPos,targetPos,targetSize):
                            softmaxBarColor[chosen] = self.softmaxBarCorrColor * 1.5

                if (self.gamify or self.softmaxStyle == "cross") and len(softmax) == 5:
                    
                    pygame.draw.rect(self.window, softmaxBarColor[0],(self.softmaxCrossImgPosL + self.softmaxBarImgSizei * np.array([1-secondSoftmax[0],0]),self.softmaxBarImgSizei * np.array([secondSoftmax[0],0.5])),0)
                    pygame.draw.rect(self.window, softmaxBarColor[1],(self.softmaxCrossImgPosR,self.softmaxBarImgSizei * np.array([secondSoftmax[1],0.5])),0) 
                    pygame.draw.rect(self.window, softmaxBarColor[2],(self.softmaxCrossImgPosU + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[2]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[2]])),0)
                    pygame.draw.rect(self.window, softmaxBarColor[3],(self.softmaxCrossImgPosD,self.softmaxBarImgSize * np.array([0.5,secondSoftmax[3]])),0)
                    pygame.draw.circle(self.window, softmaxBarColor[4], self.softmaxCrossImgPosS, self.L * 0.1 * secondSoftmax[4], 0)
                    
                else:
                    if len(softmax) == 5:
                        pygame.draw.rect(self.window, softmaxBarColor[0],(self.softmaxBarImgPosA + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[0]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[0]])),0)
                        pygame.draw.rect(self.window, softmaxBarColor[1],(self.softmaxBarImgPosB + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[1]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[1]])),0)
                        pygame.draw.rect(self.window, softmaxBarColor[2],(self.softmaxBarImgPosC + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[2]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[2]])),0)
                        pygame.draw.rect(self.window, softmaxBarColor[3],(self.softmaxBarImgPosD + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[3]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[3]])),0)
                        pygame.draw.rect(self.window, softmaxBarColor[4],(self.softmaxBarImgPosE + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[4]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[4]])),0)
                    elif len(softmax) == 2:
                        pygame.draw.rect(self.window, softmaxBarColor[0],(self.softmaxBarImgPosB + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[0]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[0]])),0)
                        pygame.draw.rect(self.window, softmaxBarColor[1],(self.softmaxBarImgPosD + self.softmaxBarImgSize * np.array([0,1-secondSoftmax[1]]),self.softmaxBarImgSize * np.array([0.5,secondSoftmax[1]])),0)
        


        # stop state
        if target == None:
            pygame.draw.rect(self.window, (250,10,10),(self.stopImgPos,self.stopImgSize),0)
        # still state
        elif type(self.targetsPos[target]) == str and  self.targetsPos[target] == "still":
            pygame.draw.circle(self.window, (200, 200, 200), self.cursorBias, 120 / 1000 * self.L, 8)
        # other state
        else:
            if self.usePizza:

                # pizza added
                radius = self.pizza['radius']
                phase = self.pizza['phase']
                slices = self.pizza['slices']
                arcWidth = int(self.L/1000*24)
                imgRadius = self.L / 2 * radius + 0.5*arcWidth # add arcWidth to account for width of arc. More closely aligns with acquisition region.
                pizzaArea = (self.cursorBias - imgRadius, (imgRadius * 2,imgRadius * 2))
                sliceAngle = 2*np.pi / slices
                mag = np.linalg.norm(cursorPos)
                perceivable = mag > radius #* 0.1

                # cursor Angle
                cursorAngle = np.arctan2(cursorPos[1],cursorPos[0]) - phase
                cursorAngle = (cursorAngle + np.pi * 2) % (np.pi * 2)
                chosenSlice = int((cursorAngle) // sliceAngle) 
                correctSlice = int(target[1:])

                for i in range(slices):
                    if i == chosenSlice and perceivable:
                        pygame.draw.arc(self.window, (128, 128, 64), pizzaArea, i*sliceAngle+0.03+phase, (i+1)*sliceAngle-0.03+phase, arcWidth)
                    else:
                        pygame.draw.arc(self.window, (128, 128, 128), pizzaArea, i*sliceAngle+0.03+phase, (i+1)*sliceAngle-0.03+phase, arcWidth)
                    
                # correct pizza indicator
                pygame.draw.arc(self.window, (100, 200, 100), pizzaArea, correctSlice*sliceAngle+0.03+phase, (correctSlice+1)*sliceAngle-0.03+phase, arcWidth)

                # target light up added here as well. selected has different color
                    
            elif self.hideAllTargets: pass
            # draw targets:
            elif not self.styleChange:
                if self.promptArrow:
                    arrowStart = self.cursorBias

                    if target is not None: 
                        targetPos, targetSize = self.targetsInfo[target]
                    
                    cursor_to_target = targetPos - cursorPos
                    #if np.abs(cursor_to_target[0]) < 0.2:
                    #    cursor_to_target[0] *= 0.01
                    #if np.abs(cursor_to_target[1]) < 0.2:
                    #    cursor_to_target[1] *= 0.01
                    snap_arrow = True
                    if snap_arrow:
                        cursor_to_target_angle = np.arctan2(cursor_to_target[1], cursor_to_target[0])*180/np.pi
                        cursor_to_target_angle = cursor_to_target_angle % 360
                        angle_threshold = 22.5
                        if cursor_to_target_angle >= 0 and cursor_to_target_angle <= angle_threshold:
                            cursor_to_target[1] = 0
                        elif cursor_to_target_angle >= 360 - angle_threshold:
                            cursor_to_target[1] = 0
                        elif cursor_to_target_angle >= 90 - angle_threshold and cursor_to_target_angle <= 90 + angle_threshold:
                            cursor_to_target[0] = 0
                        elif cursor_to_target_angle >= 180 - angle_threshold and cursor_to_target_angle <= 180 + angle_threshold:
                            cursor_to_target[1] = 0
                        elif cursor_to_target_angle >= 270 - angle_threshold and cursor_to_target_angle <= 270 + angle_threshold:
                            cursor_to_target[0] = 0
                    distance_l1 = np.sum(np.abs(cursor_to_target))
                    u = cursor_to_target/np.maximum(distance_l1, 1e-8)
                    arrow_length_au = 0.2*distance_l1 + 0.04 # note: 0.4 at distance of 1.8.
                    #arrow_length_au = 0.4 - arrow_length_au # uncomment this to use long arrow == close to target.
                    arrowEnd = self.cursorBias + self.cursorBias[0]*arrow_length_au*(u * [1, -1]) # flip y bc of display
                    #arrowOffset = 0.5*(arrowStart - arrowEnd)
                    arrowOffset = 0.7*(arrowStart - arrowEnd)
                    self.draw_arrow(self.window, (127, 127, 127), arrowStart + arrowOffset, arrowEnd + arrowOffset, 0.05*self.cursorBias[0], filled=True)
                if self.showAllTarget:
                    # reorder render so that targetHit shows up on top, then target, then all the rest of the targets
                    order = list(self.targetsImgPos.keys())
                    if target in order:
                        order.remove(target)
                        order.append(target)
                    if targetHit in order:
                        order.remove(targetHit)
                        order.append(targetHit)
                    #for name, (leftTop, _, size) in self.targetsImgPos.items():
                    for name in order:
                        (leftTop, _, size) = self.targetsImgPos[name]
                        # square target
                        targetColor = self.targetWrongColor
                        targetBorderL = self.targetBorderL
                        if name == target: 
                            targetColor = self.targetDesiredColor #(254,71,91)
                            targetBorderL = self.targetDesiredBorderL
                            if self.gamify: targetBorderL = self.targetBorderL
                        if name == targetHit:
                            if target == targetHit:
                                targetColor = (100, 200, 100)
                            else:
                                targetColor = self.targetHoldColor
                        #pygame.draw.rect(self.window, targetColor, (leftTop, size), targetBorderL)

                        if self.usePolygonTargets:
                            polyVerts = np.array(self.targetsInfo[name][1])
                            #print(polyVerts.shape, polyVerts)
                            polyVerts = polyVerts*[1, -1]*self.cursorBias[0]
                            polyVerts = [tuple(antirotation_matrix@(np.array(vert))) for vert in polyVerts]
                            polyVerts = [self.squareScale*np.array(vert) + self.cursorBias for vert in polyVerts]
                        else:
                            polyVerts = [leftTop, (leftTop[0], leftTop[1] + size[1]), (leftTop[0] + size[0], leftTop[1] + size[1]), (leftTop[0] + size[0], leftTop[1])]
                            polyVerts = [tuple(antirotation_matrix@(np.array(vert) - self.cursorBias) + self.cursorBias) for vert in polyVerts]
                            polyVerts = [self.squareScale*(vert - self.cursorBias) + self.cursorBias for vert in polyVerts]
                        pygame.draw.lines(self.window, targetColor, True, polyVerts, 2*self.targetBorderL) # target rectangle

                        if name in self.targetsWord:
                            text = self.targetsWord[name]
                            if text == 'math': text = self.multiply_equation
                            elif text == 'Subtract': text = self.subtract_equation
                            elif text == 'Word Association': text = "#"+self.word_association

                            action_text_color = (192, 192, 192)
                            action_text = self.pygameActionFont.render(text, True, action_text_color)
                            text_width = action_text.get_width()
                            text_height = action_text.get_height()
                            text_shift = np.array((text_width,text_height))/2
                            text_center = (leftTop+size/2)
                            text_center = antirotation_matrix@(text_center - self.cursorBias) + self.cursorBias
                            text_center = self.squareScale*(text_center - self.cursorBias) + self.cursorBias

                            self.window.blit(action_text, text_center-text_shift)
                            #self.window.blit(action_text,(leftTop+size/2)-text_shift)

                else:
                    leftTop, _, size = self.targetsImgPos[target]
                    targetColor = self.targetHoldColor if targetHit == target else self.targetDesiredColor
                    polyVerts = [leftTop, (leftTop[0], leftTop[1] + size[1]), (leftTop[0] + size[0], leftTop[1] + size[1]), (leftTop[0] + size[0], leftTop[1])]

                    if not self.centerCorrectTarget:
                        if self.usePolygonTargets:
                            pass
                        else:
                            polyVerts = [tuple(antirotation_matrix@(np.array(vert) - self.cursorBias) + self.cursorBias) for vert in polyVerts]
                    else:
                        offset = self.cursorBias - (leftTop + 0.5*size)
                        polyVerts = [vert + offset for vert in polyVerts]
                    polyVerts = [self.squareScale*(vert - self.cursorBias) + self.cursorBias for vert in polyVerts]
                    #polyVerts = [(int(item[0]), int(item[1])) for item in polyVerts]
                    #pygame.draw.polygon(self.window, targetColor, polyVerts, self.targetBorderL) # target rectangle
                    pygame.draw.lines(self.window, targetColor, True, polyVerts, self.targetBorderL) # target rectangle
                    #pygame.draw.rect(self.window, targetColor, (leftTop, size), self.targetBorderL) # old target rectangle
                    #print(leftTop, self.windowSize, size, self.cursorBias)

                    name = target
                    if name in self.targetsWord:
                        text = self.targetsWord[name]
                        if 'default' in self.targetsWord:
                            text = self.targetsWord['default']

                        if text == 'math': text = self.multiply_equation
                        elif text == 'Subtract': text = self.subtract_equation
                        elif text == 'Word Association': text = "#"+self.word_association

                        action_text_color = (192, 192, 192)
                        action_text = self.pygameActionFont.render(text, True, action_text_color)
                        text_width = action_text.get_width()
                        text_height = action_text.get_height()
                        text_shift = np.array((text_width,text_height))/2
                        text_center = (leftTop+size/2)
                        text_center = antirotation_matrix@(text_center - self.cursorBias) + self.cursorBias
                        text_center = self.squareScale*(text_center - self.cursorBias) + self.cursorBias

                        if self.centerCorrectTarget:
                            text_center = self.cursorBias + np.array([0.0, 0.5*size[1]]) - np.array([0.0, text_height])
                        self.window.blit(action_text, text_center-text_shift)
            else:
                # if style change
                if self.showAllTarget:
                    for name, (leftTop, _, size) in self.imgTargetsImgPos.items():
                        # circle target
                        
                        targetColor = (71, 71, 69); targetImgName = 0
                        if name == target: targetColor = (124, 221, 99); targetImgName = 1
                        if name == targetHit: targetColor = (49, 50, 174); targetImgName = 2

                        # pygame.draw.circle(self.window,targetColor, leftTop+size/2, 20)
                        self.window.blit(self.pygameImage[name][targetImgName], leftTop)
                else:
                    leftTop, _, size = self.imgTargetsImgPos[target]
                    targetColor = (49, 50, 174) if targetHit == target else (124, 221, 99)
                    targetImgName = 2 if targetHit == target else 1
                    # pygame.draw.circle(self.window,targetColor, leftTop+size/2, 20)
                    self.window.blit(self.pygameImage[target][targetImgName], leftTop)

                    if 'default' in self.targetsWord:
                        text = self.targetsWord['default']
                        if text == 'math': text = self.multiply_equation
                        elif text == 'Subtract': text = self.subtract_equation
                        elif text == 'Word Association': text = "#"+self.word_association
                        (leftTop, _, size) = self.subtractImgPos
                        text = self.subtract_equation
                        action_text_color = (192, 192, 192)
                        action_text = self.pygameActionFont.render(text, True, action_text_color)
                        text_width = action_text.get_width()
                        text_height = action_text.get_height()
                        text_shift = np.array((text_width,text_height))/2
                        self.window.blit(action_text,(leftTop+size/2)-text_shift)

            

            # find image coordinate pos of cursor assuming cursorPos range from +-1
            #cursorImagePos = np.array([cursorPos[0], -cursorPos[1]])
            cursorImagePos = (rotation_matrix@cursorPos)*[1, -1]
            cursorImagePos = cursorImagePos * self.cursorMagnitude + self.cursorBias
            cursorImagePos = self.squareScale*(cursorImagePos - self.cursorBias) + self.cursorBias


            # draw cursor
            if self.styleChange:
                # pygame.draw.circle(self.window, (200, 200, 200), cursorImagePos, self.styleCursorRadius / 1000 * self.L, 0)
                self.window.blit(self.pygameImage['cursor'], cursorImagePos-self.cursorImgSize/2)
            else:
                if self.hideCursor:
                    pass
                else:
                    pygame.draw.circle(self.window, self.cursorColor, cursorImagePos, self.cursorRadius / 1000 * self.L, 0)
                    if secondCursorPos is not None:
                        secondCursorPos = np.array([secondCursorPos[0],-secondCursorPos[1]])
                        pygame.draw.circle(self.window, self.secondCursorColor, secondCursorPos * self.cursorMagnitude + self.cursorBias, self.cursorRadius / 1000 * self.L, 0)


            # draw etc
            for k,v in etc.items():
                if len(v) < 3: continue
                itemImagePos = np.array([v[0][0], -v[0][1]])
                itemRadius = v[1]
                itemColor = v[2]
                itemImagePos = itemImagePos * self.cursorMagnitude + self.cursorBias

                # draw cursor
                pygame.draw.circle(self.window, itemColor, itemImagePos, itemRadius / 1000 * self.L, 0)
            
            if self.drawFixationCross:
                pygame.draw.line(self.window, (255, 140, 238), self.cursorBias - self.cursorBias*[0.05, 0.0], self.cursorBias + self.cursorBias*[0.05, 0.0], 3)
                pygame.draw.line(self.window, (255, 140, 238), self.cursorBias - self.cursorBias*[0.0, 0.05], self.cursorBias + self.cursorBias*[0.0, 0.05], 3)



            

        # add text (prepare text content)
        elapsed = time.time() - self.startTime
        if replayTimeElapsed is not None: elapsed = replayTimeElapsed
        timeMetricText = 'Time Elapsed'
        if self.showCountDown: 
            elapsed = max(0,self.sessionLength-elapsed)
            timeMetricText = '       Time Left'
        min = int(elapsed // 60)
        sec = int(elapsed % 60)
        total = self.missRate + self.hitRate
        acc = "___" if total == 0 else  "{:.1f}".format(self.hitRate / total * 100) 
        if min < 10: min = "0"+str(min)
        if sec < 10: sec = "0"+str(sec)

        hitRateText = self.hitRate
        try:
            if self.ignoreCorrectTarget:
                hitRateText = 'N/A'
        except:
            pass

        if self.showTextMetrics == 'a': # show accuracy metric
            text = f"Hit: {hitRateText}     Miss: {self.missRate}     {timeMetricText}: {min}:{sec}     Acc: {acc}%"
        elif self.showTextMetrics == 'b': # show bitrate metric
            if self.cumulativeActiveTime == 0: bitRate = 0
            else: bitRate = self.nBit * (self.hitRate - self.missRate) / self.cumulativeActiveTime
            text = f"Hit: {hitRateText}     Miss: {self.missRate}              Time: {min}:{sec}     BitRate: {'%s' % float('%.3g' % bitRate)}"
        else: # show total hit (default)
            text = f"Hit: {hitRateText}         Total: {total}                      {timeMetricText}: {min}:{sec}"
        name_surface = self.pygameFont.render(text, True, (255, 255, 255))
        self.screen.blit(name_surface,self.textImgPos)

        
        # add progress bar
        if self.tickTimer > self.inactiveTickLength:
            barRemain = 1
            if self.tickTimer > self.inactiveTickLength + self.delayedTickLength:
                barRemain = 1-((self.tickTimer - (self.inactiveTickLength + self.delayedTickLength)) / self.activeTickLength)
            barcolor = (230,230, 230) if self.tickTimer >= self.inactiveTickLength + self.graceTickLength else (230,79, 90)
            pygame.draw.rect(self.screen, barcolor, self.barPos * np.array([1,1,barRemain,1]), 0)

        # update screen
        if self.backgroundMoves:
            if 'cursorImagePos' in locals():
                backgroundPadding = self.cursorBias-cursorImagePos
                colorWindowDupPadding = -cursorImagePos
            else:
                backgroundPadding = np.zeros(2)
                colorWindowDupPadding = np.zeros(2)
            if self.useColorBackground:
                self.windowDup.blit(self.generatedImage['colorBackground'], colorWindowDupPadding)
            else:
                self.windowDup.fill((0,0,0))
            self.windowDup.blit(self.window, backgroundPadding)
            self.screen.blit(self.windowDup,self.windowPadding)
        else:
            self.screen.blit(self.window,self.windowPadding)
        if self.renderStatWindow:
            self.update_stats_window(self.statWindow)
            self.screen.blit(self.statWindow,self.statWindoPadding)

        # draw gamify related content
        if self.gamify: self.GamifyClass.draw_state(self.screen)

        pygame.display.flip()

    def update_stats_window(self,statWindow):
        statWindow.fill((40,40,40))

        # to avoid division by zero
        TrialTime = 1 if self.stats['TrialTime'] == 0 else self.stats['TrialTime'] 
        totalAction = np.copy(self.stats['TotalAction']) 
        CorrectAction = self.stats['CorrectAction'].astype(float)
        CorrectAction[totalAction==0] = np.nan
        totalAction[totalAction==0] = 1

        # correctness is shown as rate
        CorrectRate = CorrectAction / totalAction

        texts = [f"Stats",
                 "H Correct Softmax: {:.2f}".format(self.stats['CorrectSoftmax'] / TrialTime),
                 "S Correct Softmax: {:.2f}".format(0 if self.stats['NonStillTTimer'] == 0 else self.stats['CorrectSoftmax']/self.stats['NonStillTTimer']), # soft correct softmax (ignore s)
                 f"L Streak:  {self.stats['Streak'][0]}",
                 f"R Streak:  {self.stats['Streak'][1]}",
                 f"U Streak:  {self.stats['Streak'][2]}",
                 f"D Streak:  {self.stats['Streak'][3]}",
                 f"S Streak:  {self.stats['Streak'][4]}",
                 f"Second Hitrate: {self.stats['Second Hitrate']}",
                #  f"L Total:  {self.stats['TotalAction'][0]}",
                #  f"R Total:  {self.stats['TotalAction'][1]}",
                #  f"U Total:  {self.stats['TotalAction'][2]}",
                #  f"D Total:  {self.stats['TotalAction'][3]}",
                #  f"S Total:  {self.stats['TotalAction'][4]}",
                #  "L Correct:  {:.2f}".format(CorrectRate[0]),
                #  "R Correct:  {:.2f}".format(CorrectRate[1]),
                #  "U Correct:  {:.2f}".format(CorrectRate[2]),
                #  "D Correct:  {:.2f}".format(CorrectRate[3]),
                #  "S Correct:  {:.2f}".format(CorrectRate[4]),
                 f"Trial Time:   {self.stats['TrialTime']}"
                 
                 ]
        for i,text in enumerate(texts):
            stat_surface = self.statFont.render(text, True, (255, 255, 255))
            statWindow.blit(stat_surface,self.statImgPos+self.statNewlineImgPos*i)


    def determine_taskidx(self,target,cursorPos):
        
        # while in delayed phase and before, it should always return -1 even if target is specified
        if self.tickTimer < self.inactiveTickLength + self.delayedTickLength:
            return self.target2state_task[None]
        
        return self.target2state_task[target]
    
    def create_output(self,target,softmax):
        """ 
            create output for update using what we know so far
            also update some of the pastCursorPos, pastCursorVel, pastTargetPos etc for other uses
        """

        # prepare current target pos and target size
        currTargetPos, currTargetSize = self.targetsInfo[target] if target in self.targetsImgPos else (np.array([np.NaN,np.NaN]),np.array([np.NaN,np.NaN]))

        # prepare detail
        self.currCursorVel = self.cursorPos - self.pastCursorPos
        currCursorAcc = self.currCursorVel - self.pastCursorVel
        self.pastCursorPos = np.copy(self.cursorPos)
        self.pastCursorVel = np.copy(self.currCursorVel)
        self.pastTargetPos = np.copy(currTargetPos) # used in gamify
        self.pastTargetSize = np.copy(currTargetSize) # used in gamify
        detail = {'vel':self.currCursorVel, 'acc':currCursorAcc, 'softmax':softmax}
        return currTargetPos, currTargetSize, detail

    def addTargets(self,extra_targets):
        # used by rl agent (add target to yaml if you want to add target on live run)
        # extra_argets must be {target_str: [np,array([x,y]),np,array([w,h])]...} info
        self.targetsInfo.update(extra_targets)
        self.targetsPos.update({k:v[0] for k,v in extra_targets.items()})
        usedNum = max(self.target2state_task.values())
        for targetName in list(extra_targets.keys()):
            usedNum += 1
            self.target2state_task[targetName] = usedNum # unknown target
            self.desiredTargetList.append(targetName)
        self.update_window_size_constants()
        return self.targetsPos

    def useTargetYamlFile(self,extra_targets_yaml):
        yamlpath = f"SJtools/copilot/targets/{extra_targets_yaml}"
        with open(yamlpath) as yaml_file:
            yaml_data = yaml.load(yaml_file, Loader=Loader)
            
        # store target info
        self.targetsInfo = {}
        self.targetsPos = {}
        for k,v in yaml_data["targetsInfo"].items():
            if type(v) is str: self.targetsInfo[k] = v
            else: self.targetsInfo[k] = [np.array(v[0]),np.array(v[1])]
        for k,v in self.targetsInfo.items():
            self.targetsPos[k] = v if type(v) is str else v[0]
        self.desiredTargetList = [k for k in self.targetsInfo]

        # give each target id
        usedNum = 0
        for targetName in self.desiredTargetList:
            usedNum += 1
            self.target2state_task[targetName] = usedNum # unknown target
        self.update_window_size_constants()
        

    def getCurrentCursorPos(self):
        return self.cursorPos

    def getTruthPredictorTargeterPos(self, target):
        if target is None:
            return np.array([0,0])
        if type(self.targetsPos[target]) == str:
            return np.array([0,0])
        return self.targetsPos[target]

    def changeTargetSize(self,targetSize):
        # used by copilot training to change targetsize
        # sets targetsize and imgtarget size
        
        for k,v in self.targetsInfo.items():
            if type(v) is not str: 
                pos, size = v
                size[0] = targetSize
                size[1] = targetSize

        self.update_window_size_constants()

    def setAttributeFromCurriculum(self,curriculum):

        # target_size
        if 'target_size' in curriculum: self.changeTargetSize(curriculum['target_size']['current'])
        if 'hold_time' in curriculum: self.holdTimeThres = curriculum['hold_time']['current']
        if 'dwell_time' in curriculum: self.dwellTimeThres = curriculum['dwell_time']['current']
        if 'wrong_tolerance' in curriculum: self.wrongTolerance = curriculum['wrong_tolerance']['current']

    def get_env_obs(self, softmax, action=None):
        """
        for use regarding copilot only
        returns observation demanded by the env
        the dimension and content of obs get decided by the model's parameter
        i.e) cursorpos, softmax

        action of copilot causes changes in game state then softmax from user
        these changed gamestate then softmax should be delivered to the user
        action_t0 -> gamestate_t0 -> softmax_t1 -> action_t1
        which means the given softmax should depend on previous game state,
        and should not effect this game state. this function should just return
        gamestate of previous tick
        online: should be called before copilot.predict, and before task.update
        training: should be called with previous softmax in the arg, and returned as obs
        """

        # choose only the relevant obs
        obs, etc = self.TaskCopilotObservation.get_env_obs(action)        
        if etc is not None: self.add_etc(*etc)
        return obs

    def save_to_observation(self, softmax, cursorPos, holdTime, target, targetHit):
        # save everything that needs to be sent to the observation for copilot
        # called once at initialization
        # then called at the end of every round
        if self.noCopilot: return
        
        self.TaskCopilotObservation.softmax = softmax
        self.TaskCopilotObservation.cursorPos = cursorPos
        self.TaskCopilotObservation.correctTargetPos = self.getTruthPredictorTargeterPos(target)
        self.TaskCopilotObservation.targetCursorOnPos = self.getTruthPredictorTargeterPos(targetHit)
        self.TaskCopilotObservation.massPos = self.TaskCopilotAction.massPos
        self.TaskCopilotObservation.game_state = self.game_state
        self.TaskCopilotObservation.detail = self.detail
        self.TaskCopilotObservation.timeRemain = 1-((self.tickTimer - (self.inactiveTickLength + self.delayedTickLength)) / self.activeTickLength)
        self.TaskCopilotObservation.emptyData = (self.reset) or (self.target == None)
        self.TaskCopilotObservation.holdTime = holdTime
        self.TaskCopilotObservation.holdTimeThres = self.holdTimeThres
        
        # print("softmax", self.TaskCopilotObservation.softmax)
        # print("cursorPos", self.TaskCopilotObservation.cursorPos)
        # print("correctTargetPos", self.TaskCopilotObservation.correctTargetPos)
        # print("game_state", self.TaskCopilotObservation.game_state)
        # print("detail", self.TaskCopilotObservation.detail)
        # print("timeRemain", self.TaskCopilotObservation.timeRemain)
        # print("emptyData", self.TaskCopilotObservation.emptyData)
        # print("holdTime", self.TaskCopilotObservation.holdTime)
        # print()


    def add_etc(self, tag, target_pos, color = (18, 105, 196), radius=10):
        # purpose: things to draw additionally on the board
        # etc is dictionary {} that SJ_4_directions_task can accept and will render on the board
        # its basic format is {name:[pos,radius,color]}
        # add_etc maybe called anytime during an update (can also be used as public method call)
        # at each end of update, etc is reset to {}
        
        # color =  (18, 105, 196), (200, 100, 255) (255,205,0)
        self.etc[tag] = [target_pos, radius, color]
    
    def reset_etc(self):
        self.etc = {}

    def generate_next_word_association(self):
        return random.choice(self.word_association_list)
        
    def getDriftCompensation(self):

        # no drift compensation during inactive or delayed
        if self.tickTimer < self.inactiveTickLength + self.delayedTickLength:
            return np.array([0.,0])

        # average drift compensation
        if self.driftCompensation == 'average':
            # collect current directions, pile them up and use it to find average per tick
            return -self.driftTotal / self.driftN
        elif type(self.driftCompensation) is np.ndarray:
            return self.driftCompensation
        else:
            raise Exception("something wrong with get drift compensation")

    def isHeadedInCorrectDirection(self,decoder_output,cursorPos,targetPos):
        desired_direction = targetPos - cursorPos
        decoded_direction = self.decoder_direction(decoder_output)
        
        v1 = desired_direction
        v2 = decoded_direction
        
        # find cos angle
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 
        return angle < np.pi/2

    def get_kf_detail(self,reset):
        """ return kf related details to output from update_kf function """
        adapt = self.enableKfAdapt if self.tickTimer > self.KfSyncDelayTickLength else False
        if self.enableKfSyncInternal == -1:
            # should only init this once.
            self.numCompletedBlocks = 0 if self.skipFirstNtrials <= 0 else -1
            self.enableKfSyncInternal = self.enableKfSync
        
        # increment numCompletedBlocks first
        if reset and len(self.nextTargetBucket) == 0:
                self.integerBlocks = True
                self.numCompletedBlocks += 1
                # There might be a bug for self.numCompletedBlocks when center targets are repeated upon failure, but I'm not sure.
                print('numCompletedBlocks incremented')
        else:
            self.integerBlocks = False
        
        if self.numCompletedBlocks < self.numInitialEvaluationBlocks:
            self.enableKfSyncInternal = False
        else:
            try:
                self.enableKfSyncInternalFlag
            except:
                # i.e. only run this ONCE.
                # Otherwise it would override the syncLength checking that occurs later on.
                self.enableKfSyncInternal = self.enableKfSync
                self.enableKfSyncInternalFlag = True
                print('warning: set self.enableKfSyncInternal to', self.enableKfSyncInternal)
        sync = self.enableKfSyncInternal if reset else False
        #print('warning', self.numCompletedBlocks, self.enableKfSyncInternal, sync, reset)
    
        # is this a good location?
        # after assigning sync so that it is synced one last time.
        if len(self.nextTargetBucket) == 0 and reset:
            print('warning: bucket empty')
            try:
                if (time.time() - self.startTime > self.syncLength):
                    self.enableKfSyncInternal = False
                    print('warning: enableKfSyncInternal set to false')
            except:
                pass
        
        return [adapt, sync]
    
    def complete_restart(self):
        """ do all things here to completely restart """
        self.init_variables() # internally sets self.reset to True, so will redetermine targets
    
    def log_to_sharedmemory(self, namespace):
        var_info = [
            (self.sessionLength, 'sessionLength'),
            (self.activeLength, 'activeLength'),
            (self.cursorVel, 'cursorVel'),
            (self.ignoreWrongTarget, 'ignoreWrongTarget'),
            (self.cursorMoveInCorretDirectionOnly, 'cursorMoveInCorretDirectionOnly'),
            (self.assistValue, 'assistValue'),
            (self.assistMode, 'assistMode'),
            (self.softmaxThres, 'softmaxThres'),
            (self.holdTimeThres, 'holdTimeThres'),
            (self.kfCopilotAlpha, 'kfCopilotAlpha'),
            (self.hitRate, 'hitRate'),
            (self.missRate, 'missRate'),
            (self.timeoutRate, 'timeoutRate'),
            (self.trialCount, 'trialCount'),
            (self.render_angle, 'render_angle'),
            (self.numCompletedBlocks, 'numCompletedBlocks'),
            (self.enableKfSyncInternal, 'enableKfSyncInternal'),
        ]
        for info in var_info:
            v_self = info[0]
            try:
                v = namespace[info[1]]
                if isinstance(v_self, str):
                    v[0:len(v_self)] = np.frombuffer(v_self.encode(), dtype='int8')
                else:
                    v[:] = v_self
            except Exception as e:
                print(f'failed to save {info[1]} to sharedmemory', e)
        return
    
    def get_my_decoder_output(self,decoder_output):
            
        if self.pygameKeyPressed == pygame.K_LEFT:
            print("manual input: left")
            direction = np.array([-1.,0])
        elif self.pygameKeyPressed == pygame.K_RIGHT:
            print("manual input: right")
            direction = np.array([1.,0])
        elif self.pygameKeyPressed == pygame.K_DOWN:
            print("manual input: down")
            direction = np.array([0.,-1])
        elif self.pygameKeyPressed == pygame.K_UP:
            print("manual input: up")
            direction = np.array([0.,1])
        else:
            return decoder_output
        
        decoder_output = self.tempSynSoft.correctSoftmax(np.zeros(2), direction, self.defaultTargetSize,0) #, CSvalue=1.0, stillCS=1.0)
        return decoder_output


    def softmax_to_vel(self,softmax):
        vx = softmax[1]-softmax[0]
        vy = softmax[2]-softmax[3]
        vel = np.array([vx,vy])
        vel = np.clip(vel,-1,1)
        return vel

    def set_pizza_param(self, pizzaParam):
        
        self.usePizza = True
        if len(self.desiredTargetList) > pizzaParam['slices']: self.nextTargetBucket = []
        self.desiredTargetList = []
        for i in range(pizzaParam['slices']):
            name = 'p'+str(i)
            self.desiredTargetList.append(name)
            self.targetsInfo[name] = name
            self.targetsPos[name] = name
            self.target2state_task[name] = i
            
        # other misc paramter that needs setting
        self.centerIn = False



# initiate constructor if called by main
initTask = True
copilotReady = False
targetPredReady = False

try: params # we have params
except NameError: initTask = False


if initTask: 

    # start task
    copilotReady = "copilot_path" in params

    if copilotReady:
        task = SJ_4_directions(params,copilotYamlParam=params["copilot_path"])
    else:
        task = SJ_4_directions(params)
    # boot copilot

    if copilotReady: 
        if task.copilotInfo["model"] == "PPO":
            print("is a PPO model")
            trial_per_episode = 1
            trial_per_episode_counter = 0
            copilot_state = None
            copilot = PPO.load(params["copilot_path"])
        elif task.copilotInfo["model"] == "RecurrentPPO":
            trial_per_episode = 1
            trial_per_episode_counter = 0
            if 'reset' in task.TaskCopilotAction.copilotYamlParam:
                trial_per_episode = task.TaskCopilotAction.copilotYamlParam['reset'].get('trial_per_episode',1)
            copilot_state = None
            copilot = RecurrentPPO.load(params["copilot_path"])


        specialOption = None # for copilot


# numpy print option
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
