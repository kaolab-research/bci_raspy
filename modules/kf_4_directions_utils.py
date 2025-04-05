
import pygame
import numpy as np
import time
import random
import yaml
import os
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pyautogui # to prpoperly use this, goto system preference, security, accessibility, check terminal


class DisplayConstants():
    def __init__(self,showAllTarget=True,showSoftmax=True,showHeatmap=True,hideMass=True,showNonCopilotCursor=True,softmaxStyle='normal'):

        # constants
        self.screenSize = np.array([700,700]) # w,h
        self.cursorRadius = 10
        self.showAllTarget = showAllTarget
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
        self.showTextMetrics = None # *logic
        self.showCountDown = False
        self.showSoftmax = showSoftmax
        self.showColorSoftmax = False
        self.showStats = False
        self.showHeatmap = showHeatmap
        self.showColorCursor = True
        self.hideMass = hideMass
        self.hideCom = False
        self.showNonCopilotCursor = showNonCopilotCursor
        self.styleChange = False
        self.styleChangeBallSize = 0.5
        self.styleChangeCursorSize = np.array([0.2, 0.2])
        self.softmaxStyle = softmaxStyle
        self.useCmBoard = False #uses centimeter board
        self.cmBoardDetail = {}
        self.cmBoardScreenInfo = {}
        self.hideAllTargets = False
        self.fullScreen = False

        # unchanging constants
        self.softmaxBarColor = np.array((40,40,40))
        self.softmaxBarArgColor = np.array((80,40,40))
        self.softmaxBarCorrColor = np.array((40,80,40))


        # constants from non-display
        self.defaultTargetSize = np.array([0.2,0.2])
        self.targetsInfo = { # max is 1 [pos, size]
            'left' :[np.array([-0.85, 0  ]), np.array([0.2,0.2])],   #   0 : left,
            'right':[np.array([ 0.85, 0  ]), np.array([0.2,0.2])],   #   1 : right,
            'up'   :[np.array([ 0  , 0.85]), np.array([0.2,0.2])],   #   2 : up,
            'down' :[np.array([ 0  ,-0.85]), np.array([0.2,0.2])],   #   3 : down,
            'still':'still',
        }
        self.centerIn = False
        self.gamify = False
        self.target2state_task = {
            "left":  0,
            "right": 1,
            "up":    2,
            "down":  3,
            "still": 4,
            "center": ord('c'),
            "random": ord('r'),
            None:    -1, # when there is no target (i.e stop)
        }
        self.state_task2target = {v:k for k,v in self.target2state_task.items()}
        
    def initParamWithYaml(self,params):

        self.fullScreen = params['display'].get('fullScreen',False)
        self.screenSize = np.array(params['display']['screenSize']) # w,h
        self.objScale = params['display']['objScale']
        self.cursorRadius = params['display']['cursorRadius']
        self.targetsWord = params['display'].get('targetsWord',{})
        self.showAllTarget = params['display']['showAllTarget']
        self.targetWrongColor =  params['display'].get('targetWrongColor',self.targetWrongColor)
        self.targetDesiredColor = params['display'].get('targetDesiredColor',self.targetDesiredColor)
        self.targetHoldColor = params['display'].get('targetHoldColor',self.targetHoldColor)
        self.showTextMetrics = params['display'].get('showTextMetrics', None)
        self.showCountDown = params['display'].get('showCountDown', False)
        self.showSoftmax = params['display']['showSoftmax']
        self.showColorSoftmax = params['display'].get('showColorSoftmax',False)
        self.showStats = params['display'].get('showStats',False)
        self.showHeatmap = params['display'].get('showHeatmap',self.showHeatmap)
        self.showColorCursor = params['display'].get('showColorCursor',False)
        self.hideCom = params['display'].get("hideCom",self.hideCom)
        self.hideMass = params['display'].get("hideMass",self.hideMass)
        self.styleChange = params['display']['styleChange']
        self.styleChangeBallSize = params['display'].get('styleChangeBallSize',self.styleChangeBallSize)
        self.styleChangeCursorSize = np.array(params['display'].get('styleChangeCursorSize',self.styleChangeCursorSize))
        self.softmaxStyle = params['display'].get('softmaxStyle',self.softmaxStyle)
        self.useCmBoard = ('cmBoardDetail' in params['display']) and params['display']["useCmBoard"]
        self.cmBoardDetail = params['display']["cmBoardDetail"] if self.useCmBoard else {} 
        
        if self.useCmBoard: self.cmBoardScreenInfo = self.getScreenInfo()

        # read from yaml (needed it every trial basis likely)
        self.gamify = params.get('gamify',False)
        self.defaultTargetSize = np.array(params['defaultTargetSize'])
        self.centerIn = params.get('centerIn',False)
        self.targetsInfo = {}
        for k,v in params['targetsInfo'].items():
            if type(v) is str: self.targetsInfo[k] = v
            else: self.targetsInfo[k] = [np.array(v[0]),np.array(v[1])]
        if self.centerIn: self.targetsInfo['center'] = [np.array([ 0., 0.]), self.defaultTargetSize] # center target information is needed
        self.target2state_task = params.get('target2state_task',self.target2state_task)
        self.target2state_task.update({"center": ord('c'),"random": ord('r'),})
        self.state_task2target = {v:k for k,v in self.target2state_task.items()}
    
    def getScreenInfo(self):
        """ very awful way to get screen info regarding pixel and cm but this is the best I have currently. 
        it will cause the Tk inter window to appear for split second at the beginning of trial """
        root = tk.Tk()
        info = {}
        info["width_px"] = root.winfo_screenwidth()
        info["height_px"] = root.winfo_screenheight()
        info["width_cm"] = self.display.cmBoardDetail["actualScreenSize"][0] #root.winfo_screenmmwidth() / 10
        info["height_cm"] = self.display.cmBoardDetail["actualScreenSize"][1] #root.winfo_screenmmheight() / 10
        info["width_cm2px"] = info["width_px"] / info["width_cm"]
        info["height_cm2px"] = info["height_px"] / info["height_cm"]
        root.after(0, root.destroy)
        root.mainloop()
        return info
        


class PygameDisplay:

    def __init__(self,pygame:pygame, separatedPygame:bool, identity:str, displayConstants:DisplayConstants, render=True):

        self.pygame = pygame
        self.separatedPygame = separatedPygame
        self.identity = identity # task or pygame
        self.display = displayConstants
        self.render = render

        # init pygame
        if not self.separatedPygame or self.identity == 'pygame':
            if self.display.fullScreen: self.screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
            else: self.screen = pygame.display.set_mode(self.display.screenSize, pygame.RESIZABLE)

           # init pygame
            if self.render:
                pygame.font.init()

        # constants
        self.generatedImage = {} # cached pygame image

        # states (variables / draw_state origianl parameters)
        self._cursorPos = np.array([0., 0.],dtype=np.float32)
        self._targetHit = np.array([-1],dtype=np.int32)
        self._target = np.array([-1],dtype=np.int32) # -1 means None use number to substitue target
        self._softmax = np.array([0., 0., 0., 0., 0.],dtype=np.float32)
        self._etc = {} # cannot be handled by separatedPygame
        self._secondCursorPos = np.array([0., 0.],dtype=np.float32) 
        self._secondSoftmax = np.array([0., 0., 0., 0., 0.],dtype=np.float32)
        self._replayTimeElapsed = np.array([0],dtype=np.float32)

        # states (variables)
        self._render_angle = np.array([0],dtype=np.float32) # float32 # in degrees. Used to render the cursor and target at different angles.
        self._dragArrowStartDefined = np.array([False],dtype=bool) # bool
        self._dragArrowStart = np.array([0., 0.],dtype=np.float32) # float32
        self._dragArrowEnd = np.array([0., 0.],dtype=np.float32) # float32
        self._progressBarColor = np.array([230, 230, 230,],dtype=np.int32) # maybe replaceable with char
        self._progressBarRemain = np.array([0],dtype=np.float32) # float32

        self._hitCount = np.array([0],dtype=np.int32) # int
        self._missCount = np.array([0],dtype=np.int32) # int
        self.totalTrial = 0 #DEP
        self._trialAccuracy = np.array([-1],dtype=np.float32) # float32 #-1 means "___"
        self._timeMetricText = np.array([ord('l')],dtype=np.int8) # char l:left, e:elapsed
        self._min = np.array([0],dtype=np.int8)
        self._sec = np.array([0],dtype=np.int8)
        self._bitRate = np.array([0],dtype=np.float32) #float32

        # to be sent to task
        self._screenSize = np.array([0, 0],dtype=np.int32)

        # once function to be run by standalone pygame process
        # if True, then pygame must run them and set them back to False
        self._once_insert_random_target_pos = np.array([False],dtype=bool)
        self._once_update_window_size_constants = np.array([False],dtype=bool)
        self._once_update_action_text = np.array([False],dtype=bool)
        self._onceRandomTargetPos = np.array([0., 0.],dtype=np.float32) # argument for _once_insert_random_target_pos
        

        # text for targets
        self.multiply_equation = ''
        self.subtract_equation = ''
        self.word_association = ''
        with open('./asset/text/random_word.txt', "r") as my_file:
            self.word_association_list = my_file.read().split()

        # unknown at the time
        self.yamlName = None
        self.yamlLastModified = None

    def init(self):

        self.cursorPos = np.array([0., 0.],dtype=np.float32)
        self.targetHit = None
        self.target = None # -1 means None use number to substitue target
        self.softmax = np.array([0., 0., 0., 0., 0.],dtype=np.float32)
        self.etc = {} # cannot be handled by separatedPygame
        self.secondCursorPos = np.array([0., 0.],dtype=np.float32) 
        self.secondSoftmax = np.array([0., 0., 0., 0., 0.],dtype=np.float32)
        self.replayTimeElapsed = 0.

        self.cursorPos = np.array([0., 0.])
        self.targetHit = None
        self.target = None
        self.softmax = np.array([0., 0., 0., 0., 0.])
        self.secondCursorPos = np.array([0., 0.])
        self.secondSoftmax = np.array([0., 0., 0., 0., 0.])
        self.replayTimeElapsed = 0
        self.render_angle = 0
        self.dragArrowStartDefined = False
        self.dragArrowStart = np.array([0., 0.])
        self.dragArrowEnd = np.array([0., 0.])
        self.progressBarColor = np.array([230, 230, 230,],dtype=np.int32)
        self.progressBarRemain = 0
        self.hitCount = 0
        self.missCount = 0
        self.totalTrial = 0
        self.trialAccuracy = -1
        self.timeMetricText = 'l'
        self.min = 0
        self.sec = 0
        self.bitRate = 0.
        self.screenSize = np.array([0, 0])

        self._once_insert_random_target_pos[:] = False
        self._once_update_window_size_constants[:] = False
        self._once_update_action_text[:] = False

    @property
    def cursorPos(self): return self._cursorPos
    @cursorPos.setter
    def cursorPos(self,value): self._cursorPos[:] = value
    @property
    def targetHit(self): return self.display.state_task2target[self._targetHit[0]]
    @targetHit.setter
    def targetHit(self,value): self._targetHit[:] = self.display.target2state_task[value]
    @property
    def target(self): 
        return self.display.state_task2target[self._target[0]]
    @target.setter
    def target(self,value): 
        self._target[:] = self.display.target2state_task[value]
    @property
    def softmax(self): return self._softmax
    @softmax.setter
    def softmax(self,value): self._softmax[:] = value
    @property
    def secondCursorPos(self): 
        if np.isnan(self._secondCursorPos[0]): return None
        return self._secondCursorPos
    @secondCursorPos.setter
    def secondCursorPos(self,value): 
        if value is None: self._secondCursorPos[:] *= np.NAN
        else: self._secondCursorPos[:] = value
    @property
    def secondSoftmax(self): return self._secondSoftmax
    @secondSoftmax.setter
    def secondSoftmax(self,value): self._secondSoftmax[:] = value
    @property
    def etc(self): return self._etc
    @etc.setter
    def etc(self,value): self._etc = value
    @property
    def replayTimeElapsed(self): return self._replayTimeElapsed[0]
    @replayTimeElapsed.setter
    def replayTimeElapsed(self,value): self._replayTimeElapsed[:] = value
    @property
    def render_angle(self): return self._render_angle[0]
    @render_angle.setter
    def render_angle(self,value): self._render_angle[:] = value
    @property
    def dragArrowStartDefined(self): return self._dragArrowStartDefined[0]
    @dragArrowStartDefined.setter
    def dragArrowStartDefined(self,value): self._dragArrowStartDefined[:] = value
    @property
    def dragArrowStart(self): return self._dragArrowStart
    @dragArrowStart.setter
    def dragArrowStart(self,value): self._dragArrowStart[:] = value
    @property
    def dragArrowEnd(self): return self._dragArrowEnd
    @dragArrowEnd.setter
    def dragArrowEnd(self,value): self._dragArrowEnd[:] = value
    @property
    def progressBarColor(self): return self._progressBarColor
    @progressBarColor.setter
    def progressBarColor(self,value): self._progressBarColor[:] = value
    @property
    def progressBarRemain(self): return self._progressBarRemain[0]
    @progressBarRemain.setter
    def progressBarRemain(self,value): self._progressBarRemain[:] = value
    @property
    def hitCount(self): return self._hitCount[0]
    @hitCount.setter
    def hitCount(self,value): self._hitCount[:] = value
    @property
    def missCount(self): return self._missCount[0]
    @missCount.setter
    def missCount(self,value): self._missCount[:] = value
    @property
    def trialAccuracy(self): return self._trialAccuracy[0]
    @trialAccuracy.setter
    def trialAccuracy(self,value): self._trialAccuracy[:] = value
    @property
    def timeMetricText(self): return chr(self._timeMetricText[0])
    @timeMetricText.setter
    def timeMetricText(self,value): self._timeMetricText[:] = ord(value)
    @property
    def min(self): return self._min[0]
    @min.setter
    def min(self,value): self._min[:] = value
    @property
    def sec(self): return self._sec[0]
    @sec.setter
    def sec(self,value): self._sec[:] = value
    @property
    def bitRate(self): return self._bitRate[0]
    @bitRate.setter
    def bitRate(self,value): self._bitRate[:] = value
    @property
    def screenSize(self): return self._screenSize
    @screenSize.setter
    def screenSize(self,value): self._screenSize[:] = value
    @property
    def onceRandomTargetPos(self): return self._onceRandomTargetPos
    @onceRandomTargetPos.setter
    def onceRandomTargetPos(self, value): self._onceRandomTargetPos[:] = value

    def convertToSharedMemory(self, cursorPos, targetHit, target, softmax, secondCursorPos, secondSoftmax, replayTimeElapsed, render_angle, dragArrowStartDefined, dragArrowStart, dragArrowEnd, progressBarColor, progressBarRemain, hitCount, missCount, trialAccuracy, timeMetricText, minute, second, bitRate, 
                              screenSize, once_insert_random_target_pos, once_update_action_text, once_update_window_size_constants, onceRandomTargetPos):

        self._cursorPos = cursorPos
        self._targetHit = targetHit
        self._target = target
        self._softmax = softmax
        self._secondCursorPos = secondCursorPos
        self._secondSoftmax = secondSoftmax
        self._replayTimeElapsed = replayTimeElapsed
        self._render_angle = render_angle
        self._dragArrowStartDefined = dragArrowStartDefined
        self._dragArrowStart = dragArrowStart
        self._dragArrowEnd = dragArrowEnd
        self._progressBarColor = progressBarColor
        self._progressBarRemain = progressBarRemain
        self._hitCount = hitCount
        self._missCount = missCount
        self._trialAccuracy = trialAccuracy
        self._timeMetricText = timeMetricText
        self._min = minute
        self._sec = second
        self._bitRate = bitRate
        self._screenSize = screenSize
        self._once_insert_random_target_pos = once_insert_random_target_pos
        self._once_update_action_text = once_update_action_text
        self._once_update_window_size_constants = once_update_window_size_constants
        self._onceRandomTargetPos = onceRandomTargetPos

        self.init()

    def update_window_size_constants(self,humanInduced=False):

        if self.separatedPygame: 
            if self.identity == 'task': 
                self._once_update_window_size_constants[:] = True
                return
            elif self.identity == 'pygame': 
                pass

        if self.display.useCmBoard:
            # use cmboard size and resize the pygame
            self.L = self.display.cmBoardDetail["gameBoardSize"][0] * self.display.cmBoardScreenInfo["width_cm2px"]
            
            if self.display.cmBoardDetail["fullSizeScreen"]:
                self.display.screenSize = np.array((self.display.cmBoardScreenInfo["width_px"],self.display.cmBoardScreenInfo["height_px"]))
            else:
                self.display.screenSize = (int(self.L / 0.9),)*2
                
            if humanInduced:
                self.display.screenSize = np.array(self.screen.get_size())
            else:
                self.pygame.display.set_mode(self.display.screenSize, self.pygame.RESIZABLE)
        
        else:
            # use human adjustable size
            if self.render: self.display.screenSize = np.array(self.screen.get_size())
            self.L = min(self.display.screenSize) * 0.9

        self.windowSize = np.array([self.L,self.L])
        windowPaddingX = (self.display.screenSize[0]-self.L)/2
        windowPaddingY = (self.display.screenSize[1]-self.L)/2
        self.windowPadding = np.array((windowPaddingX,windowPaddingY))
        self.statWindoPadding = np.array((windowPaddingX*1.1+self.L, windowPaddingY))
        if self.render: 
            self.window = self.pygame.Surface(self.windowSize)
        if self.render:
            self.renderStatWindow = self.display.showStats and (windowPaddingX > windowPaddingY * 5)
            if self.renderStatWindow:
                self.statWindow = self.pygame.Surface(np.array((windowPaddingX * 0.8,self.L)))

        # reset cursor constants
        self.cursorBias = self.windowSize / 2
        self.cursorMagnitude = self.windowSize / 2

        # name: [leftTop(x,y), bottomRight(x,y), size(w,h)]
        self.targetsImgPos = {} # depends on targetsPos / used to determine target
        for k,info in self.display.targetsInfo.items():
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

        if self.display.styleChange:
            self.imgTargetsImgPos = {k:(TL+(size*(1-self.display.styleChangeBallSize)/2),BR-(size*(1-self.display.styleChangeBallSize)/2),size*self.display.styleChangeBallSize) for k,(TL,BR,size) in self.targetsImgPos.items()}
            self.imgDefaultTargetImgSize = self.L * self.display.defaultTargetSize / 2 * self.display.styleChangeBallSize # image of target, it's image size not actual size
            corrospondence = {
                "cursor":"greyball",
                "target":["greyball","greenball","blueball"],
                }
            # print(self.imgTargetsImgPos)
            self.cursorImgSize = self.L * self.display.styleChangeCursorSize / 2
            self.pygameImage = self.initPygameImage(corrospondence,self.imgTargetsImgPos,self.imgDefaultTargetImgSize,self.cursorImgSize)

        # stop sign
        self.stopSize = np.array([200,200])
        self.stopImgSize = self.L * self.display.objScale * self.stopSize
        self.stopImgPos = self.windowSize / 2 - self.stopImgSize / 2

        # font size
        self.fontSize = 60
        self.fontImgSize = int(self.L * self.display.objScale * self.fontSize)
        statFontImgSize = int(windowPaddingX * self.display.objScale * 90)
        self.statImgPos = np.ones(2) * int(statFontImgSize * 0.3)
        self.statNewlineImgPos = np.array((0,statFontImgSize))
        if self.render: 
            self.pygameFont = self.pygame.font.SysFont(None, self.fontImgSize)
            self.statFont = self.pygame.font.SysFont(None, statFontImgSize)

        # action font size
        self.actionTextFontSize = 60
        self.actionFontImgSize = int(self.L * self.display.objScale * self.actionTextFontSize)
        if self.render: self.pygameActionFont = self.pygame.font.SysFont(None, self.actionFontImgSize)


        textImgPosY = windowPaddingY-self.L * 0.05
        # if windowPaddingY-windowPaddingX <= 0: textImgPosY = 1

        self.textImgPos = (windowPaddingX,textImgPosY)

        # bar pos
        # barPosY = self.display.screenSize[1] - windowPaddingY + windowPaddingY / 3
        barPosY = self.windowSize[1] + windowPaddingY + self.L * 0.05 / 2
        barSizeX = self.windowSize[0]
        self.barPos = np.array((windowPaddingX,barPosY,barSizeX, self.L * 0.05 / 7))

        # softmax pos
        self.softmaxBarSize = np.array([60,400])
        softmaxBarPadding = np.array([25,400])
        self.softmaxBarImgSize = self.L * self.display.objScale * self.softmaxBarSize
        self.softmaxBarImgSizei = np.array((self.softmaxBarImgSize[1],self.softmaxBarImgSize[0]))
        softmaxBarImgsoftmaxBarPadding = self.L * self.display.objScale * softmaxBarPadding
        self.softmaxBarImgPosC = self.windowSize / 2 - self.softmaxBarImgSize / 2
        self.softmaxBarImgPosA = self.softmaxBarImgPosC - (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([2.0,0])
        self.softmaxBarImgPosB = self.softmaxBarImgPosC - (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([1.0,0])
        self.softmaxBarImgPosD = self.softmaxBarImgPosC + (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([1.0,0])
        self.softmaxBarImgPosE = self.softmaxBarImgPosC + (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([2.0,0])

        self.softmaxCrossImgPosL = self.windowSize / 2 - self.softmaxBarImgSizei * np.array((1,0.5)) - np.array((0.5,0)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosR = self.windowSize / 2 - self.softmaxBarImgSizei * np.array((0,0.5)) + np.array((0.5,0)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosU = self.windowSize / 2 - self.softmaxBarImgSize * np.array((0.5,1)) - np.array((0,0.5)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosD = self.windowSize / 2 - self.softmaxBarImgSize * np.array((0.5,0)) + np.array((0,0.5)) * self.softmaxBarImgSize[0]
        self.softmaxCrossImgPosS = self.windowSize / 2


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
    
    def update_action_text(self):

        if self.separatedPygame: 
            if self.identity == 'task': 
                self._once_update_action_text[:] = True
                return
            elif self.identity == 'pygame':
                pass

        self.multiply_equation = f"{random.randint(20,40)}x{random.randint(6,50)}=?"
        primeList = [3,4,5,6,7,8,9,11] # [3,5,7,11,13,17,19,23]
        self.subtract_equation = f"{random.randint(100,300)}  -  {random.choice(primeList)}"
        self.word_association = random.choice(self.word_association_list)

    def insertRandomTargetPos(self,pos,size):

        # put information in targetsInfo
        self.display.targetsInfo.update({"random" : [pos,size]})

        # depending on task or pygame
        if self.separatedPygame: 
            if self.identity == 'task': 
                self.onceRandomTargetPos = pos
                self._once_insert_random_target_pos[:] = True
                return
            elif self.identity == 'pygame':
                pass

        # pygame must update all the img related parts
        realTargetSize = self.L * size / 2
        pos = np.array([pos[0],-pos[1]]) # need to flip y axis in image
        topLeft = self.cursorBias + self.cursorMagnitude * pos - realTargetSize / 2
        bottomRight = self.cursorBias + self.cursorMagnitude * pos + realTargetSize / 2
        self.targetsImgPos = {"random": (topLeft,bottomRight, realTargetSize)}
        if self.display.styleChange:
            self.imgTargetsImgPos = {k:(TL+(size*(1-self.display.styleChangeBallSize)/2),BR-(size*(1-self.display.styleChangeBallSize)/2),size) for k,(TL,BR,size) in self.targetsImgPos.items()}
            self.pygameImage["random"] = self.pygameImage["default"]


    def draw_state(self):
        # etc is any thing that needs to be drawn on the board. normally it is an empty {}

        if self.separatedPygame and self.identity == 'task': return

        # take out state variables (param)
        cursorPos = self.cursorPos
        targetHit = self.targetHit
        target = self.target
        softmax = self.softmax
        etc = self.etc
        secondCursorPos = self.secondCursorPos
        secondSoftmax = self.secondSoftmax
        render_angle = self.render_angle

        # other state variables
        dragArrowStartDefined = self.dragArrowStartDefined
        dragArrowStart = self.dragArrowStart
        dragArrowEnd = self.dragArrowEnd
        progressBarRemain = self.progressBarRemain
        progressBarColor = self.progressBarColor

        hitCount = self.hitCount
        missCount = self.missCount
        totalTrial = self.hitCount + self.missCount
        timeMetricText = self.timeMetricText
        trialAccuracy = self.trialAccuracy
        min = self.min
        sec = self.sec
        bitRate = self.bitRate

        # reset image
        self.screen.fill((51,51,51))
        self.window.fill((0,0,0))

        # calculate rotation and antirotation matrix based on self.render_angle (degrees)
        rotation_matrix = np.array([[np.cos(render_angle/180*np.pi), -np.sin(render_angle/180*np.pi)], [np.sin(render_angle/180*np.pi), np.cos(render_angle/180*np.pi)]])
        antirotation_matrix = np.array([[np.cos(render_angle/180*np.pi), np.sin(render_angle/180*np.pi)], [-np.sin(render_angle/180*np.pi), np.cos(render_angle/180*np.pi)]])
        
        # # draw heatmap
        # if self.display.showHeatmap and self.heatmap is not None:
        #     self.heatmap.draw_state(self.window)
             
        # draw softmax bar
        if self.display.showSoftmax:
            softmaxBarColor = np.ones((5,3)) * 40
            chosen = np.argmax(softmax)
            if self.display.showColorSoftmax: 
                softmaxBarColor[chosen] = self.display.softmaxBarArgColor
                if target is not None: 
                    targetPos,targetSize = self.display.targetsInfo[target]
                    if self.is_correct_softmax(softmax,cursorPos,targetPos,targetSize):
                        softmaxBarColor[chosen] = self.display.softmaxBarCorrColor
            
            # first softmax
            if (self.display.gamify or self.display.softmaxStyle == "cross") and len(softmax) == 5:
                
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
                if self.display.showColorSoftmax: 
                    softmaxBarColor[chosen] = self.display.softmaxBarArgColor * 1.5
                    if target is not None: 
                        targetPos,targetSize = self.display.targetsInfo[target]
                        if self.is_correct_softmax(softmax,cursorPos,targetPos,targetSize):
                            softmaxBarColor[chosen] = self.display.softmaxBarCorrColor * 1.5

                if (self.display.gamify or self.display.softmaxStyle == "cross") and len(softmax) == 5:
                    
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
        elif type(self.display.targetsInfo[target]) == str and  self.display.targetsInfo[target] == "still":
            pygame.draw.circle(self.window, (200, 200, 200), self.cursorBias, 120 / 1000 * self.L, 8)
        # other state
        else:
            if self.display.hideAllTargets: pass
            # draw targets:
            elif not self.display.styleChange:
                if self.display.showAllTarget:
                    for name, (leftTop, _, size) in self.targetsImgPos.items():
                        # square target
                        targetColor = self.display.targetWrongColor
                        targetBorderL = self.display.targetBorderL
                        if name == target: 
                            targetColor = self.display.targetDesiredColor #(254,71,91)
                            targetBorderL = self.display.targetDesiredBorderL
                            if self.display.gamify: targetBorderL = self.display.targetBorderL
                        if name == targetHit: 
                            targetColor = self.display.targetHoldColor
                        pygame.draw.rect(self.window, targetColor, (leftTop, size), targetBorderL)

                        if name in self.display.targetsWord:
                            text = self.display.targetsWord[name]
                            if text == 'math': text = self.multiply_equation
                            elif text == 'Subtract': text = self.subtract_equation
                            elif text == 'Word Association': text = "#"+self.word_association

                            action_text = self.pygameActionFont.render(text, True, (255, 255, 255))
                            text_width = action_text.get_width()
                            text_height = action_text.get_height()
                            text_shift = np.array((text_width,text_height))/2
                            self.window.blit(action_text,(leftTop+size/2)-text_shift)

                else:
                    leftTop, _, size = self.targetsImgPos[target]
                    targetColor = self.display.targetHoldColor if targetHit == target else self.display.targetDesiredColor
                    polyVerts = [leftTop, (leftTop[0], leftTop[1] + size[1]), (leftTop[0] + size[0], leftTop[1] + size[1]), (leftTop[0] + size[0], leftTop[1])]
                    polyVerts = [tuple(antirotation_matrix@(np.array(vert) - self.cursorBias) + self.cursorBias) for vert in polyVerts]
                    pygame.draw.polygon(self.window, targetColor, polyVerts, self.display.targetBorderL) # target rectangle
                    #pygame.draw.rect(self.window, targetColor, (leftTop, size), self.display.targetBorderL) # old target rectangle
                    #print(leftTop, self.windowSize, size, self.cursorBias)

                    name = target
                    if name in self.display.targetsWord:
                        text = self.display.targetsWord[name]
                        if text == 'math': text = self.multiply_equation
                        elif text == 'Subtract': text = self.subtract_equation
                        elif text == 'Word Association': text = "#"+self.word_association

                        action_text = self.pygameActionFont.render(text, True, (255, 255, 255))
                        text_width = action_text.get_width()
                        text_height = action_text.get_height()
                        text_shift = np.array((text_width,text_height))/2
                        text_center = (leftTop+size/2)
                        text_center = antirotation_matrix@(text_center - self.cursorBias) + self.cursorBias
                        self.window.blit(action_text, text_center-text_shift)
            else: 
                # if style change
                if self.display.showAllTarget:
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

                    if 'default' in self.display.targetsWord:
                        text = self.display.targetsWord['default']
                        if text == 'math': text = self.multiply_equation
                        elif text == 'Subtract': text = self.subtract_equation
                        elif text == 'Word Association': text = "#"+self.word_association
                        (leftTop, _, size) = self.subtractImgPos
                        text = self.subtract_equation
                        action_text = self.pygameActionFont.render(text, True, (255, 255, 255))
                        text_width = action_text.get_width()
                        text_height = action_text.get_height()
                        text_shift = np.array((text_width,text_height))/2
                        self.window.blit(action_text,(leftTop+size/2)-text_shift)

            

            # find image coordinate pos of cursor assuming cursorPos range from +-1
            #cursorImagePos = np.array([cursorPos[0], -cursorPos[1]])
            cursorImagePos = (rotation_matrix@cursorPos)*[1, -1]
            cursorImagePos = cursorImagePos * self.cursorMagnitude + self.cursorBias


            # draw cursor
            if self.display.styleChange:
                # pygame.draw.circle(self.window, (200, 200, 200), cursorImagePos, self.styleCursorRadius / 1000 * self.L, 0)
                self.window.blit(self.pygameImage['cursor'], cursorImagePos-self.cursorImgSize/2)
            else:
                pygame.draw.circle(self.window, self.display.cursorColor, cursorImagePos, self.display.cursorRadius / 1000 * self.L, 0)
                if secondCursorPos is not None:
                    secondCursorPos = np.array([secondCursorPos[0],-secondCursorPos[1]])
                    pygame.draw.circle(self.window, self.display.secondCursorColor, secondCursorPos * self.cursorMagnitude + self.cursorBias, self.display.cursorRadius / 1000 * self.L, 0)


            # draw etc
            for k,v in etc.items():
                itemImagePos = np.array([v[0][0], -v[0][1]])
                itemRadius = v[1]
                itemColor = v[2]
                itemImagePos = itemImagePos * self.cursorMagnitude + self.cursorBias

                # draw cursor
                pygame.draw.circle(self.window, itemColor, itemImagePos, itemRadius / 1000 * self.L, 0)


        if min < 10: min = "0"+str(min)
        if sec < 10: sec = "0"+str(sec)
        
        # add top bar
        if timeMetricText == 'e': timeMetricText = 'Time Elapsed'
        elif timeMetricText == 'l': timeMetricText = '       Time Left'
        if self.display.showTextMetrics == 'a': # show accuracy metric
            trialAccuracy = '___' if trialAccuracy < 0 else "{:.1f}".format(trialAccuracy)
            text = f"Hit: {hitCount}     Miss: {missCount}     {timeMetricText}: {min}:{sec}     Acc: {trialAccuracy}%"
        elif self.display.showTextMetrics == 'b': # show bitrate metric
            text = f"Hit: {hitCount}     Miss: {missCount}              Time: {min}:{sec}     BitRate: {'%s' % float('%.3g' % bitRate)}"
        else: # show total hit (default)
            text = f"Hit: {hitCount}         Total: {totalTrial}                      {timeMetricText}: {min}:{sec}"
        name_surface = self.pygameFont.render(text, True, (255, 255, 255))
        self.screen.blit(name_surface,self.textImgPos)

        
        # add progress bar
        pygame.draw.rect(self.screen, progressBarColor, self.barPos * np.array([1,1,progressBarRemain,1]), 0)

        # update screen
        self.screen.blit(self.window,self.windowPadding)
        # if self.renderStatWindow:
        #     self.update_stats_window(self.statWindow)
        #     self.screen.blit(self.statWindow,self.statWindoPadding)

        # draw gamify related content
        if self.display.gamify: 
            """ gamify draw_state """
            if dragArrowStartDefined:
                # draw line given by clickDrag
                pygame.draw.line(self.screen, (103, 154, 209), dragArrowStart, dragArrowEnd, width=3)
        

        pygame.display.flip()

    def standalone_process_update(self):
        """ to be run NOT by task, but by standalone pygame process """
        
        # check if any new function once function should be run
        if self._once_insert_random_target_pos[:]:
            pos = self.onceRandomTargetPos
            size = self.display.defaultTargetSize
            self.insertRandomTargetPos(pos,size)
            self._once_insert_random_target_pos[:] = False
        if self._once_update_window_size_constants[:]:
            self.update_yaml()
            self.update_window_size_constants()
            self._once_update_window_size_constants[:] = False
        if self._once_update_action_text[:]:
            self.update_action_text()
            self._once_update_action_text[:] = False

        self.draw_state()

    def update_yaml(self):

        lastModified = os.path.getmtime(self.yamlName)
        if self.yamlLastModified == lastModified: return
        self.yamlLastModified = lastModified

        yaml_file = open(self.yamlName, 'r')
        yaml_data = yaml.load(yaml_file, Loader=Loader)
        yaml_file.close()
        params = yaml_data["modules"]["SJ_4_directions"]["params"]

        self.display.initParamWithYaml(params)

class PygameKeyboardMouse:
    def __init__(self,pygame:pygame, separatedPygame:bool, identity:str):

        # all the constants that keyboard uses
        self.pygame = pygame
        self.separatedPygame = separatedPygame 
        self.identity = identity # task or pygame
        self.L = None
        self.windowPadding = None
        

        # communicated varaiables
        self.key_pressed = {} # indicates if any key is pressed (stored as dictionary, only used exclusively by copilot)
        self._pygameKeyPressed = np.array([0],dtype=np.int32) # 0 means None
        self._K_a_pressed = np.array([False],dtype=bool) # used for gamify (controls copilot alpha as human input)
        self._key_PRESSEDONCE = np.array([False],dtype=bool) # for denoisign multiclick for pygame key pressed. useful sometimes
        self._mousePos = np.array([0., 0.],dtype=np.float32)
        self._dragArrowStartDefined = np.array([False],dtype=bool)
        self._dragArrowStart = np.array([0., 0.],dtype=np.float32)
        self._dragArrowEnd = np.array([0., 0.],dtype=np.float32)
        self._pygame_once_session_ended = np.array([False],dtype=bool)
        self._pygame_once_screen_resized = np.array([False],dtype=bool)
        self._pygame_once_K_r = np.array([False],dtype=bool)
        self._pygame_once_K_SPACE = np.array([False],dtype=bool)

        self.once_varaiable_reset()

        self.init()

    def init(self):
        self.key_pressed = {} # indicates if any key is pressed (stored as dictionary, only used exclusively by copilot)
        self.pygameKeyPressed = 0 # 0 means None
        self.K_a_pressed = False # used for gamify (controls copilot alpha as human input)
        self.key_PRESSEDONCE = False # for denoisign multiclick for pygame key pressed. useful sometimes
        self.mousePos = np.array([0., 0])
        self.dragArrowStartDefined = False
        self.dragArrowStart = np.array([0., 0])
        self.dragArrowEnd = np.array([0., 0])
        self.pygame_once_session_ended = False
        self.pygame_once_screen_resized = False
        self.pygame_once_K_r = False
        self.pygame_once_K_SPACE = False

    def update_window_size_constants(self,L,windowPadding):
        self.L = L
        self.windowPadding = windowPadding

    def convertToSharedMemory(self,pygameKeyPressed,K_a_pressed,key_PRESSEDONCE,mousePos,dragArrowStartDefined,dragArrowStart,dragArrowEnd,pygame_once_session_ended,pygame_once_screen_resized,pygame_once_K_r,pygame_once_K_SPACE):
        
        self._pygameKeyPressed = pygameKeyPressed 
        self._K_a_pressed = K_a_pressed 
        self._key_PRESSEDONCE = key_PRESSEDONCE 
        self._mousePos = mousePos 
        self._dragArrowStartDefined = dragArrowStartDefined 
        self._dragArrowStart = dragArrowStart 
        self._dragArrowEnd = dragArrowEnd 
        self._pygame_once_session_ended = pygame_once_session_ended 
        self._pygame_once_screen_resized = pygame_once_screen_resized 
        self._pygame_once_K_r = pygame_once_K_r 
        self._pygame_once_K_SPACE = pygame_once_K_SPACE 

        self.init()

    def once_varaiable_reset(self):
        """ 
            resets all the 'once' variable to initial value 
            exists to be calld once every tick of logic game
        """
        self.pygame_once_session_ended = False
        self.pygame_once_screen_resized = False
        self.pygame_once_K_r = False
        self.pygame_once_K_SPACE = False

    # getter setter
    @property
    def pygameKeyPressed(self): return self._pygameKeyPressed
    @pygameKeyPressed.setter
    def pygameKeyPressed(self, value): self._pygameKeyPressed[:] = value
    @property
    def K_a_pressed(self): return self._K_a_pressed
    @K_a_pressed.setter
    def K_a_pressed(self, value): self._K_a_pressed[:] = value
    @property
    def key_PRESSEDONCE(self): return self._key_PRESSEDONCE
    @key_PRESSEDONCE.setter
    def key_PRESSEDONCE(self, value): self._key_PRESSEDONCE[:] = value
    @property
    def mousePos(self): return self._mousePos
    @mousePos.setter
    def mousePos(self, value): self._mousePos[:] = value
    @property
    def dragArrowStartDefined(self): return self._dragArrowStartDefined
    @dragArrowStartDefined.setter
    def dragArrowStartDefined(self, value): self._dragArrowStartDefined[:] = value
    @property
    def dragArrowStart(self): return self._dragArrowStart
    @dragArrowStart.setter
    def dragArrowStart(self, value): self._dragArrowStart[:] = value
    @property
    def dragArrowEnd(self): return self._dragArrowEnd
    @dragArrowEnd.setter
    def dragArrowEnd(self, value): self._dragArrowEnd[:] = value
    @property
    def pygame_once_session_ended(self): return self._pygame_once_session_ended
    @pygame_once_session_ended.setter
    def pygame_once_session_ended(self, value): self._pygame_once_session_ended[:] = value
    @property
    def pygame_once_screen_resized(self): return self._pygame_once_screen_resized
    @pygame_once_screen_resized.setter
    def pygame_once_screen_resized(self, value): self._pygame_once_screen_resized[:] = value
    @property
    def pygame_once_K_r(self): return self._pygame_once_K_r
    @pygame_once_K_r.setter
    def pygame_once_K_r(self, value): self._pygame_once_K_r[:] = value
    @property
    def pygame_once_K_SPACE(self): return self._pygame_once_K_SPACE
    @pygame_once_K_SPACE.setter
    def pygame_once_K_SPACE(self, value): self._pygame_once_K_SPACE[:] = value

    def check_pygame_event(self):

        # if separate pygame, then separate pygame will calcaulate this part for us
        if self.separatedPygame and self.identity == 'task': return

        # check for event
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.pygame_once_session_ended = True
                
            if event.type == self.pygame.VIDEORESIZE:
                # print("#######################\n#######################\n#######################\n#######################\n#######################\n#######################\n#######################\n")
                self.pygame_once_screen_resized = True
                
            # special pygame keypress recorder
            if event.type == self.pygame.KEYDOWN:
                # store key that was pressed
                self.key_pressed[self.pygame.key.name(event.key)] = True

                self.pygameKeyPressed = event.key
                if event.key == self.pygame.K_a: 
                    self.K_a_pressed = True

                # reset the game if r is pressed
                if self.pygameKeyPressed == self.pygame.K_r:
                    self.pygame_once_K_r = True

                if self.pygameKeyPressed == self.pygame.K_SPACE:
                    self.pygame_once_K_SPACE = True

            if event.type == self.pygame.KEYUP:
                # store key that was unpressed
                self.key_pressed[self.pygame.key.name(event.key)] = False

                if event.key == self.pygame.K_a:  self.K_a_pressed = False
                self.pygameKeyPressed = 0
                self.key_PRESSEDONCE = False

            if event.type == self.pygame.MOUSEMOTION:
                mousePos = (np.array(event.pos) - self.windowPadding)
                mousePos = (np.clip(mousePos,0,self.L) - self.L/2) / (self.L/2)
                mousePos[1] *= -1
                self.mousePos = mousePos


            # click and drag for mouse (update mouse position)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragArrowStart = np.array((event.pos))
                    self.dragArrowEnd = np.array((event.pos))
                    self.dragArrowStartDefined = True
                    print("pressed down",event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:            
                    self.dragArrowStartDefined = False

            if event.type == pygame.MOUSEMOTION:

                if self.dragArrowStartDefined:
                    self.dragArrowEnd = np.array((event.pos))
    

    def resetMousePos(self):
        center = np.array(pyautogui.size())//2
        pyautogui.moveTo(center[0], center[1])
        self.mousePos = np.zeros(2)


