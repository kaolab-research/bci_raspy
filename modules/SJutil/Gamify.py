
from modules.SJutil.SyntheticSoftmax import SyntheticSoftmax
import pygame
import numpy as np
import pyautogui

class GamifyClass():
    # stores everything pertaining to gamifying the SJ-4-directions
    # No Private Access: does not access any of the task's variable directly, they are always passed down
    # Ready Only Constant: none of the passed down argument is modified in this class. it is only read

    def __init__(self,gamify):
        
        self.gamify = gamify
        self.gamifiedSyntheticSoftmax = SyntheticSoftmax()
        self.dragArrowStartDefined = False
        self.mouseAlpha = 1 # alpha is always 1 unless manual alpha option chosen
        

    def windowSettingChanged(self, screenInfo):

        # run everytime yaml setting changes
        self.windowPadding = screenInfo['windowPadding']
        self.L = screenInfo['L']

        self.dragArrowStart = screenInfo['screenSize'] // 2 # for creating drag arrow when task is gamified
        self.dragArrowEnd = screenInfo['screenSize'] // 2 


    def initGamifyOption(self,gamifyOption,targetColor):
        # assumes gamifyOption is filled
        targetWrongColor = targetColor['wrong']
        targetDesiredColor = targetColor['desired']
        targetHoldColor = targetColor['hold']

        # configure target
        if 'target' in gamifyOption:
            targetOption = gamifyOption['target']
            if targetOption is not None:
                if 'hideAllTargets' in targetOption:
                    self.hideAllTargets = True
                    targetDesiredColor = targetWrongColor = targetHoldColor = (0,0,0)
                elif 'hideGoalTarget' in targetOption:
                    self.hideAllTargets = False
                    targetDesiredColor = targetWrongColor
            else:
                self.hideAllTargets = False
                targetWrongColor = (0,255, 200)
                targetDesiredColor = (235,235,235)
                targetHoldColor = (255,205,0)
        # configure softmax
        if 'softmax' in gamifyOption:
            softmaxOption = gamifyOption['softmax']
            if 'two_peak' in softmaxOption:
                self.gamifySyntheticSoftmax = True
                CSvalue = softmaxOption['two_peak']['CSvalue'] if 'CSvalue' in softmaxOption['two_peak'] else 0.7
                stillCS = softmaxOption['two_peak']['stillCS'] if 'stillCS' in softmaxOption['two_peak'] else 0.7
                self.getSyntheticSoftmax = lambda cursorPos, targetPos, targetSize : self.gamifiedSyntheticSoftmax.twoPeakSoftmax(cursorPos, targetPos, targetSize, CSvalue,stillCS)
            elif 'simple_softmax' in softmaxOption:
                self.gamifySyntheticSoftmax = True
                self.getSyntheticSoftmax = lambda cursorPos, targetPos, targetSize : self.gamifiedSyntheticSoftmax.simpleSoftmax(cursorPos, targetPos, targetSize )
            elif 'complex_softmax' in softmaxOption:
                self.gamifySyntheticSoftmax = True
                self.getSyntheticSoftmax = lambda cursorPos, targetPos, targetSize : self.gamifiedSyntheticSoftmax.complexSoftmax(cursorPos, targetPos, targetSize )
            else:
                self.gamifySyntheticSoftmax = False
        # configure mouse
        if 'mouse' in gamifyOption:
            mouseOption = gamifyOption['mouse']
            self.gamifyFollowMouse = True if 'followMouse' in mouseOption else False
            self.gamifyOnMouse = True if 'onMouse' in mouseOption else False
            self.gamifyManualAlpha = True if 'manualAlpha' in mouseOption else False
        
        # return target color to be populated into SJ task
        targetColor['wrong'] = targetWrongColor
        targetColor['desired'] = targetDesiredColor
        targetColor['hold'] = targetHoldColor
        return targetColor

    def pygameEvent(self,dragArrowStartDefined,dragArrowStart,dragArrowEnd):
        """ should run everytime pygame event runs """
        
        # click and drag for mouse (update mouse position)
        self.dragArrowStartDefined = dragArrowStartDefined
        self.dragArrowStart = dragArrowStart
        self.dragArrowEnd = dragArrowEnd


    def gamified_direction(self, direction, cursorPos, mousePos, K_a_pressed):
        
        # alpha is always 1 unless manual alpha option chosen
        self.mouseAlpha = int(K_a_pressed) if self.gamifyManualAlpha else 1
        if self.mouseAlpha == 0: 
            return direction, cursorPos

        # on Mouse control
        if self.gamifyOnMouse:
            cursorPos = mousePos
            direction = np.zeros(2)
            return direction, cursorPos
        
        # follow mouse pos control
        if self.gamifyFollowMouse:
            userDir = mousePos - cursorPos
            userNorm = np.linalg.norm(userDir)
            userThres = 0.1
            nearZero = 0.001
            if userNorm > userThres: direction = userDir / userNorm
            elif userNorm < nearZero: direction = userDir * 0
            else: direction = userDir / userNorm * (userNorm / userThres)
            return direction, cursorPos
        
        # mouse pad control
        else:
            if self.dragArrowStartDefined:         
                userDir = self.dragArrowEnd-self.dragArrowStart
                userDir[1] *= -1
                userNorm = (userDir**2).sum()**0.5
                if userNorm < 1:
                    userDir = np.zeros(2)
                else:
                    userDir = userDir / userNorm * 0.1
                    userDir = userDir * np.clip(np.exp(userNorm*0.1-3),0,10)
                direction = userDir
            else:
                direction = np.zeros(2)
            return direction, cursorPos
        

