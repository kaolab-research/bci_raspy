
# in:
#   - state_task
#   - decoder_output
#   - target_hit
# out:
#   - decoded_pos
#   - target_hit
#   - scores

import pygame
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
from SJtools.copilot.targetPredictor.model import LSTM,LSTM2,LSTMFCS,NN
import tkinter as tk


class SJ_text_classification:

    def __init__(self,params=None,render=True,useRealTime=True,showAllTarget=False,showSoftmax=False,randomInitCursorPosition=False):
        # tunable constants
        self.render = render # uses to render image or not
        self.useRealTime = useRealTime # uses time.time package for real time hold time and game ending signal
        # ^ if false uses tickLength for holdTime
        self.sessionLength = 300 # seconds
        self.screenSize = np.array([700,700]) # w,h
        self.objScale = 1/1000
        self.cursorRadius = 10
        self.styleCursorRadius = 30
        self.cursorVel = (0.015,0.015) # max is 1
        self.showAllTarget = showAllTarget
        self.ignoreWrongTarget = False
        self.centerIn = False
        self.resetCursorPos = True
        self.defaultCursorColor = (200, 200, 200)
        self.cursorColor = self.defaultCursorColor
        self.previewTimeThres = 0 # 0 is no preview, 1 is full preview
        self.fullScreen = False
        self.testedAction = { # max is 1 [pos, size]
            'left' :[np.array([-0.85, 0  ]), np.array([0.2,0.2])],   #   0 : left,
            'right':[np.array([ 0.85, 0  ]), np.array([0.2,0.2])],   #   1 : right,
            'up'   :[np.array([ 0  , 0.85]), np.array([0.2,0.2])],   #   2 : up,
            'down' :[np.array([ 0  ,-0.85]), np.array([0.2,0.2])],   #   3 : down,
            'still':'still',
        }
        self.desiredTargetList = [k for k in self.testedAction]
        # targets converted to decoder's label for training step
        self.action2state_task = {
                               "left":  0,
                               "right": 1,
                               "up":    2,
                               "down":  3,
                               "still": 4,
                               None:    -1, # when there is no target (i.e stop)
                               }
        # self.action2state_task["random"] = ord('r') # 114 means random 
        # self.action2state_task["center"] = ord('c') # center is special target 
        self.state_task2target = {v:k for k,v in self.action2state_task.items()} 

        # other variables (decoder label to velocity)
        self.decodedVel = {0:np.array([-1, 0]),   #   0 : left,
                            1:np.array([ 1, 0]),   #   1 : right,
                            2:np.array([ 0, 1]),   #   2 : up,
                            3:np.array([ 0,-1]),   #   3 : down,
                            4:np.array([ 0, 0]),   #   4 : still,
                            5:np.array([ 0, 0]),}  #   5 : rest

        self.holdTimeThres = 0.5 # seconds
        self.graceTimeThres = 0.0 # seconds
        self.softmaxThres = 0.0
        self.assistValue = 0.0
        self.assistMode = 'e' # efficient vs natural
        self.actionTextFontSize = 60

        self.textVelScale = 0.0

        self.activeLength = 10 # seconds
        self.inactiveLength = 2 # seconds
        self.displayLength = 10 # seconds (how long to show the word)
        self.showMissAccuracy = True
        self.showCountDown = False
        self.showSoftmax = showSoftmax
        self.randomInitCursorPosition = randomInitCursorPosition
        self.styleChange = False
        self.styleChangeBallSize = 0.5
        self.styleChangeCursorSize = np.array([0.2, 0.2])
        self.showProgressBar = True

        # dependent constants (should not be tuned!)
        self.episodeLength = self.inactiveLength + self.activeLength
        self.tickTimer = 0
        self.tickLength = 0.02 # each tick is assumed to be 20ms (value imported from timer.py)
        self.activeTickLength = self.activeLength / self.tickLength
        self.displayTickLength = self.displayLength / self.tickLength
        self.inactiveTickLength = self.inactiveLength / self.tickLength
        self.episodeTickLength = self.episodeLength / self.tickLength
        self.graceTickLength = self.graceTimeThres / self.tickLength

        if params is not None:
            self.initParamWithYaml(params)

        # init pygame
        if self.render:
            pygame.font.init()
            self.screen = pygame.display.set_mode(self.screenSize, pygame.RESIZABLE)

        # init variables
        self.cursorPos = np.array([0.,0.]) 
        self.heldTimeBegin = None
        self.startTime = time.time()
        self.reset = True
        self.nextTargetBucket = []
        self.assistNaturalDuration = 0
        self.assistNaturalDirection = np.array([0,0])
        self.pygameKeyPressed = None
        self.copilot_alpha = None
        self.target = None # initialy None
        self.pastCursorPos = np.zeros(2)
        self.pastCursorVel = np.zeros(2)
        self.randomTargetPos = np.zeros(2)
        self.trialCount = 0
        self.generatedImage = {} # cached pygame image
        self.cumulativeActiveTime = 0
        self.useClickState = False
        self.useNonLinearVel = False
        self.multiply_equation = ''
        self.subtract_equation = ''
        self.word_association = ''
        with open('./asset/text/random_word.txt', "r") as my_file:
            self.word_association_list = my_file.read().split()
        # self.consecutiveStillRequired = 15

        # set dependent variable constants
        self.update_window_size_constants()

    def update_window_size_constants(self):
        if self.render: self.screenSize = np.array(self.screen.get_size())

        self.L = min(self.screenSize) * 0.9
        self.windowSize = np.array([self.L,self.L])
        windowPaddingX = (self.screenSize[0]-self.L)/2
        windowPaddingY = (self.screenSize[1]-self.L)/2
        self.windowPadding = (windowPaddingX,windowPaddingY)
        if self.render: self.window = pygame.Surface(self.windowSize)

        # reset cursor constants
        self.cursorBias = self.windowSize / 2
        self.cursorMagnitude = self.windowSize / 2

        # name: [leftTop(x,y), bottomRight(x,y), size(w,h)]

        if self.styleChange:
            corrospondence = {
                "cursor":"greyball",
                "target":["greyball","greenball","blueball"],
                }
            self.cursorImgSize = self.L * self.styleChangeCursorSize / 2
            self.pygameImage = self.initPygameImage(corrospondence,self.cursorImgSize)

        # stop sign
        self.stopSize = np.array([200,200])
        self.stopImgSize = self.L * self.objScale * self.stopSize
        self.stopImgPos = self.windowSize / 2 - self.stopImgSize / 2

        # font size
        self.fontSize = 60
        self.fontImgSize = int(self.L * self.objScale * self.fontSize)
        if self.render: self.pygameFont = pygame.font.SysFont(None, self.fontImgSize)

        self.actionFontImgSize = int(self.L * self.objScale * self.actionTextFontSize)
        if self.render: self.pygameActionFont = pygame.font.SysFont(None, self.actionFontImgSize)

        textImgPosY = windowPaddingY-windowPaddingX
        if textImgPosY < 0: textImgPosY = 1

        self.textImgPos = (windowPaddingX,textImgPosY)

        # bar pos
        barPosY = self.screenSize[1] - windowPaddingY + windowPaddingY / 3
        barSizeX = self.windowSize[0]
        self.barPos = np.array((windowPaddingX,barPosY,barSizeX,windowPaddingY/10))

        # softmax pos
        self.softmaxBarSize = np.array([60,400])
        softmaxBarPadding = np.array([25,400])
        self.softmaxBarImgSize = self.L * self.objScale * self.softmaxBarSize
        softmaxBarImgsoftmaxBarPadding = self.L * self.objScale * softmaxBarPadding
        self.softmaxBarImgPosC = self.windowSize / 2 - self.softmaxBarImgSize / 2
        self.softmaxBarImgPosA = self.softmaxBarImgPosC - (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([2.0,0])
        self.softmaxBarImgPosB = self.softmaxBarImgPosC - (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([1.0,0])
        self.softmaxBarImgPosD = self.softmaxBarImgPosC + (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([1.0,0])
        self.softmaxBarImgPosE = self.softmaxBarImgPosC + (self.softmaxBarImgSize + softmaxBarImgsoftmaxBarPadding) * np.array([2.0,0])
        self.softmaxBarColor = (20,20,20)

    def initParamWithYaml(self,params):
        self.sessionLength = params['sessionLength']
        self.screenSize = np.array(params['screenSize']) # w,h
        self.actionTextFontSize = np.array(params['actionTextFontSize']) # w,h
        self.objScale = params['objScale']
        self.cursorRadius = params['cursorRadius']
        self.cursorVel = np.array(params['cursorVel'])
        self.resetCursorPos = params['resetCursorPos']

        self.testedAction = params['testedAction']
        self.desiredTargetList = [k for k in self.testedAction]

        self.decodedVel = {}
        for k,v in params['decodedVel'].items():
            self.decodedVel[k] = np.array(v)
        self.action2state_task = params['action2state_task']
        # self.action2state_task["random"] = ord('r') # 114 means random
        # self.action2state_task["center"] = ord('c') # center is special target

        self.graceTimeThres = params['graceTimeThres']
        self.softmaxThres = params['softmaxThres']
        self.activeLength = params['activeLength']
        self.displayLength = params['displayLength'] if 'displayLength' in params else self.activeLength
        self.inactiveLength = params['inactiveLength']
        self.yamlName = params['yamlName']
        self.showSoftmax = params['showSoftmax']
        self.showCountDown = params['showCountDown'] if 'showCountDown' in params else False
        self.styleChange = params['styleChange']
        self.previewTimeThres = params['previewTimeThres']
        self.fullScreen = params['fullScreen'] if 'showCountDown' in params else False
        if self.fullScreen:
            info = self.getScreenInfo()
            self.screenSize = np.array((info["width_px"],info["height_px"]))
        self.showProgressBar = not params['hideProgressBar'] if 'hideProgressBar' in params else True
            
        self.textVelScale = params.get('textVelScale', 0.0)
            

        # dependent constants (should not be tuned!)
        self.episodeLength = self.inactiveLength + self.activeLength
        self.tickTimer = 0
        self.tickLength = 0.02 # each tick is assumed to be 20ms (value imported from timer.py)
        self.activeTickLength = self.activeLength / self.tickLength
        self.displayTickLength = self.displayLength / self.tickLength
        self.inactiveTickLength = self.inactiveLength / self.tickLength
        self.episodeTickLength = self.episodeLength / self.tickLength
        self.graceTickLength = self.graceTimeThres / self.tickLength

    def getScreenInfo(self):
        """ very awful way to get screen info regarding pixel and cm but this is the best I have currently. 
        it will cause the Tk inter window to appear for split second at the beginning of trial """
        root = tk.Tk()
        info = {}
        info["width_px"] = root.winfo_screenwidth()
        info["height_px"] = root.winfo_screenheight()
        root.after(0, root.destroy)
        root.mainloop()
        return info
    
    def initPygameImage(self,corrospondence,cursorSize):
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

        # set up cursor
        name2image["cursor"] = getImage(corrospondence["cursor"],cursorSize)

        return name2image


    def update(self, param):

        # extract parameters
        decoder_output = param[0]
        softmax = decoder_output

        # extract etc if there is any
        etc = param[1] if len(param) > 1 else {}
        
        # check pygame event
        if self.render: self.check_pygame_event()

        # print(self.pygameKeyPressed)
        # if self.pygameKeyPressed == pygame.K_r:
        #     print("###############################hello")
        #     self.reset = True

        # determine target
        target = self.reset_and_determine_target(self.reset)
        self.target = target

        # update cursor position
        cursorPos = self.update_cursor(decoder_output,target)
        self.reset = self.tickTimer > self.episodeTickLength


        # draw
        if self.render: self.draw_state(cursorPos, target, self.nextTarget, softmax, etc)

        # end by incresing ticktimer (internal timer on unit of 20ms)
        self.tickTimer += 1
        state_taskidx = self.determine_taskidx(target)
        
        return [cursorPos, state_taskidx, target]


    def check_pygame_event(self):
        """ event handler for pygame intrinsic """
        
        # check for end 
        if self.useRealTime:
            if time.time() - self.startTime > self.sessionLength + 1:
                raise KeyboardInterrupt
                exit(1)

        # check for event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
                exit(1)
            if event.type == pygame.VIDEORESIZE:
                self.update_window_size_constants()

            # special pygame keypress recorder
            if event.type == pygame.KEYDOWN:
                self.pygameKeyPressed = event.key
                if pygame.key.name(event.key) == 'escape':
                    global quit_
                    quit_ = True
            else:
                self.pygameKeyPressed = None

    def reset_and_determine_target(self,reset):
        """ determine target based on reset flag """

        # if reset, run stop for 2 seconds
        if self.reset:
            self.trialCount += 1
            self.reset = False
            self.tickTimer = 0
            if self.resetCursorPos: 
                self.cursorPos = np.array([0.,0.])
                print("reset")
            self.resetYamlParam()

            # semi random target choice
            if len(self.nextTargetBucket) == 0: 

                # randomize target choices
                choices = list(self.desiredTargetList)
                random.shuffle(choices)
                self.multiply_equation = f"{random.randint(20,40)}x{random.randint(6,50)}=?"
                primeList = [3,4,5,6,7,8,9,11] # [3,5,7,11,13,17,19,23]
                self.subtract_equation = f"{random.randint(100,300)}  -  {random.choice(primeList)}"
                self.word_association = self.generate_next_word_association()

                # add target choices
                self.nextTargetBucket += choices
            self.nextTarget = self.nextTargetBucket.pop()
            self.text_pos = np.array([0.0, 0.0])

        # if not reset
        if self.tickTimer < self.inactiveTickLength: # 2 seconds
            return None # none means stop
        elif self.tickTimer == self.inactiveTickLength:
            # if need to start from random cursor position this is where we do it

            return self.nextTarget
        else:   # 10 seconds
            return self.nextTarget

    def update_cursor(self,decoder_output,target):
        """ update cursor position: np(+-1,+-1) """

        # get decoded direction np(+-1,+-1) from decoder output / copilot
        if target is not None:
            direction = self.decode_direction(decoder_output)
            
            # update current position
            self.cursorPos += direction * self.cursorVel
            self.cursorPos = np.clip(self.cursorPos, -1, 1)
        
        return self.cursorPos


    def decode_direction(self, decoder_output):
        """ find corrosponding direction np(+-1,+-1) from decoder_output"""

        # decoder direction
        if max(decoder_output) > self.softmaxThres: # 0.8
            directionIdx = np.argmax(decoder_output)
            direction = self.decodedVel[directionIdx]
        else:
            direction = np.array([0,0])
            
        # override
        if self.pygameKeyPressed is not None:
            if self.pygameKeyPressed == pygame.K_LEFT:
                print("manual input: left")
                direction = np.array([-1,0])
            if self.pygameKeyPressed == pygame.K_RIGHT:
                print("manual input: right")
                direction = np.array([1,0])
            if self.pygameKeyPressed == pygame.K_DOWN:
                print("manual input: down")
                direction = np.array([0,-1])
            if self.pygameKeyPressed == pygame.K_UP:
                print("manual input: up")
                direction = np.array([0,1])

        return direction

    def resetYamlParam(self):
        if hasattr(self, 'yamlName'):
            yaml_file = open(self.yamlName, 'r')
            yaml_data = yaml.load(yaml_file, Loader=Loader)
            yaml_file.close()
            
            # maintain screenSize but other than that reset everything
            yaml_data["modules"]["SJ_text_classification"]["params"]["screenSize"] = self.screenSize
            self.initParamWithYaml(yaml_data["modules"]["SJ_text_classification"]["params"])

            self.update_window_size_constants()

    def draw_text(self, surf, text, pos, color=(255, 255, 255), centered=True):
        # position is relative to surf!
        text_obj = self.pygameActionFont.render(text, True, color)
        # to-do: implement centered=False
        if centered:
            text_rect = text_obj.get_rect(center=pos)
            surf.blit(text_obj, text_rect)
        else:
            surf.blit(text_obj, pos)
        return
    
    def draw_state(self, cursorPos, target, nextTarget, softmax, etc={}):
        # etc is any thing that needs to be drawn on the board. normally it is an empty {}

        # reset image
        self.screen.fill((51,51,51))
        self.window.fill((0,0,0))

        # draw softmax bar
        if self.showSoftmax:
            pygame.draw.rect(self.window, self.softmaxBarColor,(self.softmaxBarImgPosA + self.softmaxBarImgSize * np.array([0,1-softmax[0]]),self.softmaxBarImgSize * np.array([1,softmax[0]])),0)
            pygame.draw.rect(self.window, self.softmaxBarColor,(self.softmaxBarImgPosB + self.softmaxBarImgSize * np.array([0,1-softmax[1]]),self.softmaxBarImgSize * np.array([1,softmax[1]])),0)
            pygame.draw.rect(self.window, self.softmaxBarColor,(self.softmaxBarImgPosC + self.softmaxBarImgSize * np.array([0,1-softmax[2]]),self.softmaxBarImgSize * np.array([1,softmax[2]])),0)
            pygame.draw.rect(self.window, self.softmaxBarColor,(self.softmaxBarImgPosD + self.softmaxBarImgSize * np.array([0,1-softmax[3]]),self.softmaxBarImgSize * np.array([1,softmax[3]])),0)
            pygame.draw.rect(self.window, self.softmaxBarColor,(self.softmaxBarImgPosE + self.softmaxBarImgSize * np.array([0,1-softmax[4]]),self.softmaxBarImgSize * np.array([1,softmax[4]])),0)


        # stop state
        if target == None:
            pygame.draw.rect(self.window, (250,10,10),(self.stopImgPos,self.stopImgSize),0)


            if self.tickTimer > (1-self.previewTimeThres) * self.inactiveTickLength:
                # show this target
                text = f"{nextTarget}"
                action_text = self.pygameActionFont.render(text, True, (255, 255, 255))
                text_width = action_text.get_width()
                self.window.blit(action_text,(self.L/2-text_width/2,self.L/2))
        # still state
        elif target == "still":
            pygame.draw.circle(self.window, (200, 200, 200), self.cursorBias, 120 / 1000 * self.L, 8)
        # other state
        else:
            # show this target
            text = f"{target}"
            if target == 'math':
                text = self.multiply_equation
            elif target == 'Subtract':
                text = self.subtract_equation
            elif target == 'Word Association':
                text = "#"+self.word_association
            else:
                if self.tickTimer > self.inactiveTickLength + self.displayTickLength:
                    text = ''
            #pos = (int(np.random.randint(0, self.L-1)), int(np.random.randint(0, self.L-1)))
            try:
                self.text_pos, self.text_vel
                #self.text_vel = 0.95*self.text_vel + (1 - 0.95)*np.random.randn(*self.text_pos.shape)
                #self.text_vel = self.textVelScale*np.random.rand()
                lam = 0.5
                self.text_vel = lam*self.text_vel + (1 - lam)*self.textVelScale*np.random.rand()
                dt = 0.020
                if self.text_direction == 'L':
                    self.text_pos = self.text_pos + [-dt*self.text_vel, 0]
                elif self.text_direction == 'R':
                    self.text_pos = self.text_pos + [ dt*self.text_vel, 0]
                elif self.text_direction == 'U':
                    self.text_pos = self.text_pos + [0,  dt*self.text_vel]
                elif self.text_direction == 'D':
                    self.text_pos = self.text_pos + [0, -dt*self.text_vel]
                #self.text_pos = self.text_pos + self.textVelScale*self.text_vel
                if (self.text_pos < -0.8).any() or (self.text_pos > 0.8).any():
                    self.text_pos[:] = (2*np.random.rand(2) - 1)*0.8
                    #self.text_vel[:] = 0.0
                    pool = ['L', 'R', 'U', 'D']
                    if self.text_pos[0] < -0.4:
                        pool.remove('L')
                    if self.text_pos[0] >  0.4:
                        pool.remove('R')
                    if self.text_pos[1] >  0.4:
                        pool.remove('U')
                    if self.text_pos[1] <  -0.4:
                        pool.remove('D')
                    self.text_direction = random.sample(pool, k=1)[0]
            except Exception as e:
                print('hm', e)
                self.text_pos = np.array([0.0, 0.0]) # between [-1, 1] x 2
                #self.text_vel = np.array([0.0, 0.0])
                self.text_vel = 0.0
                self.text_direction = random.sample(['L', 'R', 'U', 'D'], k=1)[0]
            pos = (self.text_pos*[1, -1] + 1)/2*self.L
            self.draw_text(self.window, text, pos)

            # find image coordinate pos of cursor assuming cursorPos range from +-1
            cursorImagePos = np.array([cursorPos[0],-cursorPos[1]])
            cursorImagePos = cursorImagePos * self.cursorMagnitude + self.cursorBias

            # draw cursor
            if self.styleChange:
                # pygame.draw.circle(self.window, (200, 200, 200), cursorImagePos, self.styleCursorRadius / 1000 * self.L, 0)
                self.window.blit(self.pygameImage['cursor'], cursorImagePos-self.cursorImgSize/2)
            else:
                pygame.draw.circle(self.window, self.cursorColor, cursorImagePos, self.cursorRadius / 1000 * self.L, 0)
                


            # draw etc
            for k,v in etc.items():
                itemImagePos = np.array([v[0][0], -v[0][1]])
                itemRadius = v[1]
                itemColor = v[2]
                itemImagePos = itemImagePos * self.cursorMagnitude + self.cursorBias

                # draw cursor
                pygame.draw.circle(self.window, itemColor, itemImagePos, itemRadius / 1000 * self.L, 0)


        # add text (prepare text content)
        elapsed = time.time() - self.startTime
        if self.showCountDown: 
            elapsed = self.sessionLength-elapsed
        min = int(elapsed // 60)
        sec = int(elapsed % 60)
        if min < 10: min = "0"+str(min)
        if sec < 10: sec = "0"+str(sec)

        text = f"Text Classification                  Trial: {self.trialCount}      Time: {min}:{sec}"
        name_surface = self.pygameFont.render(text, True, (255, 255, 255))
        self.screen.blit(name_surface,self.textImgPos)

        
        # add progress bar
        if self.showProgressBar:
            if self.tickTimer > self.inactiveTickLength:
                barRemain = np.array([1,1,1 - ((self.tickTimer - self.inactiveTickLength) / self.activeTickLength),1])

                barcolor = (230,230, 230) if self.tickTimer >= self.inactiveTickLength + self.graceTickLength else (230,79, 90)
                pygame.draw.rect(self.screen, barcolor, self.barPos * barRemain, 0)

        # update screen
        self.screen.blit(self.window,self.windowPadding)
        pygame.display.flip()

    def determine_taskidx(self,target):
        # print(self.action2state_task)
        return self.action2state_task[target]

    def generate_next_word_association(self):
        return random.choice(self.word_association_list)
        



# initiate constructor if called by main
initTask = True
copilotReady = False
targetPredReady = False
try: params # we have params
except NameError: initTask = False


if initTask: 
    # start task
    task = SJ_text_classification(params)

    # get copilot
    copilotReady = "copilot_path" in params
    if copilotReady: copilot = PPO.load(params["copilot_path"])

    targetPredReady = "targetPredictorPath" in params

