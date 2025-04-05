
import random
import numpy as np

class AssistClass():
    """ Assist refactored:
    this module automatically finds correct direction for cursor to move and moves in that direction
    it moves in 4 different style to make it hard for user to realize whether assist is being used.

    supports three strategy:
        efficient: moves directly toward goal position
        bigger magnitude: reduces bigger magnitude first
        natural (assistNaturally): randomly chooses one cardial direction to take
        natural 2 (assistNaturallyForTwoDirection) : (used only for two target) randomly choose one direction to take for certain duration

    # No Private Access: does not access any of the task's variable directly, they are always passed down
    # Ready Only Constant: none of the passed down argument is modified in this class. it is only read

    """

    def __init__(self,assistMode,assistValue,cursorSpeed,tickLength):

        # constants
        self.yamlSettingChanged(assistMode,assistValue,cursorSpeed,tickLength) # needed to determine how long to keep certain direction

        # fresh start
        self.resetEveryTrial()

    def yamlSettingChanged(self,assistMode,assistValue,cursorSpeed,tickLength):
        self.assistMode = assistMode
        self.assistValue = assistValue
        self.cursorSpeed = cursorSpeed
        self.tickLength = tickLength

    def resetEveryTrial(self):
        """
        to be called every trial beginning
        """

        # for assist naturally
        self.assistNaturalDuration = 0
        self.assistNaturalDirection = np.array([0,0])

        # for assistNaturallyForTwoDirection
        self.assistStrategyShouldAssist = False
        self.assistStrategyTimer = 0


    def assist(self,targetPos,cursorPos,currentDirection):
        # target position
        
        direction = currentDirection

        # over ride direction if we have assist
        if self.assistValue > 0 and targetPos is not None:

            targetDirection = targetPos - cursorPos
            # assist efficiently
            if self.assistMode == 'e':
                if np.random.random() < self.assistValue:
                    direction = targetDirection / np.linalg.norm(targetDirection)
            # assist by prioritizing bigger magnitude first                    
            elif self.assistMode == 'b':
                if np.random.random() < self.assistValue:
                    abx,aby = abs(targetDirection)
                    if abx > aby: direction = np.sign(targetDirection) * np.array([1,0])
                    else: direction = np.sign(targetDirection) * np.array([0,1])
            # assist efficiently
            elif self.assistMode == 'n':
                if np.random.random() < self.assistValue:
                    direction = self._assistNaturally(targetDirection)
            # assist naturally for binary
            elif self.assistMode == 'n2':
                direction = self._assistNaturallyForTwoDirection(currentDirection,targetDirection)
        
        return direction    

    def _assistNaturally(self,correctDirection):
        """
        randomly chooses one cardial direction to take
        """
        # move to certain direction for certain amount of assist (i.e)
        directionMag = abs(correctDirection)
        abx,aby = directionMag

        # print(self.assistNaturalDuration, self.assistNaturalDirection)
        
        if self.assistNaturalDuration > 0:
            dir = self.assistNaturalDirection

        if self.assistNaturalDuration <= 0:

            # reset natural direction duration
            self.assistNaturalDuration = random.random() * abs(min(correctDirection))/2

            # choose direction
            if random.random() < abx / (abx + aby):
                dir = np.array([1,0]) * np.sign(correctDirection)
            else:
                dir = np.array([0,1]) * np.sign(correctDirection)
            self.assistNaturalDirection = dir
            
        self.assistNaturalDuration -= self.cursorSpeed

        return dir
    

    def _assistNaturallyForTwoDirection(self,currentDirection,targetDirection):
        """ 
        motive: assist continuously in correct direction for x amount of time
        i.e) if assistValue is 0.2, and it randomly helps for 0.2 seconds, then it won't help for 0.8 seconds continuously
        Note about implementation: it actually starts by not assisting 0.8 seconds, then assissting 0.2 seconds
        also while it is moving in continuous direciton, it may fluctuate couple of times (according to randomValue 10%) to make it harder to tell if it is user or assist
        ^ motive is to make it more deceptive but may not be needed (it is commented out for now)
        """
        
        if self.assistStrategyTimer == 0:
            if not self.assistStrategyShouldAssist:
                # determine parameter for this round
                totalLength = random.randint(1000,3000) / 1000 / self.tickLength # between 1000-3000 ms
                self.assistStrategyHelpLength = totalLength * self.assistValue
                self.assistStrategyDontHelpLength = totalLength * (1-self.assistValue)
                # print(totalLength, self.assistStrategyHelpLength,self.assistStrategyDontHelpLength)
        
        # always increment timer
        self.assistStrategyTimer += 1

        # move continuosly in correct direction for x amount
        if self.assistStrategyShouldAssist:
            if self.assistStrategyTimer > self.assistStrategyHelpLength:
                self.assistStrategyShouldAssist = False
                self.assistStrategyTimer = 0
                # self.assistStrategyDontHelpLength = 80

            abx,aby = abs(targetDirection)
            if abx > aby: correctDirection = np.sign(targetDirection) * np.array([1,0])
            else: correctDirection = np.sign(targetDirection) * np.array([0,1])

            # if random.random() < 0.2:
            #     return -correctDirection
                        
            return correctDirection

        # don't assist for x / (1-assist value)
        else:
            if self.assistStrategyTimer > self.assistStrategyDontHelpLength:
                self.assistStrategyShouldAssist = True
                self.assistStrategyTimer = 0
                
            return currentDirection