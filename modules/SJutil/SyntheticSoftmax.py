import numpy as np


class SyntheticSoftmax:
    # self contained synthetic softmax
    # it is stateful internally. has timer that tracks how softmax changes

    def __init__(self):

        # constants
        self.stillState = 4 # still state id
        self.N_STATE = 5 # Total number of state

        # init variables
        self.previousDirection = 0
        self.transitionTimer = 0
        self.transitionTimerLength = 5
        self.chosenDirectionTimer = 0
        self.chosenDirection = 0

    def getCorrectDirection(self,cursorPos, targetPos, targetSize):

        # by target position
        state = None # either 0~4 LRUD
        dirs = targetPos - cursorPos
        posDirs = abs(dirs) 
        
        xy = np.argmax(posDirs)
        sign = dirs[xy]
        if xy == 0:
            if sign > 0: state = 1
            else: state = 0
        else:
            if sign > 0: state = 2
            else: state = 3

        inTarget = (posDirs[0] <= (targetSize[0]/2)) * (posDirs[1] <= (targetSize[1]/2))
        if inTarget: 
            state = self.stillState

        return state


    def twoPeakSoftmax(self, cursorPos, targetPos, targetSize, CSvalue=0.7, stillCS=0.7):
        # there is always two peak with a tid bit noise
        # 70% of the time, correct peak
        # 30% wrong peak chosen
        # if two different peak chosen, then there is transition between the two 15-20 frame
        # shave off still applies but only to top 5% (to suggest noise, randomly distributed)

        """ returns synthetic softmax for creating game observation """
        """ CSvalue percentage of correct softmax (does not apply for simplesoftmax!) """

        # cursorPos, targetPos, targetSize = self.SJ4.pastCursorPos, self.SJ4.pastTargetPos, self.SJ4.pastTargetSize

        # 80% of time correct direction, 20% of time deteremine how long the wrong target will last
        if self.chosenDirectionTimer > 0:
            self.chosenDirectionTimer -= 1
        else:
            correctDirection = self.getCorrectDirection(cursorPos, targetPos, targetSize)

            # choose correct direction depending on stillCS or CSvalue
            if correctDirection == self.stillState:
                useCorrectDirection = np.random.random() <= stillCS
            else:
                useCorrectDirection = np.random.random() <= CSvalue

            if useCorrectDirection: # percentage of correct softmax
            # correct target
                self.chosenDirection = correctDirection
                self.chosenDirectionTimer = np.random.randint(20,40)
                self.transitionTimerLength = np.random.randint(15,20)
                self.transitionTimer = 0
            else:
            # wrong direction
                directions = list(range(self.N_STATE))
                directions.remove(correctDirection)
                self.chosenDirection = directions[np.random.randint(self.N_STATE-1)]
            
                self.chosenDirectionTimer = np.random.randint(20,40)
                self.transitionTimerLength = np.random.randint(15,20)
                self.transitionTimer = 0
        

        # transition
        if self.transitionTimer == self.transitionTimerLength:
            self.previousDirection = self.chosenDirection

        if self.transitionTimer < self.transitionTimerLength:
            self.transitionTimer += 1
            
        if self.transitionTimer < self.transitionTimerLength and self.chosenDirection != self.previousDirection:
            stateVector = np.zeros(self.N_STATE)
            stateVector[self.chosenDirection] = 1 * self.transitionTimer / self.transitionTimerLength
            stateVector[self.previousDirection] = 1 * (1- self.transitionTimer / self.transitionTimerLength)
        else:
            # prepare one hot vector
            stateVector = np.zeros(self.N_STATE)
            stateVector[self.chosenDirection] = 1
        
        # 50 % of the time shave off top 5% and distribute it to others
        shaveOff = (np.random.random()*2-1) * 0.05
        shaveOff = 0 if shaveOff < 0 else shaveOff

        # distribute shaveOff semi equally to rest
        equal = np.random.rand(5)
        equal[self.chosenDirection] = 0
        equal = equal / np.sum(equal)
        equal *= shaveOff
        stateVector += equal
        stateVector[self.chosenDirection] -= shaveOff
        softmax = stateVector

        return softmax
    
    def simpleSoftmax(self, cursorPos, targetPos, targetSize, noiseLevel = 3):
        # cannot handle extra_target more than cardianl directional targets!! will throw an error if you attept to do so

        # cursorPos, targetPos, targetSize = self.SJ4.pastCursorPos, self.SJ4.pastTargetPos, self.SJ4.pastTargetSize

        state_taskidx  = self.getCorrectDirection(cursorPos, targetPos, targetSize)
        target_encoding = np.zeros((self.N_STATE))
        target_encoding[state_taskidx] = 1
        target_encoding += noiseLevel * np.random.rand(self.N_STATE)
        target_encoding = np.exp(target_encoding) / np.sum(np.exp(target_encoding))
        return target_encoding
    
    def correctSoftmax(self, cursorPos, targetPos, targetSize, noiseLevel = 0):
        # cannot handle extra_target more than cardianl directional targets!! will throw an error if you attept to do so

        # cursorPos, targetPos, targetSize = self.SJ4.pastCursorPos, self.SJ4.pastTargetPos, self.SJ4.pastTargetSize

        state_taskidx  = self.getCorrectDirection(cursorPos, targetPos, targetSize)
        target_encoding = np.zeros((self.N_STATE))
        target_encoding[state_taskidx] = 1
        target_encoding += noiseLevel * np.random.rand(self.N_STATE)
        target_encoding = target_encoding / np.sum(target_encoding)
        return target_encoding
    
    def complexSoftmax(self, cursorPos, targetPos, targetSize, CSvalue=0.7, stillCS=0.7):
        """ returns synthetic softmax for creating game observation """
        """ CSvalue percentage of correct softmax (does not apply for simplesoftmax!) """

        # cursorPos, targetPos, targetSize = self.SJ4.pastCursorPos, self.SJ4.pastTargetPos, self.SJ4.pastTargetSize

        # 80% of time correct direction, 20% of time deteremine how long the wrong target will last
        if self.chosenDirectionTimer > 0:
            self.chosenDirectionTimer -= 1
        else:
            # find correct direction
            correctDirection = self.getCorrectDirection(cursorPos, targetPos, targetSize)

            # choose correct direction depending on stillCS or CSvalue
            if correctDirection == self.stillState:
                useCorrectDirection = np.random.random() <= stillCS
            else:
                useCorrectDirection = np.random.random() <= CSvalue

            # once decided, fix on a direction
            if useCorrectDirection: # percentage of correct softmax
            # correct target
                self.chosenDirection = correctDirection
                self.chosenDirectionTimer = np.random.randint(20,40)
            else:
                # wrong direction
                directions = list(range(self.N_STATE))
                directions.remove(correctDirection)
                self.chosenDirection = directions[np.random.randint(self.N_STATE-1)]
                self.chosenDirectionTimer = np.random.randint(20,40)
        
        # prepare one hot vector
        stateVector = np.zeros(self.N_STATE)
        stateVector[self.chosenDirection] = 1
        
        # 50 % of the time shave off top 40% and distribute it to others
        shaveOff = (np.random.random()*2-1) * 0.4
        shaveOff = 0 if shaveOff < 0 else shaveOff

        # distribute shaveOff semi equally to rest
        equal = np.random.rand(5)
        equal[self.chosenDirection] = 0
        equal = equal / np.sum(equal)
        equal *= shaveOff
        stateVector += equal
        stateVector[self.chosenDirection] -= shaveOff
        softmax = stateVector

        # print(state_taskidx,cursorPos,chosenDirection,softmax)
        # print(state_taskidx,np.argmax(softmax),softmax)

        return softmax
    
    def normalTargetSoftmax(self, cursorPos, targetPos, targetSize, CSvalue=0.7, stillCS=0.7):
        return None
    
if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    targetPos = np.array([0.85,0])
    targetSize = np.array([0.2,0.2])
    cursorPos = np.array([0,0])
    synSof = SyntheticSoftmax()
    
    for i in range(5):
        print(synSof.twoPeakSoftmax(targetPos,cursorPos,targetSize))
    