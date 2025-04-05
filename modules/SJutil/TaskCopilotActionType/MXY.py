import numpy as np
import math

# Mass moves in X,Y movement
# controls mx and my, but only mass has effect
class ActionMXY:
    def __init__(self, copilotYamlParam, copilotActionTypes):

        # copilotActionTypes = [mx,my,mass, click] or [smx,smy, mass,click]
        self.hasClick = 'click' in copilotActionTypes
        self.addDecoderDirection = 's' in copilotActionTypes[0]
        self.targetsPos = None # needs to be updated by outside function call, expected to become list of np array(2,)
        
        
        if "mass" in copilotActionTypes: self.controlMassMethod = None
        elif "target_mass" in copilotActionTypes: self.controlMassMethod = 't' # moves mass only to target locations
        elif "target_mass1" in copilotActionTypes: self.controlMassMethod = 't1' # t + uses 1.0 threshold instead of 0.5
        elif "target_mass_t" in copilotActionTypes: self.controlMassMethod = 'tmt' # t + uses time out switch (counter)
        elif "bounded_mass" in copilotActionTypes: self.controlMassMethod = 'b'
        else: raise Exception("Undefined copilot action type")
        
        # private variable
        self.massPos = np.zeros(2)
        self.mass = 0
        self.boundary = 0.4

        # only to be used for targetmass action with mass timer (target_mass_t)
        self.massTimer = 0
        self.massTimerThres = 10 # 200ms 200/20 = 10

        # public variable
        self.saved_copilot_output = np.zeros(len(copilotActionTypes))
        self.copilotInControl = False
        self.hideMass = True
        self.add_etc = None

    def copilot_action(self, copilot_output, decoder_dir, args):
        """
        moves mass according to copilot_output
        controlMassMethod = (None = regular, 't' = target, 'b' = bounded)

        # F = GmM/r2
        # a = F/m = GM/r2
        # p = p0 + 0.5at2 = p0 + 0.5GM/r2 if t=1
        # consider G and r (distance has scale k), hence equation has constant 0.5G/k2:
        # p = p0 + M/r2 * K, hyperparameter K controls scale of effect = 0.5M/r2
        """

        self.saved_copilot_output = copilot_output

        # extract arg
        cursorPos, cursorVel, = args

        # controls mass (self.mass, self.massPos)
        self.mass, self.massPos = self.__controlMass(self.massPos, copilot_output, cursorPos, cursorVel, self.controlMassMethod)

        # controls cursor
        if self.controlMassMethod == 't': 
            newCursorPos = self.__applyGravity(cursorPos, self.mass, self.massPos, reluBase=0.020,reluMax=0.030)
        else:
            newCursorPos = self.__applyGravity(cursorPos, self.mass, self.massPos)

        # add to pygame canvas
        if not self.hideMass:
            self.add_etc("mass", self.massPos, color = (250, 187, 55))

        # add decoder direction
        if self.addDecoderDirection:
            newCursorPos += decoder_dir * np.ones(2) * cursorVel
            newCursorPos = np.clip(newCursorPos,-1,1)

        # determine click
        click = (copilot_output[2] > 0) if self.hasClick else False

        # return new cursor and click 
        return newCursorPos, click
    
    def __controlMass(self, massPos, copilot_output, cursorPos, cursorVel, how=None):
        """
        how: None: regular way (residually)
        't': across targets
        'b': bounded by boundary
        """

        mass = (copilot_output[2] + 1) / 2
        
        if how == 't' or how == 't1' or how == 'tmt':
            
            threshold = 0.5
            if how == 't1': threshold = 1.0
            if how == 'tmt':
                self.massTimer += 1

            
            # dont move if too small magnitude
            if abs(copilot_output[0]) < threshold and abs(copilot_output[1]) < threshold:
                return mass, massPos
            # don't move if massTimer 
            if how == 'tmt':
                if self.massTimer < self.massTimerThres:
                    return mass, massPos
                else:
                    # having come to this point means, the mass will move, so set it back to zero
                    self.massTimer = 0

            # move if too big magnitude
            if abs(copilot_output[0]) > abs(copilot_output[1]): #xdir, ydir
                if copilot_output[0] > 0:
                    # move in right direction
                    candidatePos = massPos
                    candidateDist = np.inf
                    for targetPos in self.targetsPos:
                        if targetPos[0] > massPos[0]:
                            distance = math.dist(massPos, targetPos)
                            if distance < candidateDist:
                                candidateDist = distance
                                candidatePos = targetPos
                    massPos = candidatePos
                else:
                    # move in left direction
                    candidatePos = massPos
                    candidateDist = np.inf
                    for targetPos in self.targetsPos:
                        if targetPos[0] < massPos[0]:
                            distance = math.dist(massPos, targetPos)
                            if distance < candidateDist:
                                candidateDist = distance
                                candidatePos = targetPos
                    massPos = candidatePos
            else:
                if copilot_output[1] > 0:
                    # move in up direction
                    candidatePos = massPos
                    candidateDist = np.inf
                    for targetPos in self.targetsPos:
                        if targetPos[1] > massPos[1]:
                            distance = math.dist(massPos, targetPos)
                            if distance < candidateDist:
                                candidateDist = distance
                                candidatePos = targetPos
                    massPos = candidatePos
                else:
                    # move in down direction
                    candidatePos = massPos
                    candidateDist = np.inf
                    for targetPos in self.targetsPos:
                        if targetPos[1] < massPos[1]:
                            distance = math.dist(massPos, targetPos)
                            if distance < candidateDist:
                                candidateDist = distance
                                candidatePos = targetPos
                    massPos = candidatePos
            return mass, massPos
                    

        else: # 'b' or none
            massPos += copilot_output[:2] * cursorVel * 4

            # stay within boundary
            if how == 'b':
                difference = massPos - cursorPos 
                norm = np.linalg.norm(difference)
                if norm > self.boundary:
                    massPos = cursorPos + difference / norm * self.boundary

            # stay within map
            massPos = np.clip(massPos,-1,1)

        return mass, massPos
    
    def __applyGravity(self, cursorPos, mass, massPos,reluBase=0.002,reluMax=0.018):
        """
        apply gravitational law by return new cursor pos that results from cursorPos being near mass

        cursorPos (2,)
        mass 0 <= float <=1
        massPos (2,)

        returns (2,)
        """

        d = math.dist(massPos, cursorPos)
        v = massPos - cursorPos
        v_norm = np.linalg.norm(v)
        if v_norm == 0: uVec = np.zeros(2)
        else: uVec = v / v_norm
        eps = 0.000001; K = 0.00005; power = 2.5 # hyper parameter
        # need to stop at certain point, otherwise it flings off the screen when it gets too close
        # print(d)
        if d > eps:
            # mag = mass / (d ** power) * K + 0.001
            # mag = 1 / (d * 1000) + 1/400 
            
            #relu
            mag = reluBase # reluBase (0.002 or 0.005)
            thres = 0.5
            if d < thres: mag = reluBase + (thres - d) / thres * reluMax
            mag *= mass

            # mag = d
            # mag = mass / ((d + 0.3) ** power) * 0.05 - 0.01
            newCursorPos = cursorPos + uVec * mag
        else:
            newCursorPos = np.copy(massPos)
        if math.dist(newCursorPos,cursorPos) > d: 
            newCursorPos = np.copy(massPos)

        return newCursorPos

    def reset_each_trial(self):

        self.massPos = np.zeros(2)