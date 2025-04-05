
import numpy as np

class PerformanceRecord:
    """
    some way to keep the 4 direction's task performance record in a safe and organized manner
    """

    def __init__(self):
        

        # variables to be read at the end
        self.timeToHitByTrials = [] # 
        self.distanceTravelledByTrials = []
        self.extraDistanceTravelledByTrials = [] # subtracts original dist(cursor to target)
        self.n_trial = 0
        self.n_hit_trial = 0
        self.n_hit_timeout_trial = 0
        self.skipTrial = False
        self.startTime = None
        self.startTickTime = None
        self.pastCursorPos = None
        self.distanceTravelled = 0 # distance travelled by that trail
        self.originalDistance = 0 # shorted distance that can be travelled in that trial

    @property
    def avgTimeToHit(self): return 0 if self.n_hit_timeout_trial == 0 else sum(self.timeToHitByTrials) / self.n_hit_timeout_trial
    @property
    def avgDistanceTravelled(self): return 0 if self.n_trial == 0 else sum(self.distanceTravelledByTrials) / self.n_trial
    @property
    def avgExtraDistanceTravelled(self): return 0 if self.n_trial == 0 else sum(self.extraDistanceTravelledByTrials) / self.n_trial
    

    def trial_start(self, cursorPos, targetPos, startTime, startTickTime):
        """ 
        must be called at the start of every trial.
        """
        if type(targetPos) == str: self.skipTrial = True
        else: self.skipTrial = False
        if self.skipTrial: return

        self.startTime = startTime
        self.startTickTime = startTickTime
        self.pastCursorPos = np.copy(cursorPos)
        self.distanceTravelled = 0
        self.originalDistance = np.linalg.norm(targetPos - cursorPos)
        # print("Trial Started", cursorPos, targetPos, startTime, startTickTime)


    def trial_end(self, game_state, cursorPos, endTime, endTickTime):
        """ 
        must be called at the end of every trial. marks the end of trial. presumably at reset 
        should save any progress for that trial as a result. without calling this reset.
        it is not fully stored yet
        """

        if self.skipTrial: return

        # print("Trial Ended",game_state, cursorPos, endTime, endTickTime)

        # distance
        self.distanceTravelledByTrials.append(self.distanceTravelled)
        self.extraDistanceTravelledByTrials.append(self.distanceTravelled-self.originalDistance)
        self.n_trial += 1

        # time
        if game_state == 'H' or game_state == 'T':
            self.timeToHitByTrials.append(endTickTime-self.startTickTime)
            self.n_hit_timeout_trial += 1
        if game_state == 'H':
            self.n_hit_trial += 1
        
        # print('time',self.timeToHitByTrials,self.avgTimeToHit)
        # print('dist',self.distanceTravelledByTrials,self.avgDistanceTravelled)
        # print('xtra dist',self.extraDistanceTravelledByTrials,self.avgExtraDistanceTravelled)
    

    def record_step(self, cursorPos):
        """ reording anything and all things at each step"""

        if self.skipTrial: return

        self.distanceTravelled += np.linalg.norm(self.pastCursorPos - cursorPos)
        self.pastCursorPos = np.copy(cursorPos)

