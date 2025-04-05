from modules.SJutil.TaskCopilot import TaskCopilotAction, TaskCopilotObservation
import numpy as np
class TaskTools:
    def __init__(self,TaskObservation:TaskCopilotObservation,TaskAction:TaskCopilotAction,tools):
        self.TaskObservation = TaskObservation
        self.TaskAction = TaskAction
        self.tools = tools

        self.seeCopilotContribution = 'seeCopilotContribution' in self.tools
        self.seeTrajectory = 'seeTrajectory' in self.tools

        # indepedent variable (constants)
        self.trajectoryColor = np.array((20,200,40))

    def mark_trajectory(self,etc):
        """ mark trajectory based on observation, mark it green"""
        
        if self.seeTrajectory:
            # color trajectory
            _etc = {}
            # n = len(self.TaskObservation.trajectoryIndex) // 2
            for i,index in enumerate(range(0,len(self.TaskObservation.trajectoryIndex),2)):
                if self.TaskObservation.trajectoryValue == []: break
                pos = self.TaskObservation.trajectoryValue[index:index+2]
                _etc[f't{int(i)}'] = (pos,10,self.trajectoryColor*np.array([0,1,0]))
            
            etc.update(_etc)
        return etc
    
    def show_copilot_contribution(self,etc):
        """
        show copilot's contribution
        """
        if self.seeCopilotContribution: 
            etc['copilot_contribution'] = self.TaskAction.saved_copilot_output[:2]
            etc['copilot_output_alpha'] = self.TaskAction.saved_copilot_output[2:3]

        return etc
        

    def use(self):
        pass