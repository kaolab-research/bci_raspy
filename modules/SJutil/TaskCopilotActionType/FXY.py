
import numpy as np

# accelerated X,Y movement
class ActionFXY:
    def __init__(self, copilotYamlParam, copilotActionTypes):

        # copilotActionTypes = [fx,fy,click]
        self.hasClick = 'click' in copilotActionTypes

        # private variable
        self.copilot_direction_value = np.zeros(2)

        # public variable
        self.copilotInControl = False
        self.saved_copilot_output = np.zeros(len(copilotActionTypes))

    def copilot_action(self, copilot_output, decoder_dir):

        # assume copilot output has [fx,fy]
        self.saved_copilot_output = copilot_output
        
        # acceleration scaling
        copilot_output[:2] *= 0.5
        if np.linalg.norm(copilot_output[:2]) < 0.02: 
            copilot_output[:2] *= 0
            self.copilotInControl = False
        else:
            self.copilotInControl = True

        # define velocity to add by force (acceleration)
        self.copilot_direction_value += copilot_output[:2] # acceleration
        self.copilot_direction_value *= 0.9 # damping term
        self.copilot_direction_value = np.clip(self.copilot_direction_value,-10,10) # clip so it doens't fly out

        # determine click
        click = (copilot_output[2] > 0) if self.hasClick else False

        return decoder_dir + self.copilot_direction_value, click

    def reset_each_trial(self):

        self.copilot_direction_value = np.zeros(2)