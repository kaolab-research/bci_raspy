
import numpy as np

# cursor moves in X,Y movement.
class ActionVXY:
    def __init__(self,copilotYamlParam, copilotActionTypes):

        # alpha type
        if copilotYamlParam["action_dim"]["alpha"] is None:
            self.copilot_default_alpha = None
            self.copilot_alpha = None
        elif copilotYamlParam["action_dim"]["alpha"] == 'binary':
            self.copilot_binary_alpha = True
            self.copilot_default_alpha = None
            self.copilot_alpha = None
        else:
            self.copilot_default_alpha = copilotYamlParam["action_dim"]["alpha"] * 2 - 1
            self.copilot_alpha = self.copilot_default_alpha
        
        # copilotActionTypes = [vx,vy,alpha,click]
        self.hasClick = 'click' in copilotActionTypes

        # public variable
        self.copilotInControl = False
        self.saved_copilot_output = np.zeros(len(copilotActionTypes))


    def copilot_action(self, copilot_output, decoder_dir):

        # assume copilot output has [vx,vy,alpha,click]
        self.saved_copilot_output = copilot_output

        # break arg into two
        copilot_dir, alpha = copilot_output[:2], (copilot_output[2]+1) / 2

        # constant copilot alpha
        if self.copilot_alpha is not None: alpha = (self.copilot_alpha+1) / 2
        
        # save snapshot of alpha
        self.copilotInControl = alpha > 0.5

        # determine click
        click = (copilot_output[3] > 0) if self.hasClick else False

        return alpha * copilot_dir + (1-alpha) * decoder_dir, click
    
    def reset_each_trial(self):
        pass