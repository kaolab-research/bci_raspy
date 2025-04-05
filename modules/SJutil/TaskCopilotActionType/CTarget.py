import numpy as np
import torch

class ActionCTarget:
    """ charge on all the target locations simply uses charges that sum to 1 """
    def __init__(self, copilotYamlParam, copilotActionTypes):

        # copilotActionTypes = [chargeTargets]
        # private constant variable
        self.set_up_action_param(copilotYamlParam)


        self.targetsPos = None # needs to be updated by outside function call, expected to become list of np array(2,)

        # public variable
        self.copilotInControl = False
        self.add_etc = None
        self.hideMass = True
        
        # other function
        self.softmax = torch.functional.F.softmax 

        """ some idea for type """
        # additive, as is
        # argmax
        # categorical distribution
        # also consider 


    def copilot_action(self, copilot_output, decoder_dir, args):
        cursorPos, cursorVel, copilotVel = args[:3]
        decoder_kf_vel = decoder_dir * np.ones(2) * cursorVel

        copilot_vel = self.calc_copilot_charge_vel(copilot_output, cursorPos, copilotVel)
        if self.noKf: newCursorPos = cursorPos + copilot_vel
        else: newCursorPos = cursorPos + decoder_kf_vel + copilot_vel
        newCursorPos = np.clip(newCursorPos,-1,1)
        return newCursorPos, False
    
    def copilot_action_kf_eff(self, copilot_output, decoder_dir, args):
        cursorPos, cursorVel, copilotVel, kf_eff = args
        decoder_kf_vel = kf_eff

        copilot_vel = self.calc_copilot_charge_vel(copilot_output, cursorPos, copilotVel)
        if self.noKf: newCursorPos = cursorPos + copilot_vel
        else: newCursorPos = cursorPos + decoder_kf_vel + copilot_vel
        newCursorPos = np.clip(newCursorPos,-1,1)
        return newCursorPos, False
        
    def calc_copilot_charge_vel(self, copilot_output, cursorPos, cursorVel):
        
        # action hyper parameter
        eps = self.eps # 0.01 
        K = self.K # 1
        power = self.power # 2
        reluBase = self.reluBase # 0.002; 
        reluMax = self.reluMax # 0.018
        temperature = self.temperature # 1/5
        useSoftmax = self.useSoftmax 
        useClip = self.useClip

        # need to stop at certain point, otherwise it flings off the screen when it gets too close
        
        chargeVel = np.zeros(2)

        # rescaled
        if useClip:
            charges = np.clip(copilot_output,0,1) #throwing away -1 to 0 may help span action better
        else:
            charges = (copilot_output + 1) / 2 # rescaling to 0, 1

        if useSoftmax:
            charges /= temperature
            charges = self.softmax(torch.tensor(charges).float(),dim=0).numpy()
        
        # print
        if not self.hideMass:
            for i,(targetPos, charge) in enumerate(zip(self.targetsPos,charges)):
                self.add_etc(f"mass{i}", targetPos, color = (250, 187, 55),radius=int(20*charge))


        # for i,(targetPos, charge) in enumerate(zip(self.targetsPos,charges)):            
        #     d = np.linalg.norm(targetPos-cursorPos) # maximum distance is 2, so scaled to 0-1
        #     chargeVel += (targetPos-cursorPos) / (d**power+eps)


        d = np.linalg.norm(self.targetsPos-cursorPos,axis=1) # maximum distance is 2, so scaled to 0-1
        mag = K * charges / (d**power + eps)
        chargeVel = mag @ (self.targetsPos-cursorPos)
        # print(mag)



        return chargeVel * cursorVel
    
    def reset_each_trial(self):

        self.massPos = np.zeros(2)

    def set_up_action_param(self, copilotYamlParam):

        self.eps = 0.01 
        self.K = 1
        self.power = 2 # hyper parameter
        self.reluBase = 0.002; 
        self.reluMax = 0.018
        self.temperature = 0.2
        self.useSoftmax = True
        self.useClip = False
        self.noKf = False
    
        if 'action_param' in copilotYamlParam['action_dim']:
            action_param = copilotYamlParam['action_dim']['action_param']

            self.eps = float(action_param.get('eps', self.eps))
            self.K = float(action_param.get('K', self.K))
            self.power = float(action_param.get('power', self.power))
            self.reluBase = float(action_param.get('reluBase', self.reluBase))
            self.reluMax = float(action_param.get('reluMax', self.reluMax))
            self.temperature = float(action_param.get('temperature', self.temperature))
            self.useSoftmax = bool(float(action_param.get('useSoftmax', self.useSoftmax)))
            self.useClip = bool(float(action_param.get('useClip', self.useClip)))
            self.noKf = bool(float(action_param.get('noKf', self.noKf)))
            

        

        