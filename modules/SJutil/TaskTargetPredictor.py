
# all the loading, unloading, predicting using target predictor is done here

import numpy as np
import torch
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from SJtools.copilot.targetPredictor.model import LSTM,LSTM2,LSTMFCS,NN

class TaskTargetPredictorClass():

    def __init__(self, name):

        """
        initialize target predictor class, there is only two type:
        - truth
        - name (should have .pt file and yaml file stored somehwere in the SJtools/.../models/; i.e swept-music)
        """

        if name is None: raise Exception("Undefined target predictor")
        
        # get path
        if name == "truth": path = "truth"
        else: path = "SJtools/copilot/targetPredictor/models/" + name
        self.targetPredictorPath = path

        # initialize with path
        TP_model, TP_input_dim = self.initTargetPredictor(path)
        self.targetPredictorModel = TP_model # None if truth chosen
        self.targetPredictorInputDim = TP_input_dim # None if truth chosen
        if TP_model is not None: self.targetPredictorModel.reset()

    def initTargetPredictor(self,modelpath):
        # prepare model according to yaml

        if modelpath == "truth": return None,None

        # load model
        yamlpath =  modelpath + ".yaml"
        
        with open(yamlpath) as yaml_file:
            yaml_data = yaml.load(yaml_file, Loader=Loader)
            if 'model' in yaml_data: model_type = yaml_data['model']
            yaml_hyp = yaml_data["hyperparameters"]
            if 'hidden_dim' in yaml_hyp: hidden_dim = yaml_hyp['hidden_dim']
            if 'num_layers' in yaml_hyp: num_layers = yaml_hyp['num_layers']
            if 'num_lstms' in yaml_hyp: num_lstms = yaml_hyp['num_lstms']
            if 'fc_dim' in yaml_hyp: fc_dim = yaml_hyp['fc_dim']
            if 'sequence_length' in yaml_hyp: sequence_length = yaml_hyp['sequence_length']
            if 'input_dim' in yaml_hyp: input_dim = yaml_hyp['input_dim']
            if 'output_dim' in yaml_hyp: output_dim = yaml_hyp['output_dim']

        # use yaml data file if it exists
        PATH =  modelpath + ".pt"
        if model_type == "LSTM_POS1":
            model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)    
        elif model_type == "LSTM_POS2":
            model = LSTM2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,num_lstms=num_lstms)
        elif model_type == "LSTMFCS":
            model = LSTMFCS(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, fc_dim=fc_dim)
        elif model_type == "NN":
            model = NN(input_dim=input_dim, sequence_length=sequence_length, output_dim=output_dim)
        else:
            raise Exception("ERROR non-existant target predictor model")
            exit(1)

        model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
        print(f"Target Predictor Model '{model_type}' Loaded")
        # if useCuda: model.cuda()

        return model, input_dim

    def predict(self, softmax, cursorPos, correctTargetPos, reset=False):
        """ returns prediction of the target position """

        # reset H and C in LSTM
        if self.targetPredictorPath == 'truth': return correctTargetPos

        if reset: self.targetPredictorModel.reset()

        if self.targetPredictorInputDim == 5:
            input = softmax
            input = torch.tensor(input).reshape((1,1,self.targetPredictorInputDim)).type(torch.float32)
        elif self.targetPredictorInputDim == 7:
            input = np.concatenate([softmax, cursorPos])
            input = torch.tensor(input).reshape((1,1,self.targetPredictorInputDim)).type(torch.float32)
        
        output = self.targetPredictorModel.predict(input).reshape((2)).numpy()
        return output
    
    def reset(self):
        # manual reset
        if self.targetPredictorPath == 'truth': 
            return
        else:
            self.targetPredictorModel.reset()
        


if __name__ == "__main__":
    softmax = np.array([0,0,1,0,0]) # up
    cursorPos = np.array([-1,0]) # left
    correctTargetPos = np.array([0.85,0])
    
    targetPC = TaskTargetPredictorClass("truth")
    y = targetPC.predict(softmax,cursorPos,correctTargetPos,reset=False)
    print("truth target predictor:",y)
    print()
    
    targetPC = TaskTargetPredictorClass("swept-music")
    y = targetPC.predict(softmax,cursorPos,correctTargetPos,reset=False)
    print("truth target predictor:",y)
    print()

    targetPC = TaskTargetPredictorClass("colorful-energy")
    y = targetPC.predict(softmax,cursorPos,correctTargetPos,reset=False)
    print("truth target predictor:",y)
    print()