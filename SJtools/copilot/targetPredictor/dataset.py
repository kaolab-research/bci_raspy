
from SJtools.copilot.env import SJ4DirectionsEnv
import numpy as np
from torch.utils.data import Dataset
import torch
import os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

"""
Relies upon env from copilot to create dataset
Consider running the game 500 episode to collect enough softmax game play
then use it to generate target
"""

#https://www.crosstab.io/articles/time-series-pytorch-lstm
class MyDataset(Dataset):
    def __init__(self, sequence_length=128,device=None,epoch_size=1000,save_path=None,ignoreStillState=True,include_cursor_pos=False,softmax_type='complex',dataset_copilot=None,copilot_param={"model":"PPO","alpha":1,"target_predictor":"LSTMFCS5_S2P_CXY_2","target_predictor_input":"softmax_pos"},show_dataset_generation=False,randomInitCursorPosition=False):

        self.epoch_size = epoch_size
        self.sequence_length = sequence_length
        self.ignoreStillState = ignoreStillState
        self.include_cursor_pos = include_cursor_pos

        # example of LSTM dataset (epoch_size, seq_length, feature_n)
        # x_train = torch.rand(1574, seq_dim, 5,)
        # y_train = torch.rand(1574, 2)
        createData = True
        if save_path is not None and save_path != "":
            folderPath = "./SJtools/copilot/targetPredictor/data/"
            filePath = folderPath + save_path
            if os.path.exists(filePath):
                createData = False

        if createData:
            X,Y = [],[]
            episodeBegin = [0]

            if dataset_copilot is not None:
                # choose model CONTINUE FROM HERE
                filePath = "SJtools/copilot/models/" + dataset_copilot
                if copilot_param["model"] == "RecurrentPPO":
                    model = RecurrentPPO.load(filePath)
                if copilot_param["model"] == "PPO":
                    model = PPO.load(filePath)

                # PERFORM
                render = show_dataset_generation
                env = SJ4DirectionsEnv(render=render,showSoftmax=render,softmax_type=softmax_type,randomInitCursorPosition=randomInitCursorPosition,setAlpha=copilot_param["alpha"],useTargetPredictor=copilot_param["target_predictor"],target_predictor_input=copilot_param["target_predictor_input"])
            
            else:
                env = SJ4DirectionsEnv(softmax_type=softmax_type,randomInitCursorPosition=randomInitCursorPosition)

            obs = env.reset()

            i = 0
            while i < epoch_size:

                # run single step with argmax action
                if dataset_copilot is None: 
                    #use argmax action
                    action = np.array([0,0,-1])
                if dataset_copilot is not None: 
                    # use copilot
                    action, _ = model.predict(obs, deterministic=True)
                    if copilot_param["alpha"] is not None: action[2] = copilot_param["alpha"]
                
                obs, reward, done, info = env.step(action)

                # extract information needed to create train of softmax
                softmax = info["softmax"]
                target_pos = info["target_pos"]
                task_id = info["task_id"]
                cursor_pos = info["cursor_pos"]

                if task_id == 4 and np.sum(np.isnan(target_pos)) > 0: target_pos = [0,0]
                
                if task_id != 4 or not self.ignoreStillState:  
                    
                    if self.include_cursor_pos:
                        X.append(np.concatenate([softmax,cursor_pos]))
                        Y.append(target_pos)
                    else:
                        X.append(softmax)
                        Y.append(target_pos)
                    i += 1
                
                if done: 
                    obs = env.reset()
                    episodeBegin.append(i+1)

            self.X = torch.tensor(np.array(X)).type(torch.float32)
            self.Y = torch.tensor(np.array(Y)).type(torch.float32)

            # for index that should start with 0 softmax because it has not seen one
            # i.e) i=0 requires 127, i=3 requires 124
            input_dim = self.X.shape[1]
            self.zeros = torch.tensor(np.zeros((self.sequence_length,input_dim))).type(torch.float32)
            self.hasManyZeroToAppend = {} 
            for i in episodeBegin:
                for j in range(sequence_length):
                    self.hasManyZeroToAppend[i+j] = sequence_length-j-1

        # save:
        # if save path exists, then load
        if save_path is not None and save_path != "":
            folderPath = "./SJtools/copilot/targetPredictor/data/"
            filePath = folderPath + save_path
            if os.path.exists(filePath):
                loaded = torch.load(filePath)
                self.X = loaded['X']
                self.Y = loaded['Y']
                self.hasManyZeroToAppend = loaded['hasManyZeroToAppend']
                input_dim = self.X.shape[1]
                self.zeros = torch.tensor(np.zeros((self.sequence_length,input_dim))).type(torch.float32)
                print("Loaded Dataset")
            else:
                folderPath = "./SJtools/copilot/targetPredictor/data/"
                m = {'X': self.X, 'Y': self.Y, 'hasManyZeroToAppend':self.hasManyZeroToAppend}
                torch.save(m, folderPath + save_path)
                print("Generated Dataset")

        # send to device
        if device is not None:
            self.X = self.X.to(device=device)
            self.Y = self.Y.to(device=device)
            self.zeros = self.zeros.to(device=device)
  

    def __len__(self):
        # returns shape: self.epoch_size

        return self.epoch_size

    def __getitem__(self, i): 
        # returns shape: (seq_length, feature_n)

        toAppend = 0
        if i in self.hasManyZeroToAppend:
            toAppend = self.hasManyZeroToAppend[i]

        xStart = i-(self.sequence_length-toAppend)+1
        xEnd = i+1
        x = torch.cat([self.zeros[:toAppend],self.X[xStart:xEnd]])
        y = self.Y[i]

        return x,y

    def info(self):
        # input_dim, output_dim
        input_dim = self.X.shape[1]
        output_dim = self.Y.shape[1]
        seq_dim = self.sequence_length

        return input_dim, output_dim, seq_dim



if __name__ == "__main__":
    dataset = MyDataset()
    for i in range(len(dataset)):
        if i < 5:
            x,y = dataset[i]
            print("x y:",x.shape,y.shape)
        else:
            exit()