
from SJtools.copilot.env import SJ4DirectionsEnv
import numpy as np
from torch.utils.data import Dataset
import torch
import os
"""
Relies upon env from copilot to create dataset
Consider running the game 500 episode to collect enough softmax game play
then use it to generate target
"""

#https://www.crosstab.io/articles/time-series-pytorch-lstm
class MyRealDataset(Dataset):
    def __init__(self, realDataPaths, sequence_length=128,device=None,save_path=None,ignoreStillState=True,include_cursor_pos=False,n_skip_real_trials=5,truncated_epoch_size=None):

        self.sequence_length = sequence_length
        self.ignoreStillState = ignoreStillState
        self.include_cursor_pos = include_cursor_pos
        self.n_skip_real_trials = n_skip_real_trials
        if ignoreStillState: print("NOTE still state (id=4) is skipped")

        # example of LSTM dataset (epoch_size, seq_length, feature_n)
        # x_train = torch.rand(1574, seq_dim, 5,)
        # y_train = torch.rand(1574, 2)
        createData = True
        if save_path is None or save_path == "":
            file_path = None
        else:
            folder_Path = "./SJtools/copilot/targetPredictor/data/"
            file_path = folder_Path + save_path
            if os.path.exists(file_path):
                self.X, self.Y, self.hasManyZeroToAppend,self.zeros = self.load(file_path)
                createData = False

        # now if we must create data:
        if createData:
            self.X, self.Y, self.hasManyZeroToAppend,self.zeros = self.createData(realDataPaths,sequence_length)
        
        # save epoch size
        self.epoch_size = self.X.shape[0]

        # save data X,Y,hasManyZeroToAppend
        if file_path is not None:
            self.save(file_path,self.X,self.Y,self.hasManyZeroToAppend) # never overwrites

        if truncated_epoch_size is not None:
            print("using truncated epoch size of ", truncated_epoch_size)
            self.X = self.X[:truncated_epoch_size]
            self.Y = self.Y[:truncated_epoch_size]
            self.epoch_size = self.X.shape[0]

        # send to device
        if device is not None:
            self.X = self.X.to(device=device)
            self.Y = self.Y.to(device=device)
            self.zeros = self.zeros.to(device=device)
  
    def createData(self,realDataPaths,sequence_length):
        

        X,Y = [],[]
        episodeBegin = [0]
        cumulativei = 0
        cumulativeEpochSize = 0

        for realDataPath in realDataPaths:
            total_step, softmaxs, cursorposs, targetposs, state_tasks = self.get_clean_real_data(realDataPath)

            skipStateBegan = False
            skipStateEnded = True
            i = 0
            while i < total_step:

                # # run single step with argmax action
                # argmaxAction = np.array([0,0,-1])
                # obs, reward, done, info = env.step(argmaxAction)

                # extract information needed to create train of softmax
                softmax = softmaxs[i]
                target_pos = targetposs[i]
                task_id = state_tasks[i]
                cursorpos = cursorposs[i]

                if task_id == -1:
                    if not skipStateBegan: # if you are in -1 and skip state did not begin, mark the following
                        skipStateBegan = True
                        skipStateEnded = False
                    i += 1
                    continue
                else: # task id is not in skip state
                    if not skipStateEnded: # but if it never said skip state officially ended
                        skipStateEnded = True
                        skipStateBegan = False
                        episodeBegini = cumulativei + i
                        episodeBegin.append(episodeBegini)
                        # print("episodeBegin",task_id,episodeBegini)

                # if you are in still state and so target pos is all nan then set target_pos to 0,0
                if task_id == 4 and np.sum(np.isnan(target_pos)) > 0: 
                    target_pos = [0,0]
                
                # actaully BUT if you are in still state and it told you to ignore still state, then don't include it in dataset!
                if self.ignoreStillState and task_id == 4:
                    i += 1
                    continue
                    
                if self.include_cursor_pos:
                    X.append(np.concatenate([softmax,cursorpos]))
                    Y.append(target_pos)
                else:
                    X.append(softmax)
                    Y.append(target_pos)
                i += 1
            
            print(f'{realDataPath} contains {len(X)-cumulativeEpochSize} usable data points')
            cumulativei = i
            cumulativeEpochSize = len(X)


        X = torch.tensor(np.array(X)).type(torch.float32)
        Y = torch.tensor(np.array(Y)).type(torch.float32)
        print(f"After removing skip states, dataset contains {X.shape[0]} data points and {len(episodeBegin)} trials")
        

        # for index that should start with 0 softmax because it has not seen one
        # i.e) i=0 requires 127, i=3 requires 124
        input_dim = X.shape[1] # 5 or 7
        zeros = torch.tensor(np.zeros((sequence_length,input_dim))).type(torch.float32)
        hasManyZeroToAppend = {} 
        for i in episodeBegin:
            for j in range(sequence_length):
                hasManyZeroToAppend[i+j] = sequence_length-j-1

        print()
        return X, Y, hasManyZeroToAppend,zeros



    def load(self,filePath):
        if filePath is not None:
            # if given savepath file does exists, then load
            if os.path.exists(filePath):
                loaded = torch.load(filePath)
                X = loaded['X']
                Y = loaded['Y']
                hasManyZeroToAppend = loaded['hasManyZeroToAppend']
                input_dim = X.shape[1] # 5 or 7
                zeros = torch.tensor(np.zeros((self.sequence_length,input_dim))).type(torch.float32)
                print("Loaded Dataset")
        return X,Y,hasManyZeroToAppend,zeros

    def save(self,filePath,X,Y,hasManyZeroToAppend):

        # if given savepath file does not exists, then save (so as to avoid overwritting it)
        if not os.path.exists(filePath):
            m = {'X': self.X, 'Y': self.Y, 'hasManyZeroToAppend':self.hasManyZeroToAppend}
            torch.save(m, filePath)
            print("Saved Dataset as",filePath)



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

    def load_data(self,filename, return_dict = True):
        with open(filename, 'rb') as openfile:
            name = openfile.readline().decode('utf-8').strip()
            labels = openfile.readline().decode('utf-8').strip()
            dtypes = openfile.readline().decode('utf-8').strip()
            labels = labels.split(',')
            dtypes = dtypes.split(',')
            data = np.fromfile(openfile, dtype=[item for item in zip(labels, dtypes)])
            if not return_dict:
                return data
            data_dict = {label: data[label] for label in labels}
            data_dict['name'] = name
            data_dict['labels'] = labels
            data_dict['dtypes'] = dtypes
        return data_dict

    def get_clean_real_data(self,filename):
        # get real data from filename using load data and some cleaning namely:

            # I want 3 thing: cursorpos, softmax, targetpos
            # I don't want it when state_task is -1 or 4 (when it is stop or rest)
            # I want to skip first 4 trials
            # so need to keep track of 4 values: cursorpos, softmax, targetpos, state_task

        task = self.load_data(filename + 'task.bin')



        """ only look at relevant variables """
        cursorpos = task['decoded_pos']
        softmax = task['decoder_output']
        targetpos = task['target_pos']
        state_task = task['state_task']
        total_step = task['state_task'].shape[0]
        # print(total_step, "- total step")
        # print(task['decoder_output'].shape, "- softmax shape")
        # print(task['decoded_pos'].shape, "- cursorpos shape")
        # print(task['target_pos'].shape, "- targetpos shape")
        # print(task['state_task'].shape, "- state_task shape")

        """ trim it so as to skip first 5 trials"""
        skip_trial = self.n_skip_real_trials
        start_step = 0 # value we want to find
        i_trial = 0
        indexOfInvalidTrial = 0
        for i in range(1,total_step):
            # print(i,state_task[i],indexOfInvalidTrial,i_trial)
            if state_task[i] == -1: 
                indexOfInvalidTrial = i
            else:
                if i == indexOfInvalidTrial + 1: 
                    i_trial += 1

                # if i_trial is 6 then we have our starting step:
                if i_trial == skip_trial + 1:
                    start_step = i
                    break
        print(f"{filename} starting step is: {start_step} out of {total_step} | data counted from the beginning of trial #{i_trial}")

        # actually trim it according to start_step
        cursorpos = cursorpos[start_step:]
        softmax = softmax[start_step:]
        targetpos = targetpos[start_step:]
        state_task = state_task[start_step:]
        total_step = total_step - start_step

        # note this data still includes -1 and 4 as data point
        return total_step, softmax, cursorpos, targetpos, state_task





if __name__ == "__main__":
    realDataPath = ['./data/2022-08-12_SANGJOONLEE_3/']
    dataset = MyRealDataset(realDataPath)
    for i in range(len(dataset)):
        if i < 5:
            x,y = dataset[i]
            print("x y:",x.shape,y.shape)
        else:
            exit()