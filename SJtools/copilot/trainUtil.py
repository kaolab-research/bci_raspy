"""
This file contains little bits of code that clutters train and makes it hard to understand it
the code in this file should only be used by 
    - train.py
    - test.py
it should not be a dependency to raspy or otherwise
"""
import random
import string
import os
import yaml
import shutil
import torch
from pathlib import Path

class fileOrganizer():
    """ 
    should create directory and save everything relevant in it 
    primarily should save: model.zip, model.yaml, log.txt
    """

    currPath = 'SJtools/copilot/runs/'
    runId = ''
    runFolder = ''
    modelNames = []

    def __init__(self, wandbUsed, runId='', no_save=False, fileName='', currPath='', devSave=False, overwrite=False):
        """
        id should be unique
        no_save = if True, don't save anything
        """
        if devSave: 
            no_save = False
            fileName = 'dev'
            currPath = 'SJtools/copilot/runs/'
            overwrite = True

        # generate runId if it doesn't exist
        if wandbUsed: 
            self.runId = runId
        else:
            if fileName == '': self.runId = self.createId()
            else: self.runId = fileName
        
        # use path if it exists
        if currPath != '': self.currPath = str(Path(currPath)) + '/'

        self.save = not no_save

        # create if directory for saving models do not exist
        if self.save: os.makedirs(self.currPath, exist_ok=True)
        
        # create run folder where everything will be saved
        self.runFolder = self.currPath + self.runId + "/"
        if self.save: 
            os.makedirs(self.runFolder, exist_ok=overwrite)
            print(f"Saving to: {self.runFolder}")

        # create path name with it
        if self.save: 
            self.best_model_save_path = self.runFolder # needed by eval callback
            self.bestModelPath = self.best_model_save_path + 'best_model.zip' # cannot be changed. this name is chosen by eval callback
            self.lastModelPath = self.runFolder + 'last_model.zip'
            self.envPath = self.runFolder + 'env.py'
            self.logPath = self.runFolder + 'log.txt'
            self.evalCallbackPath = self.runFolder
            self.modelYamlPath = self.runFolder + 'model.yaml'
            self.bestModelYamlPath = self.runFolder + 'best_model.yaml'
            self.lastModelYamlPath = self.runFolder + 'last_model.yaml'
            self.rewardYamlPath = self.runFolder + 'reward.yaml'
            self.tensorboard_log = self.runFolder # read by train.py

        else: 
            self.best_model_save_path = None
            self.bestModelPath = None
            self.lastModelPath = None
            self.envPath = None
            self.logPath = None
            self.evalCallbackPath = None
            self.modelYamlPath = None
            self.bestModelYamlPath = None
            self.lastModelYamlPath = None
            self.rewardYamlPath = None
            self.tensorboard_log = None 
        
    def saveRewardYaml(self,rewardFilePath):
        if self.save:
            if Path(rewardFilePath).stem != 'custom':
                shutil.copyfile(rewardFilePath,self.rewardYamlPath)

    def saveYaml(self,content):
        if self.save:
            with open(self.modelYamlPath, 'w') as file:
                yaml.dump(content, file)
            with open(self.bestModelYamlPath, 'w') as file:
                yaml.dump(content, file)
            with open(self.lastModelYamlPath, 'w') as file:
                yaml.dump(content, file)
    
    def createId(self):
        id = ''.join([random.choice(string.ascii_lowercase+string.digits) for _ in range(10)])
        id = 'now-' + id # now = nowandb
        return id

    def log(self,txt):
        if self.save:
            with open(self.logPath, 'a') as file:
                file.write(txt+'\n')


def isfloat(n):
    try:
        float(n)
        return True
    except ValueError:
        return False
    
def getDevice():
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
        
    device = torch.device(device_name)
    return device