

""" 
there is too much files inside trained model
needed a simple reader that returns just what I need


following is what I want:

read.py keyword --best # show .csv result with fold with best validation only
read.py keyword --sigfig=3 # show .csv result with 3 decimal place
read.py --unseen # show list of .csv files that hasn't been seen by me yet

return everything I need with the following keyword
keyword is *keyword* for each 

"""

import csv
import glob
import argparse
import re
import subprocess
import os
import atexit
from pathlib import Path



def findBestFold(data):
    """ find best fold with best validation / training """
    bestVal = 0
    bestValFold = 0
    bestTrain = 0
    bestTrainFold = 0

    for f,t,v in data[1:]:
        f = f
        t = float(t)
        v = float(v)

        if v > bestVal:
            bestVal = v
            bestValFold = f
        
        if t > bestTrain:
            bestTrain = t
            bestTrainFold = f

    return bestValFold, bestTrainFold


def boldCommand(command,keywords):
    for keyword in keywords:
        start = command.find(keyword)
        end = start + len(keyword)
        command = command[:start] + '\033[94m\033[1m' + command[start:end] + '\033[0m'  + command[end:]

    return command

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Specify EEGNet Architecture")
    parser.add_argument("keyword", help="keyword you want to have included in", nargs='*', default=[])
    parser.add_argument("--sigfig", help="number of sigfig to show", type=int, default=2)
    parser.add_argument("--best", help="show fold with best validation only", default=False, action="store_true")
    args = parser.parse_args()


    filenames = glob.glob(f"SJtools/copilot/runs/*/log.txt")
    startLength = len("SJtools/copilot/runs/")
    directoryNames = [f[startLength:-8] for f in filenames]

    """ collect filename and content """
    relevantFile = []
    for filename in filenames:
        with open(filename,'r') as file:
            lines = file.readlines()
            if len(lines) == 1: continue
            
            id = filename[startLength:-8]
            command = lines[0]
            scores = []
            
            
            for i in range(1,min(len(lines),5)):
                if 'with' not in lines[i]: continue
                start = lines[i].find(":") + 2
                end = lines[i].find(",")
                scores.append(float(lines[i][start:end]))
            
            relevantFile.append([id,command,scores]) 

    """ filter with keyword """
    for keyword in args.keyword:
        r = re.compile(f".*{keyword}")

        # relevantFile
        # r.match(command)
        relevantFile = list(filter(lambda x: r.match(x[1]), relevantFile)) # Read Note below

    """ print content """
    print(f'\033[91m@@@\033[0m')
    for id,command,scores in relevantFile:
        boldedCommand = boldCommand(command,args.keyword)
        print(f'\033[1m{id} \033[0m : {boldedCommand}{scores}')
        print()

    """ ask if any of them needs to be tested """
    desiredId = None
    try:
        historyFile = open('SJtools/copilot/runs/history.txt','r')
        prev_answers = [l.strip() for l in historyFile.readlines()]
    except:
        prev_answers = []
    def save_history(prev_answers):
        with open('SJtools/copilot/runs/history.txt','w') as f:
            for l in prev_answers[-5:]:
                f.write(l+'\n')
    atexit.register(save_history, prev_answers)

    while True:
        answer = input('\033[94m' + "type desired id or press enter 'q' to quit: " + '\033[0m' )
        if answer[0] == 'b':
            if len(answer) == 1:
                print(prev_answers[-1])
                continue
            if answer[1:].isdigit():
                print('\n'.join(prev_answers[-int(answer[1:])+1:]))
                continue

        splitAnswer = answer.split(' ')
        desiredId = splitAnswer[0]
        if desiredId == "q": exit()
        if Path(f"SJtools/copilot/runs/{desiredId}").exists(): 
            if len(splitAnswer) == 1: answer += ' last' # default look at last
            command = f"python -m SJtools.copilot.test {answer} -from_readpy"
            print(command)
            os.system(command)
        else: 
            print(f"invalid id: {desiredId}")
        prev_answers.append(answer)
