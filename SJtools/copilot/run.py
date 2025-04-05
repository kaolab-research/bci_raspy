# after running commands in run.sh, places little asterisk to indicate that it is done
# then result is saved in result.txt

import os
import subprocess
import time
import random
import string
import argparse

""" 
NOTE: run.sh script file can only contain *single line command* and comment
Future:
consider writing better temp log
consider adding # IN PROGRESS FLAG IN run.sh
"""

# hyper parameters
parser = argparse.ArgumentParser(description="")
parser.add_argument("-file", default='run.sh', help=".sh file that needs to be used")
args = parser.parse_args()

# create temporary logging file
os.makedirs("./SJtools/copilot/temp/",exist_ok=True)
tempid = ''.join([random.choice(string.ascii_lowercase+string.digits) for _ in range(10)])
tempPath = "./SJtools/copilot/temp/"+tempid+".txt"
tempFile = open(tempPath, 'w')
tempFile.seek(0)
tempFile.truncate()
print(f"logging to temp/{tempid}.txt")

# delete temporary logging file
import atexit
# atexit.register(os.remove, tempPath)

def find_and_write(runFilename,text2find,text2append,replace=False):
    """
    opens file, find line with text, write 
    runFileName: filepath
    fitext2findnd: text you want to find
    text2append: text you want to write
    return newly written string
    """
    with open(runFilename, "r+") as file:
        lines = file.readlines()
        for i,line in enumerate(lines):
            if line == text2find:
                lineNumber = i
                break
        
        newText = execute[:-1] + f"{text2append}\n" # signifies done
        if replace: newText = text2append
        lines[lineNumber] = newText
        
        # write content again
        file.seek(0, 0)
        file.write("".join(lines))
        file.truncate()
    
    return newText

while True:

    # read lines
    runFilename = f'SJtools/copilot/{args.file}'
    with open(runFilename) as file:
        lines = file.readlines()
        lines = [line for line in lines if line.rstrip() != '']

    # ignore line with #* at the end
    execute = None
    for line in lines:
        if '#' in line: continue
        else:
            execute = line
            break

    if execute is None:
        print("Closing: Executed all commands")
        exit()

    # mark the one to be done with # IN PROGRESS {process ID}
    newText = find_and_write(runFilename,execute,f" #IN PROGRESS {tempid}")
    
    # execute line
    keyboard_exit_pressed = None
    try:
        output = subprocess.run(execute, shell=True, text=True,stderr=tempFile)
    except KeyboardInterrupt:
        keyboard_exit_pressed = True

    # extract runid written in temp file
    score = ""
    curriculum = {}
    with open(tempPath, "r+") as f:
        for l in f: 
            if "runid:" in l:
                runid = l[6:-1]; 
            elif "last model normal_target:" in l:
                lastTwoPeak = l[25:-1]
                score += lastTwoPeak
            elif "best model normal_target:" in l:
                bestTwoPeak = l[25:-1]
                score += bestTwoPeak
            elif "best model normal_target time to hit:" in l:
                bestTimeToHit = l[37:-1]
                score += " time =" + bestTimeToHit
            elif "best model normal_target extra distance travelled:" in l:
                bestDistTravelled = l[50:-1]
                score += " dist =" + bestDistTravelled
            elif "curriculum:" in l:
                key, value = l[12:].split(' ')
                curriculum[key]=value
        # make tempFile be written from zero again
        tempFile.truncate()
        tempFile.seek(0)
    
    curriculum = ' '.join([f"{k} {v}" for k,v in curriculum.items()])
    # find that line again. indicate done #DONE
    try:
        find_and_write(runFilename, newText, f"{execute[:-1]} #DONE {runid}{score} {curriculum}\n", replace=True)
    except:
        find_and_write(runFilename, newText, f"{execute[:-1]} #ERROR\n", replace=True)
    
    # give it additional time to quit if user wants to quit
    if keyboard_exit_pressed:
        quitMessage = f"\n{runid} quit"
        print('\033[91m' + '\033[1m' + quitMessage + '\033[0m')
        time.sleep(5)

