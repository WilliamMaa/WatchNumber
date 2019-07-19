# File name: app.py
# Created by Haodong Chen
# on June 14, 2019
# All rights reserved.

from run import support, experiment
import threading

def main():
    # create locks
    expLogLock = threading.Lock()
    testLogLock = threading.Lock()
    fileLocks = (expLogLock, testLogLock)
    # create thread tasks
    tasks = []
    tasks.append(threading.Thread(target=run, args=(fileLocks, 3, 1, 0.1, 300)))
    tasks.append(threading.Thread(target=run, args=(fileLocks, 3, 2, 0.1, 300)))
    tasks.append(threading.Thread(target=run, args=(fileLocks, 3, 3, 0.1, 300)))
    tasks.append(threading.Thread(target=run, args=(fileLocks, 3, 1, 0.1, 300, 50)))
    tasks.append(threading.Thread(target=run, args=(fileLocks, 3, 2, 0.1, 300, 50)))
    tasks.append(threading.Thread(target=run, args=(fileLocks, 3, 3, 0.1, 300, 50)))
    # start running tasks
    for task in tasks:
        task.start()

def run(locks, runTimes, trainTimes, rate, *hiddenNodes):
    mean = 0
    log = support.current()
    log += "\nTrained for 180000 times and tested for 10000 times.\nRepeat times: " + str(runTimes)
    log += "\n    In: 784; Out: 10; \n    Learning rate: " + str(rate) + ";\n    Hidden layers: " + str(hiddenNodes) + ";\nAccuracy: "
    watch = support.stopwatch()
    for i in range(runTimes):
        accuracy = experiment.do(locks[1], rate, *hiddenNodes, loop=trainTimes)
        log += str(round(100.0 * accuracy, 2)) + " "
        mean += accuracy
    timeSpent = str(round(watch.lap(), 3))
    mean = 1.0 * mean / runTimes
    log += "\nAverage accuracy: " + str(round(100.0 * mean, 3)) + "\nAction takes " + timeSpent + " seconds."
    log += "\n\n"
    locks[0].acquire()
    with open("./experiment.log", 'a') as f:
        f.write(log)
    locks[0].release()
    print("Run", runTimes, "times with learning rate of", rate, "given the hidden layers as", hiddenNodes, "completed in " + timeSpent + " seconds.")

def test():
    # experiment.newDataset()
    pass

main()

