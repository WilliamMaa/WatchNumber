# File name: app.py
# Created by Haodong Chen
# on June 14, 2019
# All rights reserved.

import time, numpy, pylab, threading
from matplotlib import pylab, pyplot as plt

r = threading.Lock()

def report(obj, count, result, timeSpent=(), isPercent=False):
    log = current()
    log += "\nTrained for " + str(count[0]) + " images, tested for " + str(count[1]) + " images.\nRecognizer:"
    log += "\n    Learning rate: " + str(obj.rate)
    log += "\n    Input node #:  " + str(obj.inputNodes)
    log += "\n    Hidden layers: " + str(obj.hiddenNodes)
    log += "\n    Output node #: " + str(obj.outputNodes)
    log += "\nAmong tested data,\n    " + (str(round(100.0 * result[0] / count[1], 3)) + " %" if isPercent else str(result[0]) + " out of " + str(count[1])) + " are correct.\n"
    if len(result) > 1:
        log += "    " + (str(round(100.0 * result[1] / count[1], 3)) + " %" if isPercent else str(result[1]) + " out of " + str(count[1])) + " are in the second choice.\n"
        if len(result) > 2:
            log += "    " + (str(round(100.0 * result[2] / count[1], 3)) + " %" if isPercent else str(result[2]) + " out of " + str(count[1])) + " are in the third choice.\n"
            if len(result) > 3:
                log += "    " + (str(round(100.0 * result[3] / count[1], 3)) + " %" if isPercent else str(result[3]) + " out of " + str(count[1])) + " are not included above.\n"
    if len(timeSpent) == 2:
        log += "Training took " + str(round(timeSpent[0], 3)) + " seconds, and testing took " + str(round(timeSpent[1], 3)) + " seconds.\n"
    log += "\n"
    print(log)
    with open("./test.log", 'a') as f:
        f.write(log)

def current():
    return time.strftime("%b. %d, %Y %a. %H:%M:%S", time.localtime())

def draw(arr):
    if len(arr) == 0:
        return False
    if len(arr) == 785:
        arr = arr[1:]
    side = round(len(arr) ** 0.5)
    if side * side == len(arr):
        plt.imshow(numpy.reshape(arr, (side, side)), cmap='gray')
        return True
    else:
        return False

def show(arr):
    if draw(arr):
        pylab.show()

# def draw(num, arr):
#     draw(arr)
#     pass

class stopwatch:
    # init
    def __init__(self):
        self.timestamp = time.time()
    # reset
    def reset(self):
        self.timestamp = time.time()
    def lap(self):
        return time.time() - self.timestamp
