# File name: experiment.py
# Created by Haodong Chen
# on June 18, 2019
# All rights reserved.

import numpy
from run import neurons, support
from dataset import handler

def newDataset(trainT = 3000, testT = 1000, files=('mnist_debug','mnist_debug_test')):
    # generate files
    handler.newDataSet("mnist_train.csv", files[0], trainT if trainT <= 60000 else 60000, 55000.0 / trainT if trainT < 55000 else 1)
    handler.newDataSet("mnist_test.csv", files[1], testT if testT <= 10000 else 10000, 9000.0 / trainT if trainT < 9000 else 1)

def do(logLock, rate, *hiddenNodes, loop=1, files=('mnist_train.csv','mnist_test.csv')):

    # create network
    recognizer = neurons.NeuronsNetwork(rate, 784, 10, *hiddenNodes)
    # create stopwatch
    watch = support.stopwatch()

    # training
    trainCount = 0
    with open('../dataset/' + files[0], 'r') as trainSet:
        watch.reset()
        for t in range(loop):
            for l in trainSet:
                trainCount += 1
                data = numpy.asarray(l.split(','), 'int')
                target = numpy.zeros(10) + 0.1
                target[data[0]] = 0.99
                recognizer.train((numpy.asfarray(data[1:]) / 255.0 * 0.99 + 0.01), target)
        trainTime = watch.lap()

    # testing
    report = []
    testCount = 0
    correctFirst = correctSecond = correctThird = incorrect = 0
    with open('../dataset/' + files[1], 'r') as testSet:
        watch.reset()
        for l in testSet:
            testCount += 1
            data = numpy.asarray(l.split(','), 'int')
            result = recognizer.query(numpy.asfarray(data[1:]) / 255.0 * 0.99 + 0.01).tolist()
            first = second = third = 0
            for i in range(len(result)):
                p = result[i][0]
                if p > result[third][0]:
                    third = i
                    if p > result[second][0]:
                        third = second
                        second = i
                        if p > result[first][0]:
                            third = second
                            second = first
                            first = i
            if data[0] == first:
                correctFirst += 1
            elif data[0] == second:
                correctSecond += 1
            elif data[0] == third:
                correctThird += 1
            else:
                incorrect += 1
            report.append((data[0], first, result[first][0], second, result[second][0], third, result[third][0]))
        testTime = watch.lap()

    # output report
    logLock.acquire()
    support.report(
            recognizer,
            (trainCount, testCount),
            (correctFirst, correctSecond, correctThird, incorrect),
            (trainTime, testTime),
            True)
    logLock.release()

    return 1.0 * correctFirst / testCount
