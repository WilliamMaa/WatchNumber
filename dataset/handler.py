# File name: handler.py
# Created by Haodong Chen
# on June 18, 2019
# All rights reserved.

import random

def newDataSet(inName, outName, num, p = 0.5, ignore = 0):
    inFile = open("../dataset/" + inName, 'r')
    outFile = open("../dataset/" + outName, 'w')
    count = 0
    while count < ignore:
        inFile.readline()
    count = 0
    while count < num:
        line = inFile.readline()
        if random.random() < p:
            outFile.write(line)
            count += 1
    inFile.close()
    outFile.close()
