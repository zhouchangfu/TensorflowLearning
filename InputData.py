import numpy as np
import os
import csv
def getInputData(shape):
    return np.ones(dtype=np.float32,shape=shape)
def getInputLabel(shape):
    return np.ones(dtype=np.float32,shape=shape)
def getTrainAndTestData(len,fromPath,toTrainPath,toTestPath):
    PATH = fromPath
    filenames = os.listdir(PATH)
    data = []
    for filename in filenames:
        fullPathName = PATH + "\\" + filename
        with open(fullPathName, "r") as fd:
            csv_data = csv.reader(fd)
            csv_data=[row for row in csv_data]
            csvTrainData=csv_data[0:len]
            tmp=csv_data[len:]
            csvTestData=csv_data[0:1]
            for item in tmp:
                csvTestData.append(item)
            fullTrainPathName = toTrainPath+ filename
            fullTestOPathName=toTestPath+filename
            with open(fullTrainPathName,"w+",newline='') as fd:
                spamwriter = csv.writer(fd)
                spamwriter.writerows(csvTrainData)
            with open(fullTestOPathName, "w+", newline='') as fd:
                spamwriter = csv.writer(fd)
                spamwriter.writerows(csvTestData)
#getTrainAndTestData(2000,fromPath='TCI\\',toTrainPath="data\\train\\",toTestPath="data\\test\\")



