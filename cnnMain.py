import TrafficJamPredict as cnn
import os,csv,platform
import numpy as np
import matplotlib.pyplot as plt
def isWindows():
    return "Windows" in platform.system()
def isLinux():
    return "Linux" in platform.system()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
obj=cnn.TrafficJamPredict()
initial=0
BASETRAINDATAPATH="data\\train"
BASEMODELPATH="data\\model\\CNN"
filenames=os.listdir(BASETRAINDATAPATH)
iterations=1000
INPUT_NUM=500
CONV_NUM=3
HIDDEN_NUM=3
OUTPUT_NUM=120
##############################train##################################
for filename in filenames:
    filePath=""
    if isWindows():
        filePath=BASETRAINDATAPATH+"\\"+filename
    if isLinux():
        filePath=filePath.replace("\\","/")

    with open(filePath, "r") as fd:
        csv_data = csv.reader(fd)
        tci = [rows[1] for rows in csv_data]
        tci = tci[1:]
        tci = [float(num) for num in tci]
        #for i in range(10):
        modelInputPath = modelOutputPath =""
        javaModelPath=""
        if isWindows():
            modelInputPath=modelOutputPath=BASEMODELPATH+"\\"+filename+".pd"
        if isLinux():
            modelInputPath=modelOutputPath=modelOutputPath.replace("\\",'/')
        obj.train_1(tci,INPUT_NUM,CONV_NUM,HIDDEN_NUM,OUTPUT_NUM,initial,iterations,modelInputPath,modelOutputPath,0.001)




#############################test##################################
obj.readDataAndTest(INPUT_NUM,OUTPUT_NUM,"data\\test",BASEMODELPATH)



##############################simulink##################################






