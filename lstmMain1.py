import TrafficJamPredictLSTM as lstm
import os,csv,platform
import numpy as np
import matplotlib.pyplot as plt
def isWindows():
    return "Windows" in platform.system()
def isLinux():
    return "Linux" in platform.system()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
obj=lstm.LSTM()
initial=0
BASETRAINDATAPATH="data\\train"
BASEMODELPATH="data\\model\\LSTM"
filenames=os.listdir(BASETRAINDATAPATH)
iterations=1
INPUT_NUM=500
LSTM_CELL_NUM=2
OUTPUT_NUM=120
HIDDEN_NUM = 2
LSTMCELL_NUM = 1 + 1
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
        obj.train(tci,INPUT_NUM,HIDDEN_NUM,LSTM_CELL_NUM,OUTPUT_NUM,initial,iterations,modelInputPath,modelOutputPath,0.001)




#############################test##################################
obj.readDataAndTest(INPUT_NUM,OUTPUT_NUM,"data\\test",BASEMODELPATH)



##############################simulink##################################





