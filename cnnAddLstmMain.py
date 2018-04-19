import TrafficJamPredictCNNADDLSTM as cnnAddLstm
import os,csv,platform
import numpy as np
import matplotlib.pyplot as plt
def isWindows():
    return "Windows" in platform.system()
def isLinux():
    return "Linux" in platform.system()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
obj=cnnAddLstm.cnnAddLstm()
initial=0
BASETRAINDATAPATH="data\\train"
BASEMODELPATH="data\\model"
filenames=os.listdir(BASETRAINDATAPATH)
iterations=1000
INPUT_NUM=300
LSTM_CELL_NUM=2
OUTPUT_NUM=100
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
        if isWindows():
            modelInputPath=modelOutputPath=BASEMODELPATH+"\\"+filename+".ckpt"
        if isLinux():
            modelInputPath=modelOutputPath=modelOutputPath.replace("\\",'/')
        obj.train(tci,INPUT_NUM,LSTM_CELL_NUM,OUTPUT_NUM,initial,iterations,modelInputPath,modelOutputPath,0.001)




#############################test##################################
obj.readDataAndTest(INPUT_NUM,LSTM_CELL_NUM,OUTPUT_NUM,"data\\test","data\\model")



##############################simulink##################################

with open("data\\train\\I5-N-1205157.csv", "r") as fd:
    csv_data = csv.reader(fd)
    tci = [rows[1] for rows in csv_data]
    tci = tci[1:]
    tci = [float(num) for num in tci]
    for i in range(len(tci)-INPUT_NUM-OUTPUT_NUM+1):
        inputList=tci[i:i+INPUT_NUM]
        labelList=tci[i+INPUT_NUM:i+INPUT_NUM+OUTPUT_NUM]
        input=[]
        for item in inputList:
            input.append([item])
        input=[[input]]
        predictList=obj.simulink(input,INPUT_NUM,LSTM_CELL_NUM,OUTPUT_NUM,modelInputPath="data\\model\\I5-N-1205157.csv.ckpt")
        fig1=plt.figure("simulink")
        fig=fig1.add_subplot(111)
        x=np.linspace(-1,1,len(labelList))
        plt.plot(x,labelList,label="label")
        plt.plot(x,predictList[0],label="predict")
        plt.legend(loc='upper right')
        fig1.savefig("data\\lossImage\\"+str(i)+".png")
        plt.cla()




