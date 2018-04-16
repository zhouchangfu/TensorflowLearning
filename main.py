import os,csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import TrafficJamPredict
obj=TrafficJamPredict.TrafficJamPredict()
#obj.readDataAndTrain(inputNum=15,outputNum=1,learning_rate=0.001,modelSavePath="data\\model\\",trainDataFilePath="data\\train\\")
#obj.readDataAndPrivateTrain(inputNum=15,outputNum=1,modelSavePath="data\\model\\",trainDataFilePath="data\\train\\")
obj.readDataAndTestAccuracy(inputNum=15,outputNum=1,modelBasePath="data\\model\\",testDataFilePath="data\\train\\")


# import TrafficJamPredictLSTM
#
# obj=TrafficJamPredictLSTM.LSTM()
# obj.readDataAndTrain(50,1,trainDataFilePath="data\\train\\",modelSavePath="data\\model\\LSTM\\")