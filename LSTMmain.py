import TrafficJamPredictLSTM

obj=TrafficJamPredictLSTM.LSTM()
obj.readDataAndTrain(15,1,trainDataFilePath="data\\train\\",modelSavePath="data\\model\\LSTM\\")