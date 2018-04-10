import os,csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import TrafficJamPredict
obj=TrafficJamPredict.TrafficJamPredict()
obj.readDataAndTrain(modelSavePath="data\\model\\",trainDataFilePath="data\\train\\")
obj.readDataAndTestAccuracy(modelBasePath="data\\model\\",testDataFilePath="data\\train\\")