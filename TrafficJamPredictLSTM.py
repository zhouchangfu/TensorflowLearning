'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import tool
#import matplotlib.pyplot as plt
import datetime


class LSTM():
    def train(self,data,inputNum,hiddenNum,lstmCellNum,outputNum,state,innerIterations,modelInputPath,modelOutputPath,learningRate):
        tf.reset_default_graph()
        # 2-layer LSTM, each layer has n_hidden units.
        # Average Accuracy= 95.20% at 50k iter
        everyLstmCellNums=400
        everyHiddenUnitNums=400

        listCells=[]
        for i in range(lstmCellNum):
            listCells.append(rnn.BasicLSTMCell(everyLstmCellNums))
        rnn_cell = rnn.MultiRNNCell(listCells)

        # 1-layer LSTM with n_hidden units but with lower accuracy.
        # Average Accuracy= 90.60% 50k iter
        # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
        # rnn_cell = rnn.BasicLSTMCell(n_hidden)

        # generate prediction
        x=tf.placeholder(shape=[None,inputNum],dtype=tf.float32)
        y=tf.placeholder(shape=[None,outputNum],dtype=tf.float32)
        #x=tf.split(x,3,0)

        hiddenInput =x
        hiddenOutput = x
        for i in range(hiddenNum):
            hiddenOutput = tf.layers.dense(inputs=hiddenInput, units=everyHiddenUnitNums)
            hiddenInput = hiddenOutput

        outputs, states = tf.nn.static_rnn(rnn_cell, [hiddenOutput], dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        # output=outputs[0]
        # reshapeRel=tf.reshape(output,shape=[1,3])
        # demension=reshapeRel.shape.as_list()
        # #
        # W=tf.Variable(tf.random_normal(shape=[demension[1],outputNum],mean=0.0,stddev=50.0))
        # B=tf.Variable(tf.random_normal(shape=[1,1]),mean=0.0,stddev=50.0)
        # #
        # predict=tf.sigmoid(tf.matmul(reshapeRel,W)+B)

        predict=tf.layers.dense(inputs=outputs[0],units=outputNum,activation=tf.nn.sigmoid)

        loss=tf.abs(tf.reduce_sum(tf.abs(predict-y)))
        lossOut=tf.abs(tf.reduce_mean(tf.abs(predict-y)))

        global_step=tf.Variable(0)

        learning_rate = tf.train.exponential_decay(learningRate, global_step, decay_steps=innerIterations/100, decay_rate=0.9,
                                                   staircase=True)
        train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,global_step=global_step)
        init=tf.global_variables_initializer()
        with tf.Session() as sess:

                saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  # use this save the network model

                # save the data for using  tensorboard show the network structure
                #tf.summary.histogram("W",W)
                tf.summary.scalar("loss",loss)
                tf.summary.scalar("lossOut",lossOut)
                merged=tf.summary.merge_all()
                writer=tf.summary.FileWriter("G:/GraduationDesignModelData/logs/", sess.graph)
                sess.run(init)

                # restore the all kinds of network weights to the cnn network
                if state == 0:
                    saver_path = saver.save(sess, modelOutputPath)
                    ##print('\033[1;32;47m',end=''
                    # 将模型保存到save/model.ckpt文件
                    print("\t\t\tmodel file initial in path : " + saver_path)
                ##print('\033[1;32;47m',end=''
                print("\t\t\tmodel file restore from path : " + modelInputPath)
                saver.restore(sess=sess, save_path=modelInputPath)
                retLossValue=None
                inputMatrix,labelMatrix=self.convertToMatrixData(data,inputNum,outputNum)
                for i in range(innerIterations):
                    sess.run(train_step,feed_dict={x:inputMatrix,y:labelMatrix})
                    retLossValue=sess.run(lossOut,feed_dict={x:inputMatrix,y:labelMatrix})
                    print("lossOut:"+str(retLossValue))
                return retLossValue

    def convertToMatrixData(self,data,inputNum,outputNum):
        inputMatrix=[]
        labelMatrix=[]
        for i in range(len(data)-inputNum-outputNum+1):
            inputMatrix.append(data[i:i+inputNum])
            labelMatrix.append(data[i+inputNum:i+inputNum+outputNum])
        return [inputMatrix,labelMatrix]
    def readDataAndTrain(self,inputNum,outputNum,trainDataFilePath,modelSavePath):
        data_filename_set=tool.Tool().getPearsonClusterResult(trainDataFilePath=trainDataFilePath)
        data=data_filename_set[0]
        filenames=data_filename_set[1]
        filenames=[name.split('.')[0] for name in filenames]
        clusterRel=data_filename_set[2]
        tmpLi=[len(val) for val in clusterRel]
        iterations=max(tmpLi)*2
        BASEPATH=modelSavePath
        allListLen = [len(li) for li in clusterRel]
        allAverageTimeList = [iterations / (len(li) + 1) for li in clusterRel]
        for i in range(len(clusterRel)):
            initialState = 0
            tmpList=clusterRel[i]
            tmpModelPath=BASEPATH+"tmpModel.ckpt"
            averageForTimes=int(iterations/(len(tmpList)+1))
            tip="-".join([str(i)for i in tmpList])
            #print('\033[1;31;47m',end='')
            print("g:"+tip)
            IsPrintTrainTime=1

            learning_rate=0.001
            innerIteration=50
            averageTime=-1

            for ii in range(averageForTimes):
                #print('\033[1;31;47m',end='')
                print("\t"+tip+" group train :"+str(ii))
                for j in range(len(tmpList)):
                    #print('\033[1;31;47m',end='')
                    print("\t\t begin to train : "+str(tmpList[j])+"<filename>:"+filenames[tmpList[j]])
                    starttime = datetime.datetime.now()
                    lossValue=self.train(data[tmpList[j]],inputNum,outputNum,initialState,innerIteration,tmpModelPath,tmpModelPath,learning_rate)
                    #learning_rate=lossValue/20
                    endtime = datetime.datetime.now()
                    averageTime=(endtime-starttime).seconds
                    if initialState == 0:
                        initialState=1
                    if IsPrintTrainTime == 1:
                        print("train time cost about : " + str(np.sum(np.array(allListLen)*np.array(allAverageTimeList)*2)*averageTime) + "(second)")
                        IsPrintTrainTime = 0
                #learning_rate=learning_rate/2
            for j in range(len(tmpList)):
                initialState=0
                outputModelPath=BASEPATH+filenames[tmpList[j]]+".ckpt"
                #print('\033[1;31;47m',end='')
                print("\tbegin to train personal:"+str(tmpList[j]))
                for ii in range(averageForTimes):
                    #print('\033[1;31;47m',end='')
                    print("\t\t private train times:"+str(ii))
                    if initialState == 0:
                        lossValue1=self.train(data[tmpList[j]], inputNum,outputNum, initialState, innerIteration, tmpModelPath, outputModelPath,learning_rate)
                        #learning_rate=lossValue1/20
                        initialState = 1
                    else:
                        lossValue1=self.train(data[tmpList[j]], inputNum,outputNum, initialState, innerIteration, outputModelPath,
                                         outputModelPath,learning_rate)
                        #learning_rate=lossValue1/20
        print("train over!")


# obj=LSTM()
# obj.readDataAndTrain(15,1,trainDataFilePath="data\\train\\",modelSavePath="data\\model\\LSTM\\")