#coding:utf-8
import tensorflow as tf
import InputData as inda
import Kernel as kernel
import numpy as np
import tool
import matplotlib.pyplot as plt
import datetime
class TrafficJamPredict:

    def train(self,data,inputNum,outputNum,state,iterations,modelInputPath,modelOutputFile,learning_rate):
        '''
            input:
            data: one-denmension list
            col: input data length
            state: value must 0 or 1 0:initial train the network 1 oppsite
            iterations: train times
            modelPath:nn model network structure exist path
            output:
            one value
            '''
        tf.reset_default_graph()
        batch=1
        row=1
        channel=1


        #label container
        label=tf.placeholder(tf.float32,shape=[1,1])
        predictValue=[]
        #input data container
        input=tf.placeholder(tf.float32,shape=[batch,row,inputNum,channel],name='input')


        filterKernel=kernel.averageFilter(shape=[1,3,1,1]);


        #convlution
        conv2dRel=tf.nn.conv2d(input,filter=filterKernel,strides=[1,1,1,1],data_format='NHWC',padding='SAME')

        #pooling
        maxpoolRel=tf.nn.max_pool(conv2dRel,ksize=[1,1,3,1],strides=[1,1,3,1],padding='SAME',data_format='NHWC')
        deminsion=maxpoolRel.shape.as_list()


        #reshape
        maxPoolFlat=tf.reshape(maxpoolRel,shape=[1,deminsion[0]*deminsion[1]*deminsion[2]*deminsion[3]])

        #Hidden layer
        Hidden=tf.layers.dense(maxPoolFlat,30)


        #output layer
        rel=tf.layers.dense(Hidden,outputNum,activation=tf.nn.tanh)


        #define the loss function
        loss=(tf.abs(label-rel))

        lossOut=tf.abs(label - rel)
        #loss=tf.losses.mean_squared_error(labels=label,predictions=rel)

        #set the train algorithm
        global_step = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=len(data) / inputNum,
                                                   decay_rate=0.998,
                                                   staircase=True)
        train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,global_step=global_step)
        #train_step=tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss=loss)

        #init all the weight value
        init=tf.global_variables_initializer()


        #begin training process
        with tf.Session() as sess:
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  #use this save the network model

                #save the data for using  tensorboard show the network structure

            sess.run(init)
              # 将图形、训练过程等数据合并在一起
            tf.summary.scalar("loss",loss[0][0])
            tf.summary.scalar("lossOut",lossOut[0][0])
            writer = tf.summary.FileWriter("G:/GraduationDesignModelData/logs/", sess.graph)
            merged = tf.summary.merge_all()
                #restore the all kinds of network weights to the cnn network
            if state == 0:
                        saver_path = saver.save(sess,modelOutputFile)
                        ##print('\033[1;32;47m',end=''
                        # 将模型保存到save/model.ckpt文件
                        print("\t\t\tmodel file initial in path : "+saver_path)
            ##print('\033[1;32;47m',end=''
            print("\t\t\tmodel file restore from path : " + modelInputPath)
            saver.restore(sess=sess,save_path=modelInputPath)
            lossValue=1
            for i in range(iterations):
                    #input and output data example
                    # inputData = inda.getInputData([batch, row, col, channel])
                    # labelData = [[-1]]
                    #compute the loss and improve the network  structure
                    lossValueList=[]
                    lossOutValueList=[]
                    simulinkValueList=[]
                    for j in range(len(data)-inputNum-outputNum+1):
                        inputList=data[j:j+inputNum]
                        inputData = inda.getInputData([batch, row, inputNum, channel])
                        for jj in range(len(inputList)):
                            inputData[0][0][jj][0]=inputList[jj]
                        labelValue=data[j+inputNum:j+inputNum+outputNum]
                        labelData=[labelValue]
                        sess.run(train_step,feed_dict={input:inputData,label:labelData})
                        lossValueList.append(sess.run(loss,feed_dict={input:inputData,label:labelData}))
                        lossOutValueList.append(sess.run(lossOut,feed_dict={input:inputData,label:labelData}))
                        simulinkValueList.append(sess.run(rel,feed_dict={input:inputData}))
                        trainProcess = sess.run(merged, feed_dict={input: inputData, label: labelData})
                        writer.add_summary(trainProcess, i*iterations+j)

                        # print("inputData:")
                        # print(inputList)
                        # print("conv2d:")
                        # print(sess.run(conv2dRel,feed_dict={input:inputData}))
                        # print("maxPoolResult:")
                        # print(sess.run(maxPoolFlat,feed_dict={input:inputData}))
                        # print("loss:")
                        # print(sess.run(loss,feed_dict={input:inputData,label:labelData}))
                #save the network weights to file
                    #print('\033[1;32;47m', end='')
                    # print("\t\t\tloss   :"+str(np.average(lossValueList)))
                    print("\t\t\tlossOut:"+str(np.average(lossOutValueList)))
                    lossValue=np.average(lossOutValueList)
                    #print("\t\t\tsimulinkValue:"+str(simulinkValueList[0]))
            saver_path = saver.save(sess, modelOutputFile)  # 将模型保存到save/model.ckpt文件
            ##print('\033[1;32;47m',end=''
            print("\t\t\tModel saved in file:", saver_path)
            return lossValue
    def printTensorValue(self,sess,feedDict,names,vars):
        for name in names:
            print(name)
            print(sess.run(name,feed_dict=feedDict))

    def readDataAndPrivateTrain(self,inputNum,outputNum,trainDataFilePath,modelSavePath):
        data_filename_set = tool.Tool().getPearsonClusterResult(trainDataFilePath=trainDataFilePath)
        data = data_filename_set[0]
        filenames = data_filename_set[1]
        filenames = [name.split('.')[0] for name in filenames]
        clusterRel = data_filename_set[2]
        tmpLi = [len(val) for val in clusterRel]
        iterations = max(tmpLi) * 4 * 2
        BASEPATH = modelSavePath
        innerIteration=10
        learning_rate=0.001
        IsPrintTrainTime=1
        for i in range(len(filenames)):
            modelPath = BASEPATH + filenames[i] + ".ckpt"
            starttime=datetime.datetime.now()
            lossValue1 = self.train(data[i], inputNum,outputNum, 1, innerIteration, modelPath, modelPath,
                                    learning_rate)
            endtime=datetime.datetime.now()
            averageTime = (endtime - starttime).seconds
            if IsPrintTrainTime == 1:
                print("train time cost about : " + str(len(filenames)* averageTime) + "(second)")
                IsPrintTrainTime = 0


    def readDataAndTrain(self,inputNum,outputNum,learning_rate,trainDataFilePath,modelSavePath):
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

            innerIteration=20
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



    def readDataAndTestAccuracy(self,inputNum,outputNum,modelBasePath,testDataFilePath):
        data_filenames=tool.Tool().getDataAndFilenames(testDataFilePath=testDataFilePath)
        data=data_filenames[0]
        filenames=data_filenames[1]
        filenames = [name.split('.')[0] for name in filenames]
        BASEPATH = modelBasePath
        for i in range(len(data)):
            modelPath=BASEPATH+filenames[i]+".ckpt"
            loss=self.testAccuracy(data[i],inputNum,outputNum,modelPath)
            #fig=plt.figure(filenames[i])
            #plt.hist(loss,100)
            print("loss:"+str(np.average(loss)))
            #fig.savefig("data//AliCloudTrain//"+filenames[i]+".png")
        #plt.show()


    def testAccuracy(self,data,col,outputNum,modelInputPath):
        '''
            input:
            data: one-denmension list
            col: input data length
            state: value must 0 or 1 0:initial train the network 1 oppsite
            iterations: train times
            modelPath:nn model network structure exist path
            output:
            one value
            '''
        tf.reset_default_graph()
        batch=1
        row=1
        channel=1


        #label container
        label=tf.placeholder(tf.float32,shape=[1,1])
        predictValue=[]
        #input data container
        input=tf.placeholder(tf.float32,shape=[batch,row,col,channel],name='input')


        filterKernel=kernel.averageFilter(shape=[3,1,1,1]);


        #convlution
        conv2dRel=tf.nn.conv2d(input,filter=filterKernel,strides=[1,1,1,1],data_format='NHWC',padding='SAME')

        #pooling
        maxpoolRel=tf.nn.max_pool(conv2dRel,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',data_format='NHWC')
        deminsion=maxpoolRel.shape.as_list()


        #reshape
        maxPoolFlat=tf.reshape(maxpoolRel,shape=[1,deminsion[0]*deminsion[1]*deminsion[2]*deminsion[3]])

        #Hidden layer
        hiddenW=tf.Variable(np.zeros(shape=[deminsion[0]*deminsion[1]*deminsion[2]*deminsion[3],deminsion[0]*deminsion[1]*deminsion[2]*deminsion[3]],dtype=np.float32),name='Hidden')
        hiddenBias=tf.Variable(np.zeros(shape=[1],dtype=np.float32),name='HiddenBias')
        Hidden=tf.nn.sigmoid(tf.matmul(maxPoolFlat,hiddenW)+hiddenBias)

        #output layer
        W=tf.Variable(np.zeros(shape=[deminsion[0]*deminsion[1]*deminsion[2]*deminsion[3],outputNum],dtype=np.float32),dtype=tf.float32,name='W')
        rel=tf.sigmoid(tf.matmul(Hidden,W)+tf.Variable(np.zeros(shape=[1],dtype=np.float32),name='OutputBias'));

        # define the loss function
        loss = tf.abs(rel-label)
        # init all the weight value
        init = tf.global_variables_initializer()

        # begin training process
        with tf.Session() as sess:
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  # use this save the network model

            # save the data for using  tensorboard show the network structure
            #tf.summary.FileWriter("G:/GraduationDesignModelData/logs/", sess.graph)
            sess.run(init)

            # restore the all kinds of network weights to the cnn network
            #print('\033[1;32;47m', end='')
            print("\t\t\tmodel file restore from path : " + modelInputPath)
            saver.restore(sess=sess, save_path=modelInputPath)
            lossValueList = []
            for j in range(len(data) - col):
                    inputList = data[j:j + col]
                    inputData = inda.getInputData([batch, row, col, channel])
                    for jj in range(len(inputList)):
                        inputData[0][0][jj][0] = inputList[jj]
                    labelValue = data[j + col:j + col + 1]
                    labelData = [labelValue]
                    lossValueList.append(sess.run(loss, feed_dict={input: inputData, label: labelData}))
                    # print("loss:")
                    # print(sess.run(rel,feed_dict={input:inputData}))
                    # print(sess.run(loss,feed_dict={input:inputData,label:labelData}))
                    # save the network weights to file
            #print('\033[1;32;47m', end='')
            print("\t\t\tloss:" + str(np.average(lossValueList)))
            return lossValueList

    def simulink(self,data,modelInputPath,col=50):
        tf.reset_default_graph()
        batch = 1
        row = 1
        channel = 1

        # label container
        label = tf.placeholder(tf.float32, shape=[1, 1])

        # input data container
        input = tf.placeholder(tf.float32, shape=[batch, row, col, channel], name='input')

        filterKernel = kernel.averageFilter(shape=[3, 1, 1, 1]);

        # convlution
        conv2dRel = tf.nn.conv2d(input, filter=filterKernel, strides=[1, 1, 1, 1], data_format='NHWC', padding='SAME')

        # pooling
        maxpoolRel = tf.nn.max_pool(conv2dRel, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                    data_format='NHWC')
        deminsion = maxpoolRel.shape.as_list()

        # reshape
        maxPoolFlat = tf.reshape(maxpoolRel, shape=[1, deminsion[0] * deminsion[1] * deminsion[2] * deminsion[3]])

        # Hidden layer
        hiddenW = tf.Variable(np.zeros(shape=[deminsion[0] * deminsion[1] * deminsion[2] * deminsion[3],
                                              deminsion[0] * deminsion[1] * deminsion[2] * deminsion[3]],
                                       dtype=np.float32), name='Hidden')
        hiddenBias = tf.Variable(np.zeros(shape=[1], dtype=np.float32), name='HiddenBias')
        Hidden = tf.nn.sigmoid(tf.matmul(maxPoolFlat, hiddenW) + hiddenBias)

        # output layer
        W = tf.Variable(
            np.zeros(shape=[deminsion[0] * deminsion[1] * deminsion[2] * deminsion[3], 1], dtype=np.float32),
            dtype=tf.float32, name='W')
        rel = tf.sigmoid(tf.matmul(Hidden, W) + tf.Variable(np.zeros(shape=[1], dtype=np.float32), name='OutputBias'));

        # define the loss function
        loss = tf.losses.mean_squared_error(labels=label, predictions=rel)
        # init all the weight value
        init = tf.global_variables_initializer()

        # begin training process
        with tf.Session() as sess:
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  # use this save the network model

            # save the data for using  tensorboard show the network structure
            # tf.summary.FileWriter("G:/GraduationDesignModelData/logs/", sess.graph)
            sess.run(init)

            # restore the all kinds of network weights to the cnn network
            # print('\033[1;32;47m', end='')
            print("\t\t\tmodel file restore from path : " + modelInputPath)
            saver.restore(sess=sess, save_path=modelInputPath)
            inputData = inda.getInputData([batch, row, col, channel])
            for jj in range(len(data)):
                inputData[0][0][jj][0] = data[jj]
            predictValue=sess.run(rel,feed_dict={input:inputData})
            return predictValue[0][0]





