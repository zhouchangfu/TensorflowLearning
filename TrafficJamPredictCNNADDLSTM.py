import tensorflow as tf
import InputData as inda
import Kernel as kernel
import numpy as np
import tool
import loss as lossObj
import os,csv
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import datetime
import platform
def isWindows():
    return "Windows" in platform.system()
def isLinux():
    return "Linux" in platform.system()
class cnnAddLstm():
    def train_1(self,data,inputNum,convnum,lstmCellNum,outputNum,state,innerIterations,modelInputPath,modelOutputPath,javaModelPath,learningRate):
        tf.reset_default_graph()
        everyLstmCellNums=400

        label = tf.placeholder(tf.float32, shape=[None, outputNum],name="label")
        predictValue = []
        # input data container
        input = tf.placeholder(tf.float32, shape=[None, 1, inputNum, 1], name='input')

        filterKernel = kernel.kernelA(shape=[1, 3, 1, 1]);

        # convlution
        convInput = input;
        convOutput = input;
        for i in range(convnum):
            convOutput = tf.nn.conv2d(input, filter=filterKernel, strides=[1, 1, 1, 1], data_format='NHWC', padding='SAME')
            convInput = convOutput


        # pooling
        maxpoolRel = tf.nn.max_pool(convOutput, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME',
                                    data_format='NHWC')
        demensions=maxpoolRel.shape.as_list()
        ####0:batch,1H,2W,3C

        maxPoolRelFlat=tf.reshape(maxpoolRel,shape=[-1,demensions[2]])

        listCells = []
        for i in range(lstmCellNum):
            listCells.append(rnn.BasicLSTMCell(everyLstmCellNums))
        rnn_cell = rnn.MultiRNNCell(listCells)

        outputs, states = tf.nn.static_rnn(rnn_cell, [maxPoolRelFlat], dtype=tf.float32)

        predict = tf.layers.dense(inputs=outputs[0], units=outputNum)

        predict=tf.nn.sigmoid(predict,name="predict")

        MAE = lossObj.Loss().mae(predict, label);
        MAPE = lossObj.Loss().mape(predict, label);
        RMSE = lossObj.Loss().rmse(predict, label);
        loss = tf.abs(tf.reduce_mean(tf.abs(predict - label)))
        lossOut = tf.abs(tf.reduce_mean(tf.abs(predict - label)),name="abs_mean_loss")

        global_step = tf.Variable(0)

        learning_rate=tf.train.exponential_decay(learningRate, global_step, decay_steps=innerIterations / 20,
                                                   decay_rate=0.5,
                                                   staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  # use this save the network model

            sess.run(init)
            print("begin to train:" + modelInputPath)
            retLossValue = None
            inputMatrix, labelMatrix = self.convertToMatrixData(data, inputNum, outputNum)
            for i in range(innerIterations):
                sess.run(train_step, feed_dict={input: inputMatrix, label: labelMatrix})
                retLossValue = sess.run(lossOut, feed_dict={input: inputMatrix, label: labelMatrix})
                learningRate=sess.run(learning_rate,feed_dict={input: inputMatrix, label: labelMatrix})


                print("global_step:"+str(sess.run(global_step))+"\tlossOut:" + str(retLossValue)+"\tlearningRate:"+str(learningRate)+"\tlossMul:"+str(learningRate*retLossValue))
            # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
            #                                                                 output_node_names=['predict'])
            # with tf.gfile.FastGFile(javaModelPath, mode='wb') as f:
            #     f.write(output_graph_def.SerializeToString())
            builder = tf.saved_model.builder.SavedModelBuilder(javaModelPath)
            tag_string=javaModelPath.split("\\")[-1]
            builder.add_meta_graph_and_variables(sess, [tag_string])
            builder.save()

            builder1 = tf.saved_model.builder.SavedModelBuilder(modelOutputPath)
            builder1.add_meta_graph_and_variables(sess, ['tag_string'])
            builder1.save()

            print(modelOutputPath + "  model save successfully")

    def test(self,data,inputNum,outputNum,modelPath,tagString):
        with tf.Session() as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, tagString, modelPath)
            input = sess.graph.get_tensor_by_name('input:0')
            label=sess.graph.get_tensor_by_name('label:0');
            mae=sess.graph.get_tensor_by_name("mae:0");
            mape=sess.graph.get_tensor_by_name("mape:0");
            rmse=sess.graph.get_tensor_by_name("rmse:0");
            inputMatrix, labelMatrix = self.convertToMatrixData(data, inputNum, outputNum);
            lossOut=sess.run([mae,mape,rmse],feed_dict={input:inputMatrix,label:labelMatrix})
            #print("_abs_mean_loss:"+str(_abs_mean_loss))
            return lossOut;
    def simulink_1(self,data,modelPath,tagString):
        with tf.Session() as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, tagString, modelPath)
            input = sess.graph.get_tensor_by_name('input:0')
            predict=sess.graph.get_tensor_by_name('predict:0')
            predictValue=sess.run([predict], feed_dict={input: data})
            return predictValue;

    def readDataAndTest_1(self, inputNum, outputNum, testDataBasePath, modelInputBasePath):
        filenames = os.listdir(testDataBasePath)
        for filename in filenames:
            modelInputPath = ""
            testDataPath = ""
            modelInputPath = modelInputBasePath + "\\" + filename + ".pd"
            testDataPath = testDataBasePath + "\\" + filename
            with open(testDataPath, "r") as fd:
                csv_data = csv.reader(fd)
                tci = [rows[1] for rows in csv_data]
                tci = tci[1:]
                tci = [float(num) for num in tci]
                tag_string = modelInputPath.split("\\")[-1]
                lossValue = self.test(tci, inputNum, outputNum, modelInputPath, ['tag_string'])
                print(filename + "(loss):" + str(lossValue))
    def train(self,data,inputNum,lstmCellNum,outputNum,state,innerIterations,modelInputPath,modelOutputPath,javaModelPath,learningRate):
        tf.reset_default_graph()
        everyLstmCellNums=400

        label = tf.placeholder(tf.float32, shape=[None, outputNum])
        predictValue = []
        # input data container
        input = tf.placeholder(tf.float32, shape=[None, 1, inputNum, 1], name='input')

        filterKernel = kernel.kernelA(shape=[1, 3, 1, 1]);

        # convlution
        conv2dRel = tf.nn.conv2d(input, filter=filterKernel, strides=[1, 1, 1, 1], data_format='NHWC', padding='SAME')

        # pooling
        maxpoolRel = tf.nn.max_pool(conv2dRel, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME',
                                    data_format='NHWC')
        demensions=maxpoolRel.shape.as_list()
        ####0:batch,1H,2W,3C

        maxPoolRelFlat=tf.reshape(maxpoolRel,shape=[-1,demensions[2]])

        listCells = []
        for i in range(lstmCellNum):
            listCells.append(rnn.BasicLSTMCell(everyLstmCellNums))
        rnn_cell = rnn.MultiRNNCell(listCells)

        outputs, states = tf.nn.static_rnn(rnn_cell, [maxPoolRelFlat], dtype=tf.float32)

        predict = tf.layers.dense(inputs=outputs[0], units=outputNum)

        predict=tf.nn.sigmoid(predict,name="predict")

        loss = tf.abs(tf.reduce_mean(tf.abs(predict - label)))
        lossOut = tf.abs(tf.reduce_mean(tf.abs(predict - label)))

        global_step = tf.Variable(0)

        learning_rate=tf.train.exponential_decay(learningRate, global_step, decay_steps=innerIterations / 20,
                                                   decay_rate=0.5,
                                                   staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  # use this save the network model

            # save the data for using  tensorboard show the network structure
            # tf.summary.histogram("W",W)
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("lossOut", lossOut)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("G:/GraduationDesignModelData/logs/", sess.graph)
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
            retLossValue = None
            inputMatrix, labelMatrix = self.convertToMatrixData(data, inputNum, outputNum)
            for i in range(innerIterations):
                sess.run(train_step, feed_dict={input: inputMatrix, label: labelMatrix})
                retLossValue = sess.run(lossOut, feed_dict={input: inputMatrix, label: labelMatrix})
                learningRate=sess.run(learning_rate,feed_dict={input: inputMatrix, label: labelMatrix})


                print("global_step:"+str(sess.run(global_step))+"\tlossOut:" + str(retLossValue)+"\tlearningRate:"+str(learningRate)+"\tlossMul:"+str(learningRate*retLossValue))
            saver.save(sess=sess,save_path=modelOutputPath)
            print("\t\t\tmodel save to path:"+modelOutputPath)
            # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
            #                                                                 output_node_names=['predict'])
            # with tf.gfile.FastGFile(javaModelPath, mode='wb') as f:
            #     f.write(output_graph_def.SerializeToString())
            builder = tf.saved_model.builder.SavedModelBuilder(javaModelPath)
            tag_string=javaModelPath.split("\\")[-1]
            builder.add_meta_graph_and_variables(sess, [tag_string])
            builder.save()

            return retLossValue


    def convertToMatrixData(self,data,inputNum,outputNum):
            inputMatrix=[]
            labelMatrix=[]
            for i in range(len(data)-inputNum-outputNum+1):
                inputList=data[i:i+inputNum]
                labelList=data[i+inputNum:i+inputNum+outputNum]
                inputList1=[]
                for item in inputList:
                    inputList1.append([item])
                inputMatrix.append([inputList1])
                labelMatrix.append(labelList)
            return [inputMatrix,labelMatrix]
    def testAccuracy(self,data,inputNum,lstmCellNum,outputNum,modelInputPath):
        tf.reset_default_graph()
        everyLstmCellNums = 400

        label = tf.placeholder(tf.float32, shape=[None, outputNum])
        predictValue = []
        # input data container
        input = tf.placeholder(tf.float32, shape=[None, 1, inputNum, 1], name='input')

        filterKernel = kernel.kernelA(shape=[1, 3, 1, 1]);

        # convlution
        conv2dRel = tf.nn.conv2d(input, filter=filterKernel, strides=[1, 1, 1, 1], data_format='NHWC', padding='SAME')

        # pooling
        maxpoolRel = tf.nn.max_pool(conv2dRel, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME',
                                    data_format='NHWC')
        demensions = maxpoolRel.shape.as_list()
        ####0:batch,1H,2W,3C

        maxPoolRelFlat = tf.reshape(maxpoolRel, shape=[-1, demensions[2]])

        listCells = []
        for i in range(lstmCellNum):
            listCells.append(rnn.BasicLSTMCell(everyLstmCellNums))
        rnn_cell = rnn.MultiRNNCell(listCells)

        outputs, states = tf.nn.static_rnn(rnn_cell, [maxPoolRelFlat], dtype=tf.float32)

        predict = tf.layers.dense(inputs=outputs[0], units=outputNum)

        predict = tf.nn.sigmoid(predict, name="predict")

        lossOut = tf.abs(tf.reduce_mean(tf.abs(predict - label)))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  # use this save the network model

            # save the data for using  tensorboard show the network structure
            # tf.summary.histogram("W",W)

            sess.run(init)
            print("\t\t\tmodel file restore from path : " + modelInputPath)
            saver.restore(sess=sess, save_path=modelInputPath)
            retLossValue = None
            inputMatrix, labelMatrix = self.convertToMatrixData(data, inputNum, outputNum)

            retLossValue = sess.run(lossOut, feed_dict={input: inputMatrix, label: labelMatrix})

            return retLossValue

    def readDataAndTest(self,inputNum,lstmCellNum,outputNum,testDataBasePath,modelInputBasePath):
        filenames=os.listdir(testDataBasePath)
        for filename in filenames:
            modelInputPath = ""
            testDataPath=""
            if isWindows():
                modelInputPath=modelInputBasePath+"\\"+filename+".ckpt"
                testDataPath =testDataBasePath+"\\"+filename
            if isLinux():
                modelInputPath=modelInputPath.replace("\\",'/')
                testDataPath=testDataPath.replace("\\",'/')
            with open(testDataPath, "r") as fd:
                csv_data = csv.reader(fd)
                tci = [rows[1] for rows in csv_data]
                tci = tci[1:]
                tci = [float(num) for num in tci]
                lossValue=self.testAccuracy(tci,inputNum,lstmCellNum,outputNum,modelInputPath)
                print(filename+"(loss):"+str(lossValue))
    def simulink(self,inputData,inputNum,lstmCellNum,outputNum,modelInputPath):
        tf.reset_default_graph()
        everyLstmCellNums = 400

        label = tf.placeholder(tf.float32, shape=[None, outputNum])
        predictValue = []
        # input data container
        input = tf.placeholder(tf.float32, shape=[None, 1, inputNum, 1], name='input')

        filterKernel = kernel.kernelA(shape=[1, 3, 1, 1]);

        # convlution
        conv2dRel = tf.nn.conv2d(input, filter=filterKernel, strides=[1, 1, 1, 1], data_format='NHWC', padding='SAME')

        # pooling
        maxpoolRel = tf.nn.max_pool(conv2dRel, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME',
                                    data_format='NHWC')
        demensions = maxpoolRel.shape.as_list()
        ####0:batch,1H,2W,3C

        maxPoolRelFlat = tf.reshape(maxpoolRel, shape=[-1, demensions[2]])

        listCells = []
        for i in range(lstmCellNum):
            listCells.append(rnn.BasicLSTMCell(everyLstmCellNums))
        rnn_cell = rnn.MultiRNNCell(listCells)

        outputs, states = tf.nn.static_rnn(rnn_cell, [maxPoolRelFlat], dtype=tf.float32)

        predict = tf.layers.dense(inputs=outputs[0], units=outputNum)

        predict = tf.nn.sigmoid(predict, name="predict")


        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  # use this save the network model

            # save the data for using  tensorboard show the network structure
            # tf.summary.histogram("W",W)

            sess.run(init)
            print("\t\t\tmodel file restore from path : " + modelInputPath)
            saver.restore(sess=sess, save_path=modelInputPath)

            predictValue = sess.run(predict, feed_dict={input: inputData})

            return predictValue







