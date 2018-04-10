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
import random
import collections
import time


def LSTM():

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(50),rnn.BasicLSTMCell(30),rnn.BasicLSTMCell(30),rnn.BasicLSTMCell(3)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    x=tf.placeholder(shape=[1,5],dtype=tf.float32)
    y=tf.placeholder(shape=[1,3],dtype=tf.float32)
    #x=tf.split(x,3,0)


    outputs, states = tf.nn.static_rnn(rnn_cell, [x], dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    # output=outputs[0]
    # reshapeRel=tf.reshape(output,shape=[1,None])
    # demension=reshapeRel.shape.as_list()
    #
    # W=tf.Variable(np.random.random(shape=[demension[1],1]))
    # B=tf.Variable(np.random.random(shape=[1,1]))
    #
    # predict=tf.sigmoid(tf.matmul(reshapeRel*W)+B)
    predict=tf.layers.dense(inputs=outputs[0],units=3)

    loss=tf.losses.mean_squared_error(predictions=predict,labels=y)


    train_step=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
            sess.run(init)
            for i in range(5000):
                    sess.run(train_step,feed_dict={x:np.array([[1,2,3,4,5]]),y:np.array([[6,7,8]])})
                    print("loss:")
                    print(sess.run(loss,feed_dict={x:np.array([[1,2,3,4,5]]),y:np.array([[6,7,8]])}))
            print("predict:")
            print(sess.run(predict,feed_dict={x:np.array([[1,2,3,4,5]])}))



