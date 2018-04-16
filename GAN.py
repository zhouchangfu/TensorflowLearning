import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
def GAN():
    #define the real artist
    #y=math.sin(x)
    with tf.variable_scope("Generator"):
        x=tf.placeholder(shape=[1,1],dtype=tf.float32)
        hidden=tf.layers.dense(inputs=x,units=20)
        gPredict=tf.layers.dense(hidden,1,activation=tf.nn.sigmoid)
    with tf.variable_scope("Discriminator"):
        #fake discriminate result
        D1=tf.layers.dense(gPredict,10)
        fakeArtD=tf.layers.dense(gPredict,1,activation=tf.nn.sigmoid)

        #the real result
        y=tf.placeholder(shape=[1,1],dtype=tf.float32)
        D2=tf.layers.dense(y,10)
        realArtD=tf.layers.dense(y,1,activation=tf.nn.sigmoid)

        #define D loss
    dLoss=-(tf.log(1-fakeArtD)*0.999+0.001*tf.log(realArtD))

    gLoss=tf.log(1-fakeArtD)

    gTrainStep=tf.train.AdamOptimizer().minimize(gLoss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator"))
    dTrainStep=tf.train.AdamOptimizer().minimize(dLoss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator"))
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        xValue=np.linspace(-10,10,10)
        yValue=np.sin(xValue)
        for ii in range(10000):
            simulinkValueList=[]
            for i in range(len(xValue)):
                for jj in range(10):
                    sess.run([gTrainStep,dTrainStep],feed_dict={x:[[xValue[i]]],y:[[yValue[i]]]})

                simulinkValueList.append(sess.run(gPredict,feed_dict={x:[[xValue[i]]]})[0][0])

                #print("loss:"+str(np.abs(np.sum(np.abs(np.array(simulinkValueList)-np.array(yValue))))))
            plt.cla()
            plt.plot(xValue, yValue)
            plt.legend("generator")
            plt.plot(xValue,simulinkValueList)
            plt.legend("discriminator")
            plt.pause(0.01)
GAN()




