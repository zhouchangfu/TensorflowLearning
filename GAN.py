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
        gPredict=tf.layers.dense(hidden,1)
    with tf.variable_scope("Discriminator"):
        #fake discriminate result
        dDiscriminate=tf.layers.dense(gPredict,1)

        #the real result
        y=tf.placeholder(shape=[1,1],dtype=tf.float32)

        #define D loss
        dLoss=-tf.sigmoid(tf.losses.mean_squared_error(y,dDiscriminate))

        gLoss=tf.losses.mean_squared_error(y,dDiscriminate)

    gTrainStep=tf.train.AdamOptimizer().minimize(gLoss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator"))
    dTrainStep=tf.train.AdamOptimizer().minimize(dLoss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator"))
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        xValue=np.linspace(-10,10,1000)
        yValue=np.sin(xValue)
        for i in range(10000):
            simulinkValueList=[]
            for i in range(len(xValue)):
                sess.run(gTrainStep,feed_dict={x:[[xValue[i]]],y:[[yValue[i]]]})

                sess.run(dTrainStep, feed_dict={x: [[xValue[i]]], y: [[yValue[i]]]})

                simulinkValueList.append(sess.run(gPredict,feed_dict={x:[[xValue[i]]]})[0][0])
            if (i+1)%1==0:
                plt.cla()
                plt.plot(xValue, yValue)
                plt.legend("generator")
                plt.plot(xValue,simulinkValueList)
                plt.legend("discriminator")
                plt.pause(0.01)
plt.show()
GAN()




