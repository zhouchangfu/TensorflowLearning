import tensorflow as tf
import numpy as np
def averageFilter(shape):
    return tf.ones(shape=shape)/3
def kernelA(shape):
     kel=np.ones(shape=shape,dtype=np.float32)
     kel[0][0][0][0]=-1
     kel[0][0][0][0] = 2
     kel[0][0][0][0] = -1
     return tf.Variable(kel)