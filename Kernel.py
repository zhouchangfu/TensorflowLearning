import tensorflow as tf
import numpy as np
def averageFilter(shape):
    return tf.ones(shape=shape)/3