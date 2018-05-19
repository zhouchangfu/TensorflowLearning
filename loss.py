import tensorflow as tf
class Loss:
    def mae(self,predict,label):
        lossOut = tf.abs(tf.reduce_mean(tf.abs(predict - label)), name="mae")
        return lossOut;
    def rmse(self,predict,label):
        lossOut = tf.sqrt(tf.reduce_mean(tf.square(predict-label),name="rmse"))
        return lossOut;
    def mape(self,predict,label):
        lossOut = tf.reduce_mean(tf.div(tf.abs(predict - label),label), name="mape")*100
        return lossOut;

