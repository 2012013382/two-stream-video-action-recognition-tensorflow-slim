#===============================================================================
#Conv2d shape of first layer for flow stream is [3, 3, 20, 64], weights are obtained
#from the mean of first conv2d [3, 3, 3, 64]
#===============================================================================
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
TRAIN_CHECK_POINT = 'check_point/'
with tf.Graph().as_default():
    with tf.variable_scope('vgg_16/conv1/conv1_1'):
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='biases')
    with tf.variable_scope('flow_model/vgg_16/conv1/conv1_1'):
        flow_kernel = tf.Variable(tf.truncated_normal([3, 3, 20, 64], dtype=tf.float32), name='weights')
        flow_biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='biases')

    restorer = tf.train.Saver([kernel, biases])
    saver = tf.train.Saver([flow_kernel, flow_biases])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, '../check_point/vgg_16.ckpt')
        k, b = sess.run([kernel, biases])
        fk = np.zeros((3, 3, 20, 64)).astype(np.float32)
        k_mean = np.zeros((3, 3, 1, 64)).astype(np.float32)
        for i in range(3):
            k_mean[:, :, 0, :] += k[:, :, i, :]
        k_mean /= 3
        for i in range(20):
            fk[:, :, i, :] = k_mean[:, :, 0, :]
        sess.run(tf.assign(flow_kernel, fk))
        sess.run(tf.assign(flow_biases, b))
        saver.save(sess, '../' + TRAIN_CHECK_POINT + 'flow_inputs_weights.ckpt')
