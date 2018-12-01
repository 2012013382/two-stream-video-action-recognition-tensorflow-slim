#===============================================================================
#Conv2d shape of first layer for flow stream is [7, 7, 20, 64], weights are obtained
#from the mean of first conv2d [7, 7, 3, 64]
#===============================================================================
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
TRAIN_CHECK_POINT = 'check_point/'
with tf.Graph().as_default():
    with tf.variable_scope('resnet_v1_50/conv1'):
        kernel = tf.Variable(tf.truncated_normal([7, 7, 3, 64], dtype=tf.float32), name='weights')
    with tf.variable_scope('flow_model/resnet_v1_50/conv1'):
        flow_kernel = tf.Variable(tf.truncated_normal([7, 7, 20, 64], dtype=tf.float32), name='weights')

    restorer = tf.train.Saver([kernel])
    saver = tf.train.Saver([flow_kernel])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, '../check_point/resnet_v1_50.ckpt')
        k = sess.run(kernel)
        fk = np.zeros((7, 7, 20, 64)).astype(np.float32)
        k_mean = np.zeros((7, 7, 1, 64)).astype(np.float32)
        for i in range(3):
            k_mean[:, :, 0, :] += k[:, :, i, :]
        k_mean /= 3
        for i in range(20):
            fk[:, :, i, :] = k_mean[:, :, 0, :]
        sess.run(tf.assign(flow_kernel, fk))
        saver.save(sess, '../' + TRAIN_CHECK_POINT + 'flow_inputs_weights_res_v1_50.ckpt')
