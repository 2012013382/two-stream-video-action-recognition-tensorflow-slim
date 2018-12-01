import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
from tensorflow.contrib.slim.nets import nets_factory

def two_stream_model(rgb_image, flow_image, model_name, num_classes, keep_prob, batch_size, FRAMES_PER_VIDEO, is_training=True):
    #build model by slim.
    network_fn = nets_factory.get_network_fn(model_name, num_classes=num_classes, is_training=is_training)
    with tf.variable_scope('rgb_model'):
        _, rgb_end_points = network_fn(images=rgb_image)
    with tf.variable_scope('flow_model'):
        _, flow_end_points = network_fn(images=flow_image)

    with tf.variable_scope('two_stream'):
        #rgb stream
        with tf.variable_scope('rgb_stream'):
            if model_name == 'vgg_16':
                rgb_logits = rgb_end_points['rgb_model/vgg_16/fc8']
            elif model_name == 'resnet_v1_50':
                rgb_logits = rgb_end_points['rgb_model/resnet_v1_50/logits']
		rgb_logits = tf.squeeze(rgb_logits, [1, 2])

        #flow stream
        with tf.variable_scope('flow_stream'):
            if model_name == 'vgg_16':
                flow_logits = flow_end_points['flow_model/vgg_16/fc8/suqeezed']
            elif model_name == 'resnet_v1_50':
                flow_logits = flow_end_points['flow_model/resnet_v1_50/logits']
		flow_logits = tf.squeeze(flow_logits, [1, 2])

    return rgb_logits, flow_logits
