import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os
from data_loader import get_batches
from data_loader import get_frame_indices
import argparse
from nets import nets_factory
from model import two_stream_model
from data_loader import IMG_HEIGHT
from data_loader import IMG_WIDTH
from data_loader import IMG_RGB_CHANNEL
from data_loader import IMG_FLOW_CHANNEL
from data_loader import FRAMES_PER_VIDEO
TRAIN_RGB_PATH = '../cache_data/train_rgb.txt'
TRAIN_FLOW_PATH = '../cache_data/train_flow.txt'
VALIDATION_RGB_PATH = '../cache_data/validation_rgb.txt'
VALIDATION_FLOW_PATH = '../cache_data/validation_flow.txt'
TEST_RGB_PATH = '../cache_data/test_rgb.txt'
TEST_FLOW_PATH = '../cache_data/test_flow.txt'
MODEL_DIR = '../check_point/flow_trained.ckpt'

parser = argparse.ArgumentParser(description='test UCF101 flow stream.')
parser.add_argument('--network', default='resnet_v1_50', type=str, help='network name')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--class_number', default=101, type=int, help='class number')
parser.add_argument('--keep_prob', default=0.5, type=float, help='keep prob')
args = parser.parse_args()
print(args)

train_video_indices, validation_video_indices, test_video_indices = get_frame_indices(TRAIN_RGB_PATH,
                                                                                      VALIDATION_RGB_PATH,
                                                                                      TEST_RGB_PATH)
    
def test_fusion():
    with tf.Graph().as_default() as g:
        flow_image = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_FLOW_CHANNEL], name='flow_image')
        label = tf.placeholder(tf.int32, [None, args.class_number], name = 'label')
        is_training = tf.placeholder(tf.bool)
        flow_logits = two_stream_model('None', 
                                       flow_image, 
                                       args.network, 
                                       args.class_number, 
                                       args.keep_prob,
                                       args.batch_size,
                                       FRAMES_PER_VIDEO,
                                       is_training,
                                       'flow')
        #Loss
        with tf.name_scope('loss'):
            flow_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flow_logits, labels=label))

        with tf.name_scope('accuracy'):
            flow_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(flow_logits, 1), tf.argmax(label, 1)), tf.float32))
        
        restorer = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restorer.restore(sess, MODEL_DIR)
            g.finalize()
            
            ls_epoch = 0
            acc_epoch = 0
            batch_index = 0
            v_step = 0
            for i in range(len(test_video_indices) // args.batch_size):
                v_step += 1
                if i % 20 == 0:
                    print('{} / {}'.format(i, len(test_video_indices) // args.batch_size))
                flow_batch_data, batch_index = get_batches('None',
                                                           TEST_FLOW_PATH,
                                                           args.batch_size,
                                                           test_video_indices,
                                                           batch_index,
                                                           'flow')
                ls, acc = sess.run([flow_loss, flow_accuracy], feed_dict={flow_image: flow_batch_data['images'],
                                                                          label: flow_batch_data['labels'],
                                                                          is_training: False})
                ls_epoch += ls
                acc_epoch += acc
            print('Loss {}, acc {}'.format(ls_epoch / v_step, acc_epoch / v_step))
if __name__ == "__main__":
    test_fusion()
