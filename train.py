import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os
from data_loader import get_batches
from data_loader import get_frame_indices
import argparse
from model import two_stream_model
from data_loader import IMG_HEIGHT
from data_loader import IMG_WIDTH
from data_loader import IMG_RGB_CHANNEL
from data_loader import IMG_FLOW_CHANNEL
from data_loader import FRAMES_PER_VIDEO
TRAIN_LOG_DIR = os.path.join('./log/train/', time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
TRAIN_CHECK_POINT = './check_point/'
TRAIN_RGB_PATH = './cache_data/train_rgb.txt'
TRAIN_FLOW_PATH = './cache_data/train_flow.txt'
VALIDATION_RGB_PATH = './cache_data/validation_rgb.txt'
VALIDATION_FLOW_PATH = './cache_data/validation_flow.txt'
TEST_RGB_PATH = './cache_data/test_rgb.txt'
TEST_RGB_PATH = './cache_data/test_flow.txt'
VGG_16_MODEL_DIR = './check_point/vgg_16.ckpt'
RES_v1_50_MODEL_DIR = './check_point/resnet_v1_50.ckpt'
FLOW_INPUT_WEIGHTS_VGG_16 = './check_point/flow_inputs_weights.ckpt'
FLOW_INPUT_WEIGHTS_RES_v1_50 = './check_point/flow_inputs_weights_res_v1_50.ckpt'

#Default settings
parser = argparse.ArgumentParser(description='UCF101 two stream fusion')
parser.add_argument('--network', default='resnet_v1_50', type=str, help='network name')
parser.add_argument('--epoches', default=100, type=int, help='number of total epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='start learning rate')
parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--class_number', default=101, type=int, help='class number of UCF-101')
parser.add_argument('--keep_prob', default=0.85, type=float, help='drop out rate')
args = parser.parse_args()
print(args)

#shuffle sample list for get_batches function.
train_video_indices, validation_video_indices, test_video_indices = get_frame_indices(TRAIN_RGB_PATH,
                                                                                      VALIDATION_RGB_PATH,
                                                                                      TEST_RGB_PATH)
def rename_in_checkpoint(var):
    return (var.op.name).split('/', 1)[1]
#only restore basic layers
def restore_variables():
    if args.network == 'vgg_16':
        flow_variables_to_restore = slim.get_variables_to_restore(exclude=['rgb_model', 'flow_model/vgg_16/conv1/conv1_1', 'flow_model/vgg_16/fc8'])
        rgb_variables_to_restore = slim.get_variables_to_restore(exclude=['flow_model', 'rgb_model/vgg_16/fc8'])
    elif args.network == 'resnet_v1_50':
        flow_variables_to_restore = slim.get_variables_to_restore(exclude=['rgb_model', 'flow_model/resnet_v1_50/conv1', 'flow_model/resnet_v1_50/logits'])
        rgb_variables_to_restore = slim.get_variables_to_restore(exclude=['flow_model', 'rgb_model/resnet_v1_50/logits'])

    rgb_variables_to_restore = {rename_in_checkpoint(var): var for var in rgb_variables_to_restore}
    flow_variables_to_restore = {rename_in_checkpoint(var): var for var in flow_variables_to_restore}

    return rgb_variables_to_restore, flow_variables_to_restore


def train_fusion():
    with tf.Graph().as_default() as g:
        rgb_image = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_RGB_CHANNEL], name='rgb_image')
        flow_image = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_FLOW_CHANNEL], name='flow_image')
        label = tf.placeholder(tf.int32, [None, args.class_number], name = 'label')
        is_training = tf.placeholder(tf.bool)
        rgb_logits, flow_logits = two_stream_model(rgb_image,
                                                   flow_image,
                                                   args.network,
                                                   args.class_number,
                                                   args.keep_prob,
                                                   args.batch_size,
                                                   FRAMES_PER_VIDEO,
                                                   is_training)
        rgb_variables, flow_variables = restore_variables()

        rgb_restorer = tf.train.Saver(rgb_variables)
        flow_restorer = tf.train.Saver(flow_variables)
        #vgg_16 first layer fo flow model
        if args.network == 'vgg_16':
            fiw_variables = slim.get_variables_to_restore(include=['flow_model/vgg_16/conv1/conv1_1'])
        elif args.network == 'resnet_v1_50':
            fiw_variables = slim.get_variables_to_restore(include=['flow_model/resnet_v1_50/conv1/weights'])
        flow_input_weights_restorer = tf.train.Saver(fiw_variables)
        #Loss
        with tf.name_scope('loss'):
            rgb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rgb_logits, labels=label))
            tf.summary.scalar('rgb_loss', rgb_loss)
            flow_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flow_logits, labels=label))
            tf.summary.scalar('flow_loss', flow_loss)
        #Accuracy
        with tf.name_scope('accuracy'):
            rgb_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(rgb_logits, 1), tf.argmax(label, 1)), tf.float32))
            tf.summary.scalar('rgb_accuracy', rgb_accuracy)
            flow_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(flow_logits, 1), tf.argmax(label, 1)), tf.float32))
            tf.summary.scalar('flow_accuracy', flow_accuracy)
            two_stream_logits = rgb_logits + flow_logits
            two_stream_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(two_stream_logits, 1), tf.argmax(label, 1)), tf.float32))
            tf.summary.scalar('two_stream_accuracy', two_stream_accuracy)

        #variables for training, we only train weights after fusion
        #variables_for_training = slim.get_variables_to_restore(include=['flow_model', 'rgb_model'])#include=['rgb_model/vgg_16/fc8'])
        two_stream_loss = flow_loss + rgb_loss
        opt = tf.train.AdamOptimizer(args.lr)
	optimizer = slim.learning.create_train_op(two_stream_loss, opt)#, variables_to_train = variables_for_training)
        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=g) as sess:
            summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            g.finalize()
            #restore weights
            if args.network == 'vgg_16':
                rgb_restorer.restore(sess, VGG_16_MODEL_DIR)
                flow_restorer.restore(sess, VGG_16_MODEL_DIR)
                flow_input_weights_restorer.restore(sess, FLOW_INPUT_WEIGHTS_VGG_16)
            elif args.network == 'resnet_v1_50':
                rgb_restorer.restore(sess, RES_v1_50_MODEL_DIR)
                flow_restorer.restore(sess, RES_v1_50_MODEL_DIR)
                flow_input_weights_restorer.restore(sess, FLOW_INPUT_WEIGHTS_RES_v1_50)

            step = 0
            best_acc = 0.0
            best_ls = 10000.0
            best_val_acc = 0.0
            best_val_ls = 10000.0
            for epoch in range(args.epoches):
                acc_epoch = 0
                rgb_acc_epoch = 0
                flow_acc_epoch = 0
                ls_epoch = 0
                rgb_ls_epoch = 0
                flow_ls_epoch = 0
                batch_index = 0
                for i in range(len(train_video_indices) // args.batch_size):
                    step += 1
                    start_time = time.time()
                    #get batch data for training
                    rgb_batch_data, flow_batch_data, batch_index = get_batches(TRAIN_RGB_PATH,
                                                                               TRAIN_FLOW_PATH,
                                                                               args.batch_size,
                                                                               train_video_indices,
                                                                               batch_index)

                    _, ls, rgb_ls, flow_ls, rgb_acc, flow_acc, acc, summary = sess.run([optimizer,
                                                                                    two_stream_loss,
                                                                                    rgb_loss,
                                                                                    flow_loss,
                                                                                    rgb_accuracy,
                                                                                    flow_accuracy,
                                                                                    two_stream_accuracy,
                                                                                    summary_op],
                                                                                    feed_dict={rgb_image: rgb_batch_data['images'],
                                                                                               flow_image: flow_batch_data['images'],
                                                                                               label: rgb_batch_data['labels'],
                                                                                               is_training: True})
                    ls_epoch += ls
                    rgb_ls_epoch += rgb_ls
                    flow_ls_epoch += flow_ls
                    acc_epoch += acc
                    rgb_acc_epoch += rgb_acc
                    flow_acc_epoch += flow_acc
                    #every 10 step to show the step restult and write summary.
                    if i % 10 == 0:
                        end_time = time.time()
                        print('runing time {} :'.format(end_time - start_time))
                        print('Epoch {}, step {}, rgb loss {}, flow loss {}, rgb acc {}, flow acc {}, loss {}, acc {}'.format(epoch + 1,
                                                                                                                     step,
                                                                                                                     rgb_ls,
                                                                                                                     flow_ls,
                                                                                                                     rgb_acc,
                                                                                                                     flow_acc,
                                                                                                                     ls,
                                                                                                                     acc))
                        summary_writer.add_summary(summary, step)

                num = len(train_video_indices) // args.batch_size
                if best_acc < acc_epoch / num:
                    best_acc = acc_epoch / num
                if best_ls > ls_epoch / num:
                    best_ls = ls_epoch / num
                print('=========\n Epoch {}, best acc {}, best ls {}, rgb loss {}, flow loss {}, rgb acc {}, flow acc {}, loss {}, acc {}======'.format(epoch + 1,
                                                                                                                               best_acc,
                                                                                                                               best_ls,
                                                                                                                               rgb_ls_epoch / num,
                                                                                                                               flow_ls_epoch / num,
                                                                                                                               rgb_acc_epoch / num,
                                                                                                                               flow_acc_epoch / num,
                                                                                                                               ls_epoch / num,
                                                                                                                               acc_epoch / num))
                #validation
                ls_epoch = 0
                acc_epoch = 0
		rgb_acc_epoch = 0
		flow_acc_epoch = 0
		best_epoch = 0
		best_rgb_acc = 0
		best_flow_acc = 0
                batch_index = 0
                v_step = 0
                for i in range(len(validation_video_indices) // args.batch_size):
                    v_step += 1
                    rgb_batch_data, flow_batch_data, batch_index = get_batches(VALIDATION_RGB_PATH,
                                                                               VALIDATION_FLOW_PATH,
                                                                               args.batch_size,
                                                                               validation_video_indices,
                                                                               batch_index)
                    ls, acc, rgb_acc, flow_acc = sess.run([two_stream_loss, two_stream_accuracy, rgb_accuracy, flow_accuracy], feed_dict={rgb_image: rgb_batch_data['images'],
                                                                                          flow_image: flow_batch_data['images'],
                                                                                          label: rgb_batch_data['labels'],
                                                                                          is_training: False})
                    ls_epoch += ls
                    acc_epoch += acc
		    rgb_acc_epoch += rgb_acc
		    flow_acc_epoch += flow_acc

                if best_val_acc < acc_epoch / v_step:
                    best_val_acc = acc_epoch / v_step
		    best_epoch = epoch
		    saver.save(sess, TRAIN_CHECK_POINT + 'best_trained.ckpt')
                if best_val_ls > ls_epoch / v_step:
                    best_val_ls = ls_epoch / v_step
		if best_rgb_acc < rgb_acc_epoch / v_step:
                    best_rgb_acc = rgb_acc_epoch / v_step
		if best_flow_acc < flow_acc_epoch / v_step:
                    best_flow_acc = flow_acc_epoch / v_step

                print('Validation best acc {}, best ls {}, best rgb acc {}, best flow acc {}, loss {}, acc {}, rgb_acc {}, flow_acc {}'.format(best_val_acc, best_val_ls, best_rgb_acc, best_flow_acc, ls_epoch / v_step, acc_epoch / v_step, rgb_acc_epoch / v_step, flow_acc_epoch / v_step))

if __name__ == "__main__":
    train_fusion()
