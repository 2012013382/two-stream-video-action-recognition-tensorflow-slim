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
TRAIN_LOG_DIR = os.path.join('../log/train/rgb/', time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
TRAIN_CHECK_POINT = '../check_point/'
TRAIN_RGB_PATH = '../cache_data/train_rgb.txt'
VALIDATION_RGB_PATH = '../cache_data/validation_rgb.txt'
TEST_RGB_PATH = '../cache_data/test_rgb.txt'
VGG_16_MODEL_DIR = '../check_point/vgg_16.ckpt'
RES_v1_50_MODEL_DIR = '../check_point/resnet_v1_50.ckpt'

parser = argparse.ArgumentParser(description='UCF101 two stream fusion')
parser.add_argument('--network', default='resnet_v1_50', type=str, help='network name')
parser.add_argument('--epoches', default=500, type=int, help='number of total epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='start learning rate')
parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--class_number', default=101, type=int, help='class number of UCF-101')
parser.add_argument('--keep_prob', default=0.5, type=float, help='drop out rate for vgg.')
args = parser.parse_args()
print(args)

train_video_indices, validation_video_indices, test_video_indices = get_frame_indices(TRAIN_RGB_PATH,
                                                                                      VALIDATION_RGB_PATH,
                                                                                      TEST_RGB_PATH)

def rename_in_checkpoint(var):
    return (var.op.name).split('/', 1)[1]

def restore_variables():
    if args.network == 'vgg_16':
        rgb_variables_to_restore = slim.get_variables_to_restore(exclude=['rgb_model/vgg_16/fc8'])
    elif args.network == 'resnet_v1_50':
        rgb_variables_to_restore = slim.get_variables_to_restore(exclude=['rgb_model/resnet_v1_50/logits'])
    
    rgb_variables_to_restore = {rename_in_checkpoint(var): var for var in rgb_variables_to_restore}

    return rgb_variables_to_restore

    
def train_rgb():
    with tf.Graph().as_default() as g:
        rgb_image = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_RGB_CHANNEL], name='rgb_image')
        label = tf.placeholder(tf.int32, [None, args.class_number], name = 'label')
        is_training = tf.placeholder(tf.bool)
        rgb_logits = two_stream_model(rgb_image, 
                                  'None', 
                                  args.network, 
                                  args.class_number, 
                                  args.keep_prob,
                                  args.batch_size,
                                  FRAMES_PER_VIDEO,
                                  is_training,
                                  'rgb')

        rgb_variables = restore_variables()
        rgb_restorer = tf.train.Saver(rgb_variables)

        #Loss
        with tf.name_scope('loss'):
            rgb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rgb_logits, labels=label))
            tf.summary.scalar('rgb_loss', rgb_loss)
        #Accuracy
        with tf.name_scope('accuracy'):
            rgb_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(rgb_logits, 1), tf.argmax(label, 1)), tf.float32))
            tf.summary.scalar('rgb_accuracy', rgb_accuracy)

        opt = tf.train.AdamOptimizer(args.lr)
	optimizer = slim.learning.create_train_op(rgb_loss, opt)
        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=g) as sess:
            summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            g.finalize()
            if args.network == 'vgg_16':
                rgb_restorer.restore(sess, VGG_16_MODEL_DIR)
            elif args.network == 'resnet_v1_50':
                rgb_restorer.restore(sess, RES_v1_50_MODEL_DIR)

            step = 0
            best_acc = 0.0
            best_ls = 10000.0
            best_val_acc = 0.0
            best_val_ls = 10000.0
	    best_epoch = 0
            for epoch in range(args.epoches):
                acc_epoch = 0
                ls_epoch = 0
                batch_index = 0
                for i in range(len(train_video_indices) // args.batch_size):
                    step += 1
                    start_time = time.time()
                    rgb_batch_data, batch_index = get_batches(TRAIN_RGB_PATH, 
                                                              'None', 
                                                              args.batch_size, 
                                                              train_video_indices, 
                                                              batch_index,
                                                              'rgb')
                    
                    _, ls, acc, summary = sess.run([optimizer, rgb_loss, rgb_accuracy, summary_op],
                                                                                    feed_dict={rgb_image: rgb_batch_data['images'],
                                                                                               label: rgb_batch_data['labels'],
                                                                                               is_training: True})
                    ls_epoch += ls
                    acc_epoch += acc

                    if i % 10 == 0:
                        end_time = time.time()
                        print('runing time {} :'.format(end_time - start_time))
                        print('Epoch {}, step {}, loss {}, acc {}'.format(epoch + 1, step, ls, acc))
                        summary_writer.add_summary(summary, step)

                num = len(train_video_indices) // args.batch_size
                if best_acc < acc_epoch / num:
                    best_acc = acc_epoch / num
                if best_ls > ls_epoch / num:
                    best_ls = ls_epoch / num
                print('=========\n Epoch {}, best acc {}, best ls {}, loss {}, acc {}======'.format(epoch + 1, best_acc, best_ls, ls_epoch / num, acc_epoch / num))
                
                #validation
                ls_epoch = 0
                acc_epoch = 0
                batch_index = 0
                v_step = 0
                for i in range(len(validation_video_indices) // args.batch_size):
                    v_step += 1
                    rgb_batch_data, batch_index = get_batches(VALIDATION_RGB_PATH,
                                                              'None',
                                                              args.batch_size,
                                                              validation_video_indices,
                                                              batch_index,
                                                              'rgb')
                    ls, acc = sess.run([rgb_loss, rgb_accuracy], feed_dict={rgb_image: rgb_batch_data['images'], 
                                                                            label: rgb_batch_data['labels'], 
                                                                            is_training: False})
                    ls_epoch += ls
                    acc_epoch += acc
                
                if best_val_acc < acc_epoch / v_step:
                    best_val_acc = acc_epoch / v_step
		    best_epoch = epoch
		    saver.save(sess, TRAIN_CHECK_POINT + 'rgb_trained.ckpt')
                if best_val_ls > ls_epoch / v_step:
                    best_val_ls = ls_epoch / v_step

                print('Validation best epoch {}, best acc {}, best ls {}, loss {}, acc {}'.format(best_epoch + 1, best_val_acc, best_val_ls,  ls_epoch / v_step, acc_epoch / v_step))
                
if __name__ == "__main__":
    train_rgb()
