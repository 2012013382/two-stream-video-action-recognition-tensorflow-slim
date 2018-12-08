from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL.Image as Image
import numpy as np
import os
import time
import cv2
import random
FLOW_LENGTH = 10
FRAMES_PER_VIDEO = 5
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_RGB_CHANNEL = 3
IMG_FLOW_CHANNEL = 2 * FLOW_LENGTH

def get_frame_indices(train_rgb_file, validation_rgb_file, test_rgb_file):
    tf = open(train_rgb_file, 'r')
    train_lines = list(tf)
    vf = open(validation_rgb_file, 'r')
    validation_lines = list(vf)
    tef = open(test_rgb_file, 'r')
    test_lines = list(tef)
    train_video_indices = range(len(train_lines))
    validation_video_indices = range(len(validation_lines))
    test_video_indices = range(len(test_lines))
    random.seed(time.time())
    random.shuffle(train_video_indices)
    random.shuffle(validation_video_indices)
    random.shuffle(test_video_indices)
    tf.close()
    vf.close()
    tef.close()
    return train_video_indices, validation_video_indices, test_video_indices

def read_image(file_name, mode):
    file_name = '../' + file_name
    if mode == 'flow':
        img = cv2.imread(file_name, 0)
    else:
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    rate = random.uniform(-0.25, 0.25)#Rescale +- 25%
    img = cv2.resize(img, (int(256 * (1 + rate)), int(256 * (1 + rate))))#Rescale HEIGHT, WIDTH
    img = np.array(cv2.resize(np.array(img), (IMG_HEIGHT, IMG_WIDTH))).astype(np.float32)
    img = img / 255

    if mode == 'rgb':#bgr
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    if mode == 'flow':
        img[:, :] = (img[:, :] - 0.5) / 0.226
    
    return img

def read_label(label):
    temp_label = np.zeros((FRAMES_PER_VIDEO, 101)).astype(np.int32)
    temp_label[:, label] = 1
    return temp_label

#Generate batches for the network
def get_batches(rgb_file_name, flow_file_name, batch_size, video_indices, batch_index, stream_type):
    if stream_type == 'rgb' or stream_type == 'two_stream':
        #obtain rgb images per batch. FRAMES_PER_VIDEO rgb images per video
        rgb_f = open(rgb_file_name, 'r')
        rgb_images = []
        rgb_labels = []
        images_labels = list(rgb_f)
        for i in video_indices[batch_index: batch_index + batch_size]:#batch size
            image_label = images_labels[i].split()#5 frames and 1 label
            rgb_image = [read_image(image_name, 'rgb') for image_name in image_label[:-1]]
            rgb_label = read_label(int(image_label[FRAMES_PER_VIDEO]))
            rgb_images.append(rgb_image)
            rgb_labels.append(rgb_label)
        rgb_images = np.reshape(np.array(rgb_images), [-1, IMG_HEIGHT, IMG_WIDTH, IMG_RGB_CHANNEL])#shape[BATCH_SIZE * FRAMES_PER_VIDEO, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL]
        rgb_labels = np.reshape(np.array(rgb_labels).astype(np.int32), [-1, 101])
        rgb_batch_data = {'images': rgb_images, 'labels': rgb_labels}
        rgb_f.close()
    
    if stream_type == 'flow' or stream_type == 'two_stream':
        #obtain x and y flow images per batch.
        flow_f = open(flow_file_name, 'r')
        flow_images = []
        flow_labels = []
        flow_images_labels = list(flow_f)
        for i in video_indices[batch_index: batch_index + batch_size]:#batch size
            flow_image_label = flow_images_labels[i].split()
            for j in range(FRAMES_PER_VIDEO):# 5
                x_flow_images = []
                y_flow_images = []
                for h in range(j * FLOW_LENGTH, (1 + j) * FLOW_LENGTH):
                    #x flow
                    x_image_name = flow_image_label[h]
                    x_img = read_image(x_image_name, 'flow')
                    x_flow_images.append(x_img)
                    #y flow
                    y_image_name = flow_image_label[h + FRAMES_PER_VIDEO * FLOW_LENGTH]
                    y_img = read_image(y_image_name, 'flow')
                    y_flow_images.append(y_img)
                x_flow_images = np.array(x_flow_images)#shape[FLOW_LENGTH, IMG_HEIGHT, IMG_WIDTH, 1]
                y_flow_images = np.array(y_flow_images)
                flow_image = np.squeeze(np.concatenate((x_flow_images, y_flow_images), axis = 0))#shape[IMG_FLOW_CHANNEL, IMG_HEIGHT, IMG_WIDTH]
                flow_image = np.transpose(flow_image, (1, 2, 0))#shape[IMG_HEIGHT, IMG_WIDTH, IMG_FLOW_CHANNEL]
                flow_images.append(flow_image)
            flow_label = read_label(int(flow_image_label[-1]))
            flow_labels.append(flow_label)
        #Convert to numpy
        flow_images = np.array(flow_images).astype(np.float32)#shape[BATCH_SIZE * FRAMES_PER_VIDEO, IMG_HEIGHT, IMG_WIDTH, IMG_FLOW_CHANNEL]
        flow_labels = np.reshape(np.array(flow_labels).astype(np.int32), [-1, 101])
        flow_batch_data = {'images': flow_images, 'labels': flow_labels}
        flow_f.close()
    
    batch_index = batch_index + batch_size
    if stream_type == 'two_stream':
        return rgb_batch_data, flow_batch_data, batch_index
    elif stream_type == 'rgb':
        return rgb_batch_data, batch_index
    elif stream_type == 'flow':
        return flow_batch_data, batch_index

