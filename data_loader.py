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

    img = np.array(cv2.resize(np.array(img), (IMG_HEIGHT, IMG_WIDTH))).astype(np.float32)
    img = img / 255

    if mode == 'rgb':#substract mean and std
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    return img


#Generate batches for the network
def get_batches(rgb_file_name, flow_file_name, batch_size, video_indices, batch_index):
    #obtain rgb images per batch. FRAMES_PER_VIDEO rgb images per video
    rgb_f = open(rgb_file_name, 'r')
    rgb_images = []
    rgb_labels = []
    images_labels = list(rgb_f)
    for i in video_indices[batch_index: batch_index + batch_size]:#batch size
        image_label = images_labels[i].split()#5 frames and 1 label
        for j in range(FRAMES_PER_VIDEO):
            image_name = image_label[j]
            img = read_image(image_name, 'rgb')
            rgb_images.append(img)
	        temp_label = np.zeros((1, 101)).astype(np.int32)
            if (int(image_label[FRAMES_PER_VIDEO])) < 0 or (int(image_label[FRAMES_PER_VIDEO])) > 100:
                print('bad label')
            temp_label[0, int(image_label[FRAMES_PER_VIDEO])] = 1
            rgb_labels.append(temp_label)

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
        label = flow_image_label[2 * FRAMES_PER_VIDEO * FLOW_LENGTH]
        flow_labels.append(label)
    #Convert to numpy
    rgb_images = np.array(rgb_images).astype(np.float32)#shape[BATCH_SIZE * FRAMES_PER_VIDEO, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL]
    rgb_labels = np.reshape(np.array(rgb_labels).astype(np.int32), [-1, 101])#shape[BATCH_SIZE, 101]
    flow_images = np.array(flow_images).astype(np.float32)#shape[BATCH_SIZE * FRAMES_PER_VIDEO, IMG_HEIGHT, IMG_WIDTH, IMG_FLOW_CHANNEL]
    flow_labels = np.reshape(np.array(flow_labels).astype(np.int32), [-1, 1])#shape[BATCH_SIZE, 1]
    batch_index = batch_index + batch_size
    rgb_batch_data = {'images': rgb_images, 'labels': rgb_labels}
    flow_batch_data = {'images': flow_images, 'labels': flow_labels}

    rgb_f.close()
    flow_f.close()
    return rgb_batch_data, flow_batch_data, batch_index
