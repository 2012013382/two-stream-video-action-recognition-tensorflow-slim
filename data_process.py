from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#==========================================================================
#Write frames pathes into files, one line for one video. One video is divided
#into 5 rgb frames and 100 flow frames(50 x frames and 50 y frames).
#==========================================================================
import PIL.Image as Image
import numpy as np
import os
import time
import cv2
import re
from os.path import join
#Some pathes
FLOW_PATH = './gpu_flow-master/build/UCF_101_flow/flow'
FLOW_X_PATH = FLOW_PATH + '/' + 'x/'
FLOW_Y_PATH = FLOW_PATH + '/' + 'y/'
RGB_PATH = './gpu_flow-master/build/UCF-101/'
TRAIN_SPLIT = './UCF_list/trainlist01.txt'
TEST_SPLIT = './UCF_list/testlist01.txt'
CLASS_NAME_PATH = './UCF_list/classInd.txt'
TRAIN_FLOW_PATH = './cache_data/train_flow.txt'
TRAIN_RGB_PATH = './cache_data/train_rgb.txt'
VALIDATION_FLOW_PATH = './cache_data/validation_flow.txt'
VALIDATION_RGB_PATH = './cache_data/validation_rgb.txt'
TEST_FLOW_PATH = './cache_data/test_flow.txt'
TEST_RGB_PATH = './cache_data/test_rgb.txt'
FLOW_LENGTH = 10
FRAMES_PER_VIDEO = 5

def write_flow_file(root, video_path, file_names, f, label):#FLOW_LENGTH * FRAMES_PER_VIDEO x_flow frames and FLOW_LENGTH * FRAMES_PER_VIDEO y_flow frames.
    video_length = len(file_names)
    interval = video_length // FRAMES_PER_VIDEO
    sample_start = interval // 2
    for _ in range(FRAMES_PER_VIDEO):
        if sample_start - FLOW_LENGTH // 2 < 0:
            sample_start = FLOW_LENGTH // 2
        if sample_start + FLOW_LENGTH // 2 > video_length:
            sample_start = video_length - FLOW_LENGTH // 2
        for i in range(sample_start - FLOW_LENGTH // 2, sample_start + FLOW_LENGTH // 2):
            f.write(join(root, file_names[i]) + ' ')
        sample_start += interval
    sample_start = interval // 2
    for _ in range(FRAMES_PER_VIDEO):
        if sample_start - FLOW_LENGTH // 2 < 0:
            sample_start = FLOW_LENGTH // 2
        if sample_start + FLOW_LENGTH // 2 > video_length:
            sample_start = video_length - FLOW_LENGTH // 2
        for i in range(sample_start - FLOW_LENGTH // 2, sample_start + FLOW_LENGTH // 2):
            tpath = FLOW_Y_PATH + video_path[-2] + '/' + video_path[-1]
            f.write(join(tpath, file_names[i]) + ' ')
        sample_start += interval
    f.write(str(label) + '\n')

def write_rgb_file(root, video_path, file_names, f, label):
    video_length = len(file_names)
    interval = video_length // FRAMES_PER_VIDEO
    sample_start = interval // 2
    for _ in range(FRAMES_PER_VIDEO):
        f.write(join(root, file_names[sample_start]) + ' ')
        sample_start += interval
    f.write(str(label) + '\n')

#Split train, validation and test sets as image pathes.
def generate_split_files():
    train_sample_list = []
    validation_sample_list = []
    test_sample_list = []
    classes = {}
    with open(CLASS_NAME_PATH, 'r') as f:
        for line in f:
            try:
                row = [x for x in line.split()]
                classes[row[1]] = int(row[0]) - 1
            except Exception as e:
                print('No {} file!'.format(CLASS_NAME_PATH))

    with open(TRAIN_SPLIT, 'r') as f:
        count = 0
        for line in f:
            try:
                row = [x for x in re.split('[/.]', line)]#From 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi' get 'v_ApplyEyeMakeup_g01_c01'.
                if count % 8 == 0:
                    validation_sample_list.append(row[1])#Split validation set from train set.
                else:
                    train_sample_list.append(row[1])
                count += 1
            except Exception as e:
                print('No {} file!'.format(TRAIN_SPLIT))
                pass

    with open(TEST_SPLIT, 'r') as f:
        for line in f:
            try:
                row = [x for x in re.split('[/.]', line)]
                test_sample_list.append(row[1])
            except Exception as e:
                print('No {} file!'.format(TEST_SPLIT))
                pass
    #Prepare flow pathes
    with open(TRAIN_FLOW_PATH, 'w') as f_train:
        with open(VALIDATION_FLOW_PATH, 'w') as f_validation:
            with open(TEST_FLOW_PATH, 'w') as f_test:
                for root, dirs, file_names in os.walk(FLOW_X_PATH):
                    for tdir in dirs:
                        if tdir == '':
                            continue
                        class_name = tdir
                        label = classes[class_name]#obtain class name
                        for root2, _, file_names in os.walk(root + tdir):
                            if len(file_names) == 0:
                                continue
                            file_names = sorted(file_names)
                            video_path = root2.split('/')
                            if video_path[-1] in train_sample_list:#train file
                                write_flow_file(root2, video_path, file_names, f_train, label)
                            elif video_path[-1] in validation_sample_list:#validation file
                                write_flow_file(root2, video_path, file_names, f_validation, label)
                            elif video_path[-1] in test_sample_list:#test file
                                write_flow_file(root2, video_path, file_names, f_test, label)
                            else:
                                print('No such video in the list!')
                    break

    #Prepare rgb pathes
    with open(TRAIN_RGB_PATH, 'w') as f_train:
        with open(VALIDATION_RGB_PATH, 'w') as f_validation:
            with open(TEST_RGB_PATH, 'w') as f_test:
                for root, dirs, _ in os.walk(RGB_PATH):
                    for tdir in dirs:
                        if tdir == '':
                            continue
                        class_name = tdir
                        label = classes[class_name]
                        for root2, _, file_names in os.walk(root + tdir):
                            if len(file_names) == 0:
                                continue
                            file_names = sorted(file_names)
                            video_path = root2.split('/')
                            if video_path[-1] in train_sample_list:#train file
                                write_rgb_file(root2, video_path, file_names, f_train, label)
                            elif video_path[-1] in validation_sample_list:#validation file
                                write_rgb_file(root2, video_path, file_names, f_validation, label)
                            elif video_path[-1] in test_sample_list:#test file
                                write_rgb_file(root2, video_path, file_names, f_test, label)
                            else:
                                print('No such video in the list!')
                    break

if __name__ == "__main__":
    generate_split_files()
