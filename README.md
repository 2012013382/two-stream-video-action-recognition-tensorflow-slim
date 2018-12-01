# two-stream-video-action-recognition-tensorflow-slim
Basic two stream video action classification by tensorflow slim.
## Requirement
Memory: 4GB (at least)

GPU: 12GB (TITAN X in my experiments)

Hard disk: 25GB (at least)
## Some preparation
### Data set
UCF-101 http://crcv.ucf.edu/data/UCF101.php
### Flow frames
I use gpu_flow from feichtenhfer https://github.com/feichtenhofer/gpu_flow to obtain flow frames of UCF-101.
```
cd gpu_flow-master
mkdir -p build
cmake ..
make
mkdir -p UCF_101_flow
```
Then, place UCF-101 dataset into the 'build' folder.
```
sudo ./compute_flow --gpuID=0 --type=1
```
It needs several hours for running and about 14.GB space for saving.
### RGB frames
```
cd ..
cd ..
sudo ./convert_video_to_images.sh ./gpu_flow-master/build/UCF-101/
```
It needs several hours for running and about 11.GB space for saving.
## First run
### Prepare weights
I use pre-trained weights(Resnet-v1-50 and vgg_16) from slim. https://github.com/tensorflow/models/tree/master/research/slim

Then, place .ckpt files into the 'check_point' file.

Prepare conv2d weights of the first layer of the model.
```
cd utils
python prepare_weights_vgg_16.py
python prepare_weights_res_v1_50.py
```
### Some data preparation
```
cd ..
python data_process.py
```
## Train
```
python train.py
```
## Validaion
I extract 1/8 videos from train set(UCF-101 split1) adn the result is about 92%(Resnet_v1_50).
## Test
Waiting for...
