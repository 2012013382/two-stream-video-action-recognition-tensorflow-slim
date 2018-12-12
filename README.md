# two-stream-video-action-recognition-tensorflow-slim
Basic two stream video action classification by tensorflow slim. I fuse two streams by adding logits together simply. I use pretrained models to extract features and no more operations.
## Requirement
Memory: 4GB (at least)

GPU: 12GB (TITAN X in my experiments)

Hard disk: 25GB (at least)
## Some preparation
### Data set
UCF-101 http://crcv.ucf.edu/data/UCF101.php

### tensorflow nets
```
git clone https://github.com/tensorflow/models/tree/master/research/slim/nets
```
### Flow frames
I use gpu_flow from feichtenhfer https://github.com/feichtenhofer/gpu_flow to obtain flow frames of UCF-101.
```
cd gpu_flow-master
mkdir -p build
cd build
cmake ..
make
```

If there are errors, apply

```
cd ..
rm -r build
mkdir -p build
cd build
cmake -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
make
```

Then

```
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
If you want to train two stream

```
python train_two_stream.py
```

If you only want to  train rgb stream

```
python train_rgb.py
```

If you only want to train flow stream

```
python train_flow.py
```

## Validaion
I extract around 1/8 videos from train set(UCF-101 split1) and results in the following chart are base on Resnet_v1_50.

|  Two Stream   |  rgb   |  flow   |
| ------------- |--------|---------|
|     0.813     |  0.743 |  0.646  |
## Test
The following results are based on Resnet_v1_50

|  Two Stream   |      rgb      |      flow     |
| ------------- | ------------- | ------------- |
|     0.780     |      0.71     |      0.541    |
## More 
More results are coming...
