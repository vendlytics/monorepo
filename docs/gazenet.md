# `gazenet`

**gazenet** adds several modifications to [Hopenet](https://github.com/natanielruiz/deep-head-pose), where instead of using bounding boxes output by [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf), we use human pose estimation to extract the joints and use them as references on which we can locate the entire head.

## Getting Started

### Requirements

* [Pytorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/).
* NVIDIA GPU
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) v9+
* [CuDNN](https://developer.nvidia.com/cudnn) v7+
* [Docker](https://docs.docker.com/engine/installation/)

#### Face Detection

The initial part of the pipeline builds a bounding box around the faces that are in the video. The video that is being used to test should be placed inside the `data/` folder.

The output bounding boxes have this format:

```sh
frame_number x_min y_min x_max y_max confidence_score
```

#### Gaze

The second part of the pipeline outputs a file which contains the [Euler angles](https://en.wikipedia.org/wiki/Euler_angles) that describe the pitch, yaw and roll of the faces that have been detected by [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf).

## Pretrained Models

In `models`, stored on Git LFS.