# Vendgaze #

**Vendgaze** adds several modifications to [Hopenet](https://github.com/natanielruiz/deep-head-pose), where instead of using bounding boxes output by [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf), we use human pose estimation to extract the joints and use them as references on which we can locate the entire head.

## Getting Started

#### Requirements

* [Pytorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/).
* NVIDIA 1080 Ti
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) v9+
* [CuDNN](https://developer.nvidia.com/cudnn) v7+
* [Docker](https://docs.docker.com/engine/installation/)


```bash
git clone git@github.com:itselijahtai/vendlytics.git
cd vendlytics
bin/setup # install `nvidia-docker` + creates data and output directories
```

#### Video Data

The `bin/setup` script should have created a `data` folder in the project directory. Add the video you'd like to run the pipeline on into that folder.

The next section will transfer that video over to the Docker image to extract face bounding boxes, and automatically add that output to the local `output/video` folder.

#### Face Detection

The initial part of the pipeline builds a bounding box around the faces that are in the video. The video that is being used to test should be placed inside the `data/` folder.

The output bounding boxes have this format:
```
frame_number x_min y_min x_max y_max confidence_score
```

#### Gaze

The second part of the pipeline outputs a file which contains the [Euler angles](https://en.wikipedia.org/wiki/Euler_angles) that describe the pitch, yaw and roll of the faces that have been detected by [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf).

Replace:
* `PATH_OF_SNAPSHOT` with the path to the model you'd like to use
* `PATH_OF_VIDEO` with the path to the original video in `data/`
* `FACE_BOUNDING_BOX_ANNOTATIONS` with the path to the output you got from Docker (in `output/video/`)
* `STRING_TO_APPEND_TO_OUTPUT` with whatever you'd like

```bash
python code/test_on_video_dockerface.py --snapshot PATH_OF_SNAPSHOT --video PATH_OF_VIDEO --bboxes FACE_BOUNDING_BOX_ANNOTATIONS --output_string STRING_TO_APPEND_TO_OUTPUT
```

Your output should be available in `output/video`.

##  Pretrained Models

[300W-LP, alpha 1 (MAE of 6.410)](https://drive.google.com/open?id=1EJPu2sOAwrfuamTitTkw2xJ2ipmMsmD3)

[300W-LP, alpha 2 (MAE of 6.155)](https://drive.google.com/open?id=16OZdRULgUpceMKZV6U9PNFiigfjezsCY)

[300W-LP, alpha 1, robust to image quality](https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR)

