# Vendnet #

**Vendnet** adds several modifications to [Hopenet](https://github.com/natanielruiz/deep-head-pose), where instead of using bounding boxes output by [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf), we use human pose estimation to extract the joints and use them as references on which we can locate the entire head.

## Getting Started

#### Requirements

* [Pytorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/).
* NVIDIA 1080 Ti
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) v9+
* [CuDNN](https://developer.nvidia.com/cudnn) v7+
* [Docker](https://docs.docker.com/engine/installation/)


```bash
git clone git@github.com:itselijahtai/vendnet.git
cd vendnet
bin/setup # install `nvidia-docker` + creates data and output directories
```

#### Video Data

The `bin/setup` script should have created a `data` folder in the project directory. Add the video you'd like to run the pipeline on into that folder.

The next section will transfer that video over to the Docker image to extract face bounding boxes, and automatically add that output to the local `output/video` folder.

#### Face Detection

The initial part of the pipeline builds a bounding box around the faces that are in the video. The video that is being used to test should be placed inside the `data/` folder.

The bounding box around the faces happens inside a Docker image which comes with all of the dependencies pre-installed and configured. Make sure you're in the project directory.

```bash
sudo nvidia-docker run -it -v $PWD/data:/opt/py-faster-rcnn/edata -v $PWD/output/video:/opt/py-faster-rcnn/output/video -v $PWD/output/images:/opt/py-faster-rcnn/output/images natanielruiz/dockerface:latest
```

Recompile Caffe
``` bash
cd caffe-fast-rcnn
rm -rf build
mkdir build
cd build
cmake -DUSE_CUDNN=1 ..
make -j20
cd ../..
```

Run this to process the video
```bash
python tools/run_face_detection_on_video.py --gpu 0 --video edata/YOUR_VIDEO_FILENAME --output_string STRING_TO_BE_APPENDED_TO_OUTPUTFILE_NAME --conf_thresh CONFIDENCE_THRESHOLD_FOR_DETECTIONS
```

For the confidence threshold, use `0.85`. The output bounding boxes have this format:
```
frame_number x_min y_min x_max y_max confidence_score
```

Once you're done with the Docker container, you can exit with `exit`.

To make sure you don't have to compile the docker container again, get the `CONTAINER_ID` with
```bash
sudo docker ps -a
```

Substitute the `CONTAINER_ID` to start and attach to that container.
```bash
sudo docker start CONTAINER_ID
sudo docker attach CONTAINER_ID
```

Terminal may seem to be hanging - just press enter (or any key).

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

