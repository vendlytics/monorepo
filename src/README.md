# `vendlytics/src`

## Structure

1. `vendnet` - for face detection of multiple faces in the frame and their locations in the frame. General object detection network using Faster R-CNN.
2. `gaze` - for the extraction of the Euler angles of each face in the frame.
3. `facenet` - for the tokenization of each face in the frame into a unique embedding that can be used to identify the same person in multiple frames and videos.

## Getting Started

### Start virtualenv

```shell
```

### Install requirements

```shell
pip install requirements.txt
```

### Download models

```shell
# git lfs download specific models
```

### Download sample data

```shell
# git lfs download specific sample data
```

### Get bounding boxes of faces

```sh
mkdir vendnet/lib/datasets/wider_face

# Need to put video into `vendnet/lib/datasets/wider_face`

python vendnet/demo.py --video_path <video_path> --cuda --dataset wider_face --load_dir output --checkpoint 25759 --checkepoch 18
```

### Draw gaze

```sh
python gaze/code/test_on_video_dockerface.py --snapshot PATH_OF_SNAPSHOT --video PATH_OF_VIDEO --bboxes FACE_BOUNDING_BOX_ANNOTATIONS --output_string STRING_TO_APPEND_TO_OUTPUT --fps 23.98
```

## Intel Realsense

TODO