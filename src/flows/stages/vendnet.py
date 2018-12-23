import _init_paths
import os
import sys

import cv2
import numpy as np
import torch

from model.nms.nms_wrapper import nms
from model.faster_rcnn import resnet

class ExtractFaces:

    def __init__(self, job_id, video_path, model_path):
        self.job_id = job_id
        self.video_path = video_path
        self.model_path = model_path

    def run(self):
        targets = np.asarray(['__background__', 'face'])
        
        faster_rcnn = resnet(
            classes=targets, num_layers=101, pretrained=False, class_agnostic=None
        )

        faster_rcnn._init_modules()
        faster_rcnn._init_weights()
        
        checkpoint = torch.load(self.model_path)
        faster_rcnn.load_state_dict(checkpoint['model'])
        
        # run for every frame
        

        # write outputs to deterministic job ID'd path


if __name__ == "__main__":
    ExtractFaces(
        'data/videos/short-kavanaugh.mp4',
        'models/vendnet/faster_rcnn_1_18_25759.pth',
        'data/jobs'
    ).run()