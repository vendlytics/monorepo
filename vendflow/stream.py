# Provides a way to orchestrate the
# flow of vendnet streaming output to
# a vendgaze consumer

import sys
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Face and gaze detection in streaming fashion')
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        help='GPU device id to use [0]',
        default=0,
        type=int)
    parser.add_argument(
        '--face_snapshot',
        dest='face_snapshot',
        help='Path of model snapshot.',
        default='',
        type=str)
    parser.add_argument(
        '--video',
        dest='video_path',
        help='Path of original video')
    parser.add_argument(
        '--bboxes',
        dest='bboxes',
        help='Bounding box annotations of frames')
    parser.add_argument(
        '--output_string',
        dest='output_string',
        help='String appended to output file')
    parser.add_argument(
        '--n_frames',
        dest='n_frames',
        help='Number of frames',
        type=int)
    parser.add_argument(
        '--fps',
        dest='fps',
        help='Frames per second of source video',
        type=float,
        default=30.)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()


# start the vendnet process


# start producing the output from stream to topic


# start the vendgaze process

# start consuming the output from topic to vendgaze
# to inform where bounding boxes are and write to open file
