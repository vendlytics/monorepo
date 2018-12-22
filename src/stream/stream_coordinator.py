# Provides a way to orchestrate the
# flow of vendnet streaming output to
# a gaze consumer

import sys
import os
import argparse
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser(
        description='Face and gaze detection stream pipeline')
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        help='GPU device id to use [0]',
        default=0,
        type=int)
    parser.add_argument(
        '--face_snapshot',
        dest='face_snapshot',
        help='Path of face detection model snapshot.',
        default='',
        type=str)
    parser.add_argument(
        '--gaze_snapshot',
        dest='gaze_snapshot',
        help='Path of gaze detection model snapshot.',
        default='',
        type=str)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help='Path of original video')
    parser.add_argument(
        '--output_string',
        dest='output_string',
        help='String appended to output file')
    parser.add_argument(
        '--fps',
        dest='fps',
        help='Frames per second of source video',
        type=float,
        default=30.)

    args = parser.parse_args()
    return args

def run_process(process):
    os.system('python {}'.format(process))

if __name__ == '__main__':
    args = parse_args()

    # start the vendnet process
    pool = Pool(processes=2)

    # start producing the output from stream to topic
    pool.map(run_process, 'vendnet/stream_worker.py')

    # start the gaze process
    pool.map(run_process, 'gaze/stream_worker.py')
