import argparse
import os
import time

from stages.vendnet import ExtractFaces
from stages.gaze import ExtractGaze
from stages.attention import WriteAttention


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pipeline for end-to-end processing of video')
    parser.add_argument('--video_path', dest='video_path', type=str)
    parser.add_argument(
        '--vendnet_model_path',
        dest='vendnet_model_path',
        type=str)
    parser.add_argument('--gaze_model_path', dest='gaze_model_path', type=str)
    parser.add_argument(
        '--calibration_json_path',
        dest='calibration_json_path',
        type=str)
    parser.add_argument(
        '--output_dir',
        dest='output_path',
        default='data/jobs',
        type=str)

    args = parser.parse_args()
    return args


def job_id(camera_id='0'):
    """Job ID based on Unix timestamp of when job was run and source camera ID.
    """
    def timestamp_ms(): return int(round(time.time() * 1000))
    return '_'.join([str(timestamp_ms), str(camera_id)])


def check_job_dirs(output_dir, job_id):
    """Check that directories exist for outputs of pipeline, else make it.
    """
    job_directory = os.path.join([output_dir, job_id])
    if not os.path.exists(job_directory):
        os.makedirs(job_directory)

    return True


if __name__ == "__main__":
    args = parse_args()

    cur_job_id = job_id('0')

    pipeline = [
        ExtractFaces(cur_job_id, args.video_path, args.vendnet_model_path),
        ExtractGaze(cur_job_id, args.gaze_model_path),
        WriteAttention(cur_job_id, args.calibration_json)
    ]

    if check_job_dirs(args.output_dir, cur_job_id):
        for stage in pipeline:
            stage.run()
