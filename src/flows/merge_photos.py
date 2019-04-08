"""
Merges photos from rosbag converter into video.

Usage:
python3.6 -m src.flows.merge_photos
"""

import os
import sys
import cv2

IMAGE_FOLDER = '/Users/wonjunetai/src/github.com/output'
PREFIX = "simple_calibration_Depth"


if 'Color' in PREFIX:
    FPS = 20.55
    VIDEO_NAME = 'color_video.avi'
else:
    FPS = 30
    VIDEO_NAME = 'depth_video.avi'


def calculate_color_fps(num_color: int, num_depth: int, root_fps: int = 30) -> float:
    """
    The rosbag converter for depth and color doesn't convert each frame properly
    For now, just a fix

    Arguments:
        num_color {int} -- Number of color frames
        num_depth {int} -- Number of depth frames

    Keyword Arguments:
        root_fps {int} -- The original fps at filming (default: {30})

    Returns:
        [float] -- The FPS that color processing should be set at
    """

    return num_color / num_depth * root_fps


def print_progress(current: int, total: int):
    """Prints the progress in terminal on same line.

    Arguments:
        current {int}
        total {int}
    """
    sys.stdout.write("Progress: %d/%d   \r" % (current, total))
    sys.stdout.flush()

def take_frame_num(x):
    return int(x.split('_')[3].split('.')[0])


if __name__ == '__main__':
    images = [img for img in os.listdir(IMAGE_FOLDER) if img.startswith(PREFIX)]
    images = sorted(images, key=take_frame_num)

    frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(VIDEO_NAME, 0, FPS, (width, height))

    num_images = len(images)
    for i, image in enumerate(images):
        image_file = image
        image = cv2.imread(os.path.join(IMAGE_FOLDER, image_file))
        image = cv2.putText(image, image_file, (int(width/2) - 100, int(height/2)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=2)
        video.write(image)
        print_progress(i, num_images)


    cv2.destroyAllWindows()
    video.release()