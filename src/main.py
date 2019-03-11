from utils import read_bag, euler_angle_to_vector, Ray, calculate_poi
from face_detect import face_detect
from gazenet import Gazenet

import argparse
import scipy.misc
import matplotlib
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--filepath",
    type=str,
    required=True,
    help="Path to the bag file")
parser.add_argument(
    "-d",
    "--debug",
    type=int,
    default=0,
    help="Enable debug mode")
args = parser.parse_args()

GAZENET_PATH = '../models/gazenet/hopenet_robust_alpha1.pkl'


# returns color_crop, depth_crop
def crop(face, color, depth):
    assert color.shape[:2] == depth.shape, 'color shape: {} depth shape: {}'.format(
        color.shape, depth.shape)
    # depth_face should be identical to face as long as the shapes are the same
    scale_h, scale_w = color.shape[0] / \
        depth.shape[0], color.shape[1] / depth.shape[1]
    depth_face = face[0] / scale_h, face[1] / \
        scale_h, face[2] / scale_w, face[3] / scale_w

    def crop_with_bbox(image, x, y, w, h):
        from_y, to_y = int(round(y)), int(round(y + h))
        from_x, to_x = int(round(x)), int(round(x + w))
        return image[from_y:to_y, from_x:to_x]

    return crop_with_bbox(color, *face), crop_with_bbox(depth, *depth_face)


def show_images(images):
    fig = plt.figure()
    for i, image in enumerate(images):
        fig.add_subplot(221 + i)
        plt.imshow(image)
    plt.show()

# TODO
shelf_plane_normal = np.array([1000, 1000, 1000])

gazenet = Gazenet(GAZENET_PATH)

num_face_not_detected = 0
for i, (color, depth) in enumerate(read_bag(args.filepath)):
    face = face_detect(color)
    if face is None:
        num_face_not_detected += 1
        continue
    color_crop, depth_crop = crop(face, color, depth)
    if args.debug:
        show_images([color, color_crop, depth, depth_crop])
    angles = gazenet.image_to_euler_angles(
        color, (face[0], face[1], face[0] + face[2], face[1] + face[3]))

    depth = depth_crop[(int(depth_crop.shape[0]/2), int(depth_crop.shape[1]/2))]
    face_origin_2d = (np.mean([face[0], face[2]]), np.mean([face[1], face[3]]))
    gaze_origin = np.array([
            face_origin_2d[0],
            face_origin_2d[1],
            depth])
    gaze_direction = euler_angle_to_vector(
            yaw_angle=angles[0],
            pitch_angle=angles[1])
    gaze_ray = Ray(origin=gaze_origin, direction=gaze_direction)
    poi = calculate_poi(
            n_vector=shelf_plane_normal,
            ray=gaze_ray,
            dist_estimate=depth)

print('Total frames:', i + 1)
print('Frames without face detection:', num_face_not_detected)

