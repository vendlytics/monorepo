from src.utils import read_bag
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

print('Total frames:', i + 1)
print('Frames without face detection:', num_face_not_detected)
