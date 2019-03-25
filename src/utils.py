import logging
import numpy as np
import pyrealsense2 as rs
import json
from typing import NamedTuple

from scipy.spatial.distance import euclidean

# Logger
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Ray(NamedTuple):
    origin: np.array # Shape: (3,)
    direction: np.array # Shape: (3,)

# Representation of the calibration script's output
class Calibration(NamedTuple):
    shelf_plane_normal: np.array # Shape: (3,)
    products: np.array # Shape: (num_products, 3)

# yields (color np.array, depth np.array)
def read_bag(filepath, color_config, depth_config):
    # create context
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, filepath, repeat_playback=False)
    config.enable_stream(rs.stream.color, color_config['width'], color_config['height'], rs.format.rgb8, color_config['fps'])
    config.enable_stream(rs.stream.depth, depth_config['width'], depth_config['height'], rs.format.z16, depth_config['fps'])

    align = rs.align(rs.stream.color)
    pipeline_profile = pipeline.start(config)

    # If we don't set this, reading the frames takes as long as the recording
    # duration
    device = pipeline_profile.get_device()
    device.as_playback().set_real_time(False)

    while True:
        success, frames = pipeline.try_wait_for_frames()
        if not success:
            return
        frames = align.process(frames)
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        yield np.array(color_frame.get_data()), np.array(depth_frame.get_data())


def euler_angle_to_vector(yaw_angle, pitch_angle):
    """
    Converts from angles to directional vector.
    """
    yaw_angle_radians = yaw_angle * np.pi / 180
    pitch_angle_radians = pitch_angle * np.pi / 180

    return np.array([
        np.cos(yaw_angle_radians) * np.cos(pitch_angle_radians),
        np.sin(yaw_angle_radians) * np.cos(pitch_angle_radians),
        np.sin(pitch_angle_radians)
    ])


def calculate_poi(n_vector, ray, dist_estimate):
    """
    Finds the point of intersection of a vector on the shelf plane.

    n_vector is shelf plane vector i.e. normal to the shelf plane
    ray is the combination of origin and direction to intersect with the plane
    dist_estimate is estimate of shelf distance from person

    Notation from http://www.ambrsoft.com/TrigoCalc/Plan3D/PlaneLineIntersection_.htm
    """
    p_vector = n_vector + dist_estimate
    t = -(np.dot(n_vector, p_vector) + dist_estimate) / \
        float(np.dot(n_vector, ray.direction))
    return ray.origin + ray.direction * t

def read_calibration(calibration_filepath):
    with open(calibration_filepath) as f:
        calibration = json.load(f)
    shelf_plane_normal = euler_angle_to_vector(
            yaw_angle=calibration['shelf']['yaw'],
            pitch_angle=calibration['shelf']['pitch'])
    products = []
    for product in calibration['products']:
        products.append([product['poi_x'], product['poi_y'], product['poi_z']])
    products = np.array(products)
    return Calibration(shelf_plane_normal=shelf_plane_normal, products=products)

def index_nearest(vector, matrix):
    return np.linalg.norm(matrix - vector, axis=1).argmin()

