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


def calculate_poi(n_vector, ray, gaze_depth):
    """
    Finds the point of intersection of a vector on the shelf plane.

    n_vector is shelf plane vector i.e. normal to the shelf plane
    ray is the combination of origin and direction to intersect with the plane

    Notation from http://www.ambrsoft.com/TrigoCalc/Plan3D/PlaneLineIntersection_.htm
    """
    p_vector = n_vector + np.array([0, 0, gaze_depth])
    numerator = float(np.dot(n_vector, p_vector - ray.origin))
    denominator = float(np.dot(n_vector, ray.direction - ray.origin))
    print(numerator, denominator)
    t = -(numerator) / denominator
    return ray.origin + ray.direction * t


def read_calibration(calibration_filepath):
    with open(calibration_filepath) as f:
        calibration = json.load(f)
    shelf_plane_normal = (
        calibration['shelf']['x_vec'], 
        calibration['shelf']['y_vec'],
        calibration['shelf']['z_vec']
    )
    products = []
    for product in calibration['products']:
        products.append([product['poi_x'], product['poi_y'], product['poi_z']])
    products = np.array(products)
    return Calibration(shelf_plane_normal=shelf_plane_normal, products=products)


def index_nearest(vector, matrix):
    return np.linalg.norm(matrix - vector, axis=1).argmin()


def parse_output_path(path: str):
    """Assume that last number before period is frame number.

    Arguments:
        path {str} -- Path of file

    Returns:
        [tuple(str)] -- 
    """
    path_split = path.split('_')
    prefix = '_'.join(path_split[:-2])
    depth_or_color = path_split[-2]
    frame_num, file_format = path_split[-1].split('.')
    return prefix, depth_or_color, frame_num, file_format


def get_next_frame(path):
    """Increment the frame.

    Arguments:
        path {str} -- The path of the file

    Returns:
        [type] -- [description]
    """

    prefix, depth_or_color, frame_num, file_format = parse_output_path(path)
    new_frame_num = int(frame_num) + 1
    result = '.'.join(['_'.join([prefix, depth_or_color, str(new_frame_num)]), file_format])
    return result


def take_average(func, n):
    """Returns the average of `n` results of a function.
    
    Arguments:
        func {func} -- Function to wrap
        n {int} -- Number of samples of function
    
    Returns:
        [func] -- Wrapper for taking the average n
    """
    def wrapper(*args, **kwargs):
        d = np.asarray([])
        for i in range(n):
            d = d.append(func(*args, **kwargs))
        np.average(d, axis=1)
    return wrapper