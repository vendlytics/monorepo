from collections import namedtuple
import logging
import numpy as np
import pyrealsense2 as rs

# Logger
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Ray
Ray = namedtuple('Ray', ['origin', 'direction'])

# yields (color np.array, depth np.array)
def read_bag(filepath):
    # create context
    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, filepath, repeat_playback=False)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 0)

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


def abs_euclidean_distance(vec_1, vec_2):
    return np.absolute(euclidean(vec_1, vec_2))
