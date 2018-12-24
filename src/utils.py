from collections import namedtuple
import numpy as np


Ray = namedtuple('Ray', ['origin', 'direction'])


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
