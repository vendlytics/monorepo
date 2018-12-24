# goal of this is to generate a JSON calibration file for each camera at
# each shelf that looks like:

# `camera_0_calibration.json`
#
# {
#     'shelf': {
#         'calibrated_distance_mm': 3791.9,
#         'norm_x': 18371.7,
#         'norm_y': 37819.1,
#         'norm_z': 29378.4,
#     },
#     'products': [
#         {
#             'name': 'Product 0',
#             'poi_x': 332.0,
#             'poi_y': 398.3,
#             'poi_z': 298.1
#         },
#         {
#             'name': 'Product 1',
#             'poi_x': 938.1,
#             'poi_y': 198.1,
#             'poi_z': 380.2
#         }
#     ]
# }

import numpy as np

from src.face_detect import face_detect
from src.gazenet import Gazenet
from src.utils import calculate_poi, euler_angle_to_vector, Ray

GAZENET_PATH = '../models/gazenet/hopenet_robust_alpha1.pkl'


def get_shelf_plane_norm(gazenet, image):
    """Pass in the frame at which the person is standing in the middle and looking
    straight at the camera.

    Arguments:
        gazenet {Gazenet} -- Face angle model
        image {numpy.array} -- calibration image for shelf plane norm

    Returns:
        numpy.array -- vector of shelf plane norm
    """
    # TODO: face_detect() needs to return bbox coordinates of face in image, or
    # should change gazenet to allow for just accepting an image if image is
    # cropped
    bbox = face_detect(image)
    yaw, pitch, _ = gazenet.image_to_euler_angles(image, bbox)

    return euler_angle_to_vector(yaw, pitch)


def get_product_poi(gazenet, image, plane_norm):
    """Pass in image of person looking straight at particular product.

    Arguments:
        gazenet {Gazenet} -- Face angle model
        image {numpy.array} -- calibration image for shelf plane norm

    Returns:
        numpy.array -- coordinates in 3D of POI of shelf plane
    """
    # TODO: also get location of face x, y, z coords?
    bbox = face_detect(image)
    yaw, pitch, _ = gazenet.image_to_euler_angles(image, bbox)

    # gaze_z should come from depth
    gaze_origin = np.array([gaze_x, gaze_y, gaze_z])
    gaze_vec = euler_angle_to_vector(yaw, pitch)

    # TODO: double check this - gaze_z replaces shelf_dist_estimate
    return calculate_poi(
        plane_norm, Ray(
            origin=gaze_origin, direction=gaze_vec), gaze_z)


if __name__ == "__main__":
    g = Gazenet(GAZENET_PATH)

    shelf_plane_image = None  # TODO: get image
    product_poi_images = [None, None, None]

    shelf_plane_norm = get_shelf_plane_norm(g, shelf_plane_image)

    results = {'shelf': shelf_plane_norm, 'product_pois': [
        get_product_poi(g, img, shelf_plane_norm) for img in product_poi_images]}

    # write json
    # (results, 'somewhere')
