
"""
Generate a JSON calibration file for each camera at
each shelf that looks like:


`camera_0_calibration.json`

{
    'shelf': {
        'calibrated_distance_m': 3791.9,
        'yaw': 18371.7,
        'pitch': 37819.1,
        'roll': 29378.4,
    },
    'products': [
        {
            'name': 'Product 0',
            'poi_x': 332.0,
            'poi_y': 398.3,
            'poi_z': 298.1
        },
        {
            'name': 'Product 1',
            'poi_x': 938.1,
            'poi_y': 198.1,
            'poi_z': 380.2
        }
    ]
}

Usage: 

python3.6 -m src.calibrate
"""

import json
import os
from typing import List, Dict
import numpy as np
import pandas as pd
import cv2

from src.face_detect import face_detect
from src.gazenet import Gazenet
from src.utils import calculate_poi, euler_angle_to_vector, Ray, \
                      logger, get_next_frame

NEXT_N = 5
CONFIG_DIR = 'config'
CONFIG_FILE_DIR = os.path.join(CONFIG_DIR, 'files')

VIDEO_OUTPUT_PATH = os.path.join('output')

CONFIG_PATH = 'baseline_calibration.yml'
JSON_OUTPUT_PATH = 'baseline_calibration.json'
GAZENET_PATH = 'models/gazenet/hopenet_robust_alpha1.pkl'

class CalibrationConfig:

    def __init__(self, config_dict=None, yaml_path=None):
        """
        Initializes a configuration file jead in from a YAML that specifies the 
        paths of each image file used for calibrating.
        
        Arguments:
            config_dict {dict} -- Configuration dictionary
            from_yaml   {bool} -- indicate if reading from YAML
        """
        if yaml_path:
            config_dict = CalibrationConfig.parse_yaml(yaml_path)

        if not config_dict:
            raise ValueError("No yaml_path and config_dict. Need one of two")

        self.shelf_plane = config_dict['shelf']
        self.products = config_dict['products']

    @staticmethod
    def parse_yaml(yaml_path):
        import yaml

        with open(yaml_path, 'r') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as e:
                print(e)


def generate_next_n_paths(start_path: str, n: int) -> List[str]:
    """
    Generate a list of paths that are up to alphanumerically n after the 
    start_path.
    
    Arguments:
        start_path {str} -- [description]
        n {int} -- [description]
    
    Returns:
        List[str] -- [description]
    """
    d = []
    successes, attempts = 0, 0
    cur_path = start_path
    while successes <= n:
        path_to_check = os.path.join(VIDEO_OUTPUT_PATH, cur_path)
        if os.path.exists(path_to_check):
            d.append(path_to_check)
            successes += 1
        cur_path = get_next_frame(cur_path)
        attempts += 1
        if attempts > (2 * n):
            raise Exception('too many attempts')
    return d


def angle_and_depth(gazenet, color_image_path: str, depth_image_path: str, 
                is_product: bool = False):
    """
    Using frames at which the subject is looking straight at the shelf, get:
    - gaze vector
    - distance from shelf

    Arguments:
        gazenet {Gazenet} -- Face angle model
        config    {dict}  -- dict of paths to color and depth images

    Returns:
        [tuple[float], float] -- Tuple of yaw, pitch and roll and float of median depth
    """
    np_img = cv2.imread(color_image_path)
    face_bbox = face_detect(np_img)
    min_x, min_y, max_x, max_y = face_bbox

    if is_product: logger.info("Product bounding box: " + str(face_bbox))
    else: logger.info("Shelf bounding box: " + str(face_bbox))

    yaw, pitch, roll = gazenet.image_to_euler_angles(np_img, face_bbox)

    # get median depth
    depth_df = pd.DataFrame.from_csv(depth_image_path)
    median_depth = np.median(depth_df.iloc[min_y:max_y, min_x:max_x].values)

    return (yaw, pitch, roll), median_depth, (min_x, min_y, max_x, max_y)


def calibrate_shelf(gazenet, shelf_config: Dict):
    """
    Using frames at which the subject is looking straight at the shelf, get:
    - shelf norm
    - distance from shelf
    
    Arguments:
        gazenet {[type]} -- [description]
        shelf_config {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    color_paths = generate_next_n_paths(shelf_config['color_path'], NEXT_N)
    depth_paths = generate_next_n_paths(shelf_config['depth_path'], NEXT_N)
    
    results = []
    for cp, dp in zip(color_paths, depth_paths):
        (yaw, pitch, _), gaze_z, (min_x, min_y, max_x, max_y) = angle_and_depth(gazenet, cp, dp)
        gaze_x = np.mean([min_x, max_x])
        gaze_y = np.mean([min_y, max_y])
        results.append(np.asarray((yaw, pitch, _, gaze_x, gaze_y, gaze_z)))
    avg_results = np.average(np.asarray(results), axis=0)
    avg_yaw, avg_pitch, avg_gaze_x, avg_gaze_y, avg_gaze_z = \
        avg_results[0], avg_results[1], avg_results[3], avg_results[4], avg_results[5]
    avg_shelf_norm = euler_angle_to_vector(avg_yaw, avg_pitch)
    avg_origin = (avg_gaze_x, avg_gaze_y, avg_gaze_z)
    
    return avg_shelf_norm, avg_origin


def calibrate_product(gazenet, product_config: Dict, plane_norm, shelf_gaze_origin: List[float]):
    """Pass in image of person looking straight at a particular product.

    Arguments:
        gazenet {Gazenet} -- Face angle model
        image {numpy.array} -- calibration image for shelf plane norm

    Returns:
        numpy.array -- coordinates in 3D of POI of shelf plane
    """
    color_paths = generate_next_n_paths(product_config['color_path'], NEXT_N)
    depth_paths = generate_next_n_paths(product_config['depth_path'], NEXT_N)
    
    results = []
    for cp, dp in zip(color_paths, depth_paths):
        (yaw, pitch, _), gaze_z, (min_x, min_y, max_x, max_y) = angle_and_depth(gazenet, cp, dp, is_product=True)
        gaze_vec = euler_angle_to_vector(yaw, pitch)
        gaze_origin = np.array([np.mean([min_x, max_x]), np.mean([min_y, max_y]), gaze_z])
        results.append(np.asarray([gaze_vec, gaze_origin]))
    avg_results = np.average(np.asarray(results), axis=0)
    avg_gaze_vec, avg_gaze_origin = avg_results[0], avg_results[1]
    avg_gaze_depth = avg_gaze_origin[2]

    # TODO: double check this - that gaze_z replaces shelf_dist_estimate?
    return calculate_poi(plane_norm, Ray(origin=avg_gaze_origin - shelf_gaze_origin, direction=avg_gaze_vec), avg_gaze_depth)


def write_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # read calibration configuration
    c = CalibrationConfig(yaml_path='config/{}'.format(CONFIG_PATH) )
    logger.info("Read shelf plane config: \n%s", json.dumps(c.shelf_plane, indent=2)) 
    logger.info("Read products config: \n%s", json.dumps(c.products, indent=2))

    g = Gazenet(GAZENET_PATH)

    # calibrate shelf
    shelf_plane_norm, shelf_gaze_origin = calibrate_shelf(g, c.shelf_plane)
    logger.info("Calculated shelf norm: %s", str(shelf_plane_norm))
    logger.info("Shelf origin: %s", str(shelf_gaze_origin))

    # calibrate products
    product_pois = []
    for product in c.products:
        for name, paths in product.items():
            logger.info('Calibrating: ' + name)
            poi_x, poi_y, poi_z = calibrate_product(g, paths, shelf_plane_norm, shelf_gaze_origin)
            product_pois.append({
                'name': name,
                'poi_x': float(poi_x),
                'poi_y': float(poi_y),
                'poi_z': float(poi_z)
            })

    # write json
    json_results = {
        'shelf': {
            'origin_x': float(shelf_gaze_origin[0]),
            'origin_y': float(shelf_gaze_origin[1]),
            'origin_z': float(shelf_gaze_origin[2]),
            'x_vec': float(shelf_plane_norm[0]),
            'y_vec': float(shelf_plane_norm[1]),
            'z_vec': float(shelf_plane_norm[2])
        },
        'products': product_pois
    }
    write_results(json_results, 'config/' + JSON_OUTPUT_PATH)
