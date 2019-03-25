# goal of this is to generate a JSON calibration file for each camera at
# each shelf that looks like:

# `camera_0_calibration.json`
#
# {
#     'shelf': {
#         'calibrated_distance_mm': 3791.9,
#         'yaw': 18371.7,
#         'pitch': 37819.1,
#         'roll': 29378.4,
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

import cv2
import json
import numpy as np
import os
import pandas as pd

from src.face_detect import face_detect
from src.gazenet import Gazenet
from src.utils import calculate_poi, euler_angle_to_vector, Ray, logger

CONFIG_DIR = 'config'
CONFIG_FILE_DIR = os.path.join(CONFIG_DIR, 'files')

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


def angle_and_depth(gazenet, config):
    """
        Using frames at which the subject is looking straight at the shelf, get:
    - shelf norm
    - distance from shelf
    
    Arguments:
        gazenet {Gazenet} -- Face angle model
        config   {dict}  -- dict of paths to color and depth images
    
    Returns:
        [type] -- [description]
    """
    # read image
    np_img = cv2.imread(os.path.join(CONFIG_FILE_DIR, config['color_path']))

    # find face
    face_bbox = face_detect(np_img)
    min_x, min_y, max_x, max_y = face_bbox
    logger.info("Bounding box: " + str(face_bbox))

    # get angle
    yaw, pitch, roll = gazenet.image_to_euler_angles(np_img, face_bbox)

    # get average depth
    depth_df = pd.DataFrame.from_csv(os.path.join(CONFIG_FILE_DIR, config['depth_path']))
    avg_depth = np.mean(depth_df.iloc[min_y:max_y, min_x:max_x].values) * 1000 # convert to mm
    
    return (yaw, pitch, roll), avg_depth


def calibrate_shelf(gazenet, shelf_config):
    """
    Using frames at which the subject is looking straight at the shelf, get:
    - shelf norm
    - distance from shelf
    """
    return angle_and_depth(gazenet, shelf_config)
    

def calibrate_product(gazenet, product_config, plane_norm):
    """Pass in image of person looking straight at a particular product.

    Arguments:
        gazenet {Gazenet} -- Face angle model
        image {numpy.array} -- calibration image for shelf plane norm

    Returns:
        numpy.array -- coordinates in 3D of POI of shelf plane
    """
    (yaw, pitch, _), avg_depth = angle_and_depth(gazenet, product_config)
    gaze_z = avg_depth

    # read image
    np_img = cv2.imread(os.path.join(CONFIG_FILE_DIR, product_config['color_path']))

    # find face
    face_bbox = face_detect(np_img)
    min_x, min_y, max_x, max_y = face_bbox 

    # gaze_z should come from depth
    gaze_origin = np.array([
        np.mean([min_x, max_x]), 
        np.mean([min_y, max_y]), 
        gaze_z])
    gaze_vec = euler_angle_to_vector(yaw, pitch)
    
    # TODO: double check this - that gaze_z replaces shelf_dist_estimate?
    return calculate_poi(
        plane_norm, Ray(
            origin=gaze_origin, direction=gaze_vec), gaze_z)


def write_results(results, filename, format='json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # read calibration configuration
    c = CalibrationConfig(yaml_path='config/{}'.format(CONFIG_PATH) )
    logger.info("Read shelf plane config: \n%s", json.dumps(c.shelf_plane, indent=2)) 
    logger.info("Read products config: \n%s", json.dumps(c.products, indent=2))

    g = Gazenet(GAZENET_PATH)

    # calibrate shelf
    shelf_plane_norm, calibrated_distance_mm = calibrate_shelf(g, c.shelf_plane)
    logger.info("Calculated shelf norm: %s", str(shelf_plane_norm))
    logger.info("Distance of subject from shelf: %s", str(calibrated_distance_mm))

    # calibrate products
    product_pois = []
    for product in c.products:
        for name, paths in product.items():
            poi_x, poi_y, poi_z = calibrate_product(g, paths, shelf_plane_norm)
            product_pois.append({
                'name': name,
                'poi_x': float(poi_x),
                'poi_y': float(poi_y),
                'poi_z': float(poi_z)
            })

    # write json
    shelf_yaw, shelf_pitch, shelf_roll = shelf_plane_norm
    results = {
        'shelf': {
            'calibrated_distance_mm': calibrated_distance_mm,
            'yaw': float(shelf_yaw),
            'pitch': float(shelf_pitch),
            'roll': float(shelf_roll)
        },
        'products': product_pois
    }
    write_results(results, 'config/' + JSON_OUTPUT_PATH)
