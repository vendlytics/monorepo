# takes as input:
# - CSV gaze file, and
# - CSV product calibration file
#
# outputs:
# - CSV with column per product at each frame
# with the confidence that you are looking at each product

import pandas as pd
import numpy as np

def calculate_rectangle_centroid(min_x, min_y, max_x, max_y):
    """
    Returns a tuple (x, y) which gives the position of the face bounding box.
    """
    return (min_x + max_x) / 2, (min_y + max_y) / 2  


def calculate_rectangle_area(min_x, min_y, max_x, max_y):
    """
    Returns the pixel area of the face bounding box.
    """
    return (max_x - min_x) * (max_y - min_y)

def convert_from_euler_angle_to_vector(yaw_angle, pitch_angle):
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

class Pipeline:
    def __init__(self, name, raw_gaze_path, product_calibration_path):
        self.name = name
        self.raw_gaze_path = raw_gaze_path
        self.product_calibration_path = product_calibration_path

    def ingest_raw_gaze(self):
        gaze_df = pd.read_csv(self.raw_gaze_path, delimiter=' ', header=None)
        gaze_df.columns = ['frame_number', 'x_angle', 'y_angle', 'z_angle']
        return gaze_df

    def ingest_product_calibration(self):
        return pd.read_csv(self.product_calibration_path, header=None)

    def convert_all_euler_angles_to_vectors(self, gaze_df):
        z_vectors = np.array([])
        
        for frame in gaze_df:
            gaze_x_angle = gaze_df.loc[gaze_df['frame_number'] == frame]['x_angle'].value[0]
            gaze_y_angle = gaze_df.loc[gaze_df['frame_number'] == frame]['y_angle'].value[0]
            gaze_z_angle = gaze_df.loc[gaze_df['frame_number'] == frame]['z_angle'].value[0]
            z_vectors = np.append(
                z_vectors, 
                convert_from_euler_angle_to_vector(gaze_x_angle, gaze_y_angle)
            )

        return z_vectors

    def create_probabilty_scores_for_products(self):
        raise NotImplementedError

    def write_results_per_frame(self):
        raise NotImplementedError

    def run(self):
        gaze_df = self.ingest_raw_gaze()
        product_calibration = self.ingest_product_calibration()
        self.convert_all_euler_angles_to_vectors()
        self.create_probabilty_scores_for_products()
        self.write_results_per_frame()


if __name__ == "__main__":
    RAW_GAZE_PATH = "notebooks/visualization/data/frame_to_gaze.txt"
    PRODUCT_CALIBRATION_PATH = "notebooks/visualization/data/frame_to_bbox.txt"

    pipeline = Pipeline(
        'fucked pipeline',
        RAW_GAZE_PATH,
        PRODUCT_CALIBRATION_PATH
    )

    pipeline.run()
