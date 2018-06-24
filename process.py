# takes as input:
# - CSV gaze file, and
# - CSV product calibration file
#
# outputs:
# - CSV with column per product at each frame
# with the confidence that you are looking at each product

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

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

def calculate_poi(n_vector, i_vector, dist_estimate):
    """
    Finds the point of intersection of a vector on the shelf plane.
    
    n_vector is shelf plane vector i.e. normal to the shelf plane
    i_vector is intersecting vector
    dist_estimate is estimate of shelf distance from person
    
    Notation from http://www.ambrsoft.com/TrigoCalc/Plan3D/PlaneLineIntersection_.htm
    """
    p_vector = n_vector + dist_estimate
    t = -(np.dot(n_vector, p_vector) + dist_estimate) / float(np.dot(n_vector, i_vector))  
    return i_vector * t

def get_abs_euclidean_distance(vec_1, vec_2):
    return np.absolute(euclidean(vec_1, vec_2))


class Pipeline:
    def __init__(self, name, inputs, output_path):
        self.name = name
        
        # INPUTS
        self.raw_gaze_path = inputs['raw_gaze_path']
        self.products_on_shelf_plane_coords_path = inputs['products_on_shelf_plane_coords_path']
        self.shelf_plane_norm_path = inputs['shelf_plane_norm_path']
        self.dist_estimate_path = inputs['dist_estimate_path']

        # OUTPUT
        self.output_path = output_path

    # LOAD STAGE 1
    def ingest_raw_gaze(self):
        gaze_df = pd.read_csv(self.raw_gaze_path, delimiter=' ', header=None)
        gaze_df.columns = ['frame_number', 'x_angle', 'y_angle', 'z_angle']
        return gaze_df

    # LOAD STAGE 2
    def ingest_products_on_shelf_plane_coords(self):
        return np.loadtxt(self.products_on_shelf_plane_coords_path, delimiter=' ')

    # LOAD STAGE 3
    def ingest_shelf_plane_norm(self):
        return np.loadtxt(self.shelf_plane_norm_path, delimiter=' ')

    # LOAD STAGE 4
    def ingest_dist_estimate(self):
        return np.loadtxt(self.dist_estimate_path)

    def compute_distance_scores_for_products(self, gaze_df, product_plane_coords, 
                                             plane_n_vector, shelf_dist_estimate):

        for i in range(len(product_plane_coords)):
            gaze_df['product {}'.format(i)] = None

        # 1. for each gaze vector, get the point of intersection on the plane
        for idx in gaze_df.index:
            print(idx)
            gaze_x_angle = gaze_df.iloc[idx]['x_angle']
            gaze_y_angle = gaze_df.iloc[idx]['y_angle']
            gaze_z_angle = gaze_df.iloc[idx]['z_angle']
            
            z_vector = convert_from_euler_angle_to_vector(gaze_x_angle, gaze_y_angle)
            poi = calculate_poi(plane_n_vector, z_vector, shelf_dist_estimate)

            # 2. calculate for each product the distance between the product's vector and POI on plane
            for i in range(len(product_plane_coords)):
                gaze_df.at[idx, 'product {}'.format(i)] = get_abs_euclidean_distance(-poi, product_plane_coords[i])
        
        return gaze_df

    def write_results_per_frame(self, new_gaze_df):
        # write out to self.output_path
        new_gaze_df.to_csv(self.output_path)
        return True

    def run(self):
        gaze_df = self.ingest_raw_gaze()[:5000]
        product_plane_coords = self.ingest_products_on_shelf_plane_coords()
        print(product_plane_coords)
        shelf_plane_norm = self.ingest_shelf_plane_norm()
        dist_estimate = self.ingest_dist_estimate()

        new_gaze_df = self.compute_distance_scores_for_products(gaze_df, product_plane_coords, 
                                                                shelf_plane_norm, dist_estimate)
        
        if self.write_results_per_frame(new_gaze_df):
            print("DONE")


if __name__ == "__main__":
    RAW_GAZE_PATH = "notebooks/visualization/data/frame_to_gaze.txt"
    
    # not used
    BBOX_PATH = "notebooks/visualization/data/frame_to_bbox.txt"

    # the coordinates of products on shelf
    PRODUCTS_ON_SHELF_PLANE_COORDS_PATH = "notebooks/visualization/data/product_plane_coords.txt"
    
    # norm of the shelf plane
    SHELF_PLANE_NORM_PATH = "notebooks/visualization/data/shelf_plane_norm.txt" 
    
    # the gaze vectors for the products (not used)
    PRODUCT_GAZE_VECTORS_PATH = "notebooks/visualization/data/product_vectors.txt"
    
    # the set distance of calibrated face from shelf
    DIST_ESTIMATE_PATH = "notebooks/visualization/data/dist_estimate.txt"
    
    OUTPUT_PATH = "notebooks/visualization/data/output.txt"

    pipeline = Pipeline(
        'fucked pipeline',
        inputs={
            'raw_gaze_path': RAW_GAZE_PATH,
            'products_on_shelf_plane_coords_path': PRODUCTS_ON_SHELF_PLANE_COORDS_PATH,
            'shelf_plane_norm_path': SHELF_PLANE_NORM_PATH,
            'dist_estimate_path': DIST_ESTIMATE_PATH
        },
        output_path=OUTPUT_PATH
    )

    pipeline.run()
