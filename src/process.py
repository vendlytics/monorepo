# takes as input:
# - CSV gaze file, and
# - CSV product calibration file
#
# outputs:
# - CSV with column of product being looked at for each frame
# with the confidence that you are looking at each product
# and prediction + smoothed prediction

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# np.set_printoptions(threshold=np.nan)

import os
import pandas as pd
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean

import src.utils as utils


USE_BBOX = False


class Pipeline:
    def __init__(self, inputs, outputs):
        # INPUTS
        self.raw_gaze_path = inputs['raw_gaze_path']
        self.products_on_shelf_plane_coords_path = inputs['products_on_shelf_plane_coords_path']
        self.shelf_plane_norm_path = inputs['shelf_plane_norm_path']
        self.dist_estimate_path = inputs['dist_estimate_path']

        # OUTPUT
        self.raw_output_path = outputs['raw_output_path']
        self.smooth_predictions_path = outputs['smooth_predictions_path']

    # LOAD STAGE 1
    def ingest_raw_gaze(self):
        gaze_df = pd.read_csv(self.raw_gaze_path, delimiter=' ', header=None)
        gaze_df.columns = [
            'frame_number',
            'x_angle',
            'y_angle',
            'z_angle',
            'x_center',
            'y_center']
        return gaze_df

    # LOAD STAGE 3
    def ingest_products_on_shelf_plane_coords(self):
        return np.loadtxt(
            self.products_on_shelf_plane_coords_path,
            delimiter=' ')

    # LOAD STAGE 4
    def ingest_shelf_plane_norm(self):
        return np.loadtxt(self.shelf_plane_norm_path, delimiter=' ')

    # LOAD STAGE 5
    def ingest_dist_estimate(self):
        return np.loadtxt(self.dist_estimate_path)

    def compute_distance_scores_for_products(
            self,
            gaze_df,
            product_plane_coords,
            plane_n_vector,
            shelf_dist_estimate):

        # TODO: Choose mean of calibration
        # calibration_gaze_origin = np.array([616.0491945, 126.5688365, 0])

        for i in range(len(product_plane_coords)):
            gaze_df['product {}'.format(i)] = None

        # 1. for each gaze vector, get the point of intersection on the plane
        for idx in gaze_df.index:
            print(idx)
            if np.isnan(gaze_df.iloc[idx]['x_angle']):
                for i in range(len(product_plane_coords)):
                    gaze_df.at[idx, 'product {}'.format(i)] = np.nan
                gaze_df.at[idx, 'predictions'] = np.nan
                gaze_df.at[idx, 'smooth_predictions'] = np.nan
                continue

            gaze_x_angle = gaze_df.iloc[idx]['x_angle']
            gaze_y_angle = gaze_df.iloc[idx]['y_angle']
            z_vector = utils.euler_angle_to_vector(
                gaze_x_angle, gaze_y_angle)

            gaze_origin = np.array(
                [gaze_df.iloc[idx]['x_center'], gaze_df.iloc[idx]['y_center'], 0])
            # gaze_origin = np.zeros(3)

            poi = utils.calculate_poi(
                plane_n_vector,
                utils.Ray(origin=gaze_origin, direction=z_vector),
                shelf_dist_estimate)

            # 2. calculate for each product the distance between the product's
            # vector and POI on plane
            for i in range(len(product_plane_coords)):
                gaze_df.at[idx, 'product {}'.format(
                    i)] = utils.abs_euclidean_distance(poi, product_plane_coords[i])

            product_cols = [
                'product {}'.format(i) for i in range(
                    len(product_plane_coords))]
            gaze_df.at[idx, 'predictions'] = np.argmin(
                gaze_df.iloc[idx][product_cols])[0]

        # 3. make regular and smoothed predictions based on product distances
        gaze_df['smooth_predictions'] = medfilt(
            gaze_df['predictions'], kernel_size=89)

        return gaze_df

    def write_smooth_predictions(self, smooth_predictions):
        np.save(self.smooth_predictions_path, smooth_predictions)
        return True

    def write_results_per_frame(self, new_gaze_df):
        new_gaze_df.to_csv(self.raw_output_path)
        return True

    def run(self):
        gaze_df = self.ingest_raw_gaze()
        product_plane_coords = self.ingest_products_on_shelf_plane_coords()
        shelf_plane_norm = self.ingest_shelf_plane_norm()
        dist_estimate = self.ingest_dist_estimate()

        new_gaze_df = self.compute_distance_scores_for_products(
            gaze_df, product_plane_coords, shelf_plane_norm, dist_estimate)

        if self.write_results_per_frame(new_gaze_df) and self.write_smooth_predictions(
                new_gaze_df['smooth_predictions'].values):
            print("DONE: {} frames".format(new_gaze_df.shape[0]))


if __name__ == "__main__":

    RAW_GAZE_PATH = "notebooks/data_for_demo_v2/gaze/output-test.txt"
    PRODUCTS_ON_SHELF_PLANE_COORDS_PATH = "notebooks/data_for_demo_v2/product_plane_coords.txt"
    SHELF_PLANE_NORM_PATH = "notebooks/data_for_demo_v2/shelf_plane_norm.txt"
    DIST_ESTIMATE_PATH = "notebooks/data_for_demo_v2/dist_estimate.txt"

    RAW_OUTPUT_PATH = os.path.join(
        "notebooks/data_for_demo_v2/", "output_without_bbx.txt")

    SMOOTH_PREDICTIONS_PATH = os.path.join(
        "notebooks/data_for_demo_v2/", "smooth_predictions.npy")

    pipeline = Pipeline(
        inputs={
            'raw_gaze_path': RAW_GAZE_PATH,
            'products_on_shelf_plane_coords_path': PRODUCTS_ON_SHELF_PLANE_COORDS_PATH,
            'shelf_plane_norm_path': SHELF_PLANE_NORM_PATH,
            'dist_estimate_path': DIST_ESTIMATE_PATH},
        outputs={
            'raw_output_path': RAW_OUTPUT_PATH,
            'smooth_predictions_path': SMOOTH_PREDICTIONS_PATH})

    pipeline.run()
