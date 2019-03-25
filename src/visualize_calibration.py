"""
Given a calibration json output, visualizes the products and the shelf plane.

Usage: python3.6 -m src.visualize_calibration
"""

from src.utils import read_calibration
import argparse
import numpy as np
import matplotlib
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser()
parser.add_argument(
    "-cf",
    "--calibration_filepath",
    type=str,
    default='config/baseline_calibration.json',
    help="Path to a calibration json output")
args = parser.parse_args()

def surface_points(normal, point):
    # surface is normal[0]*x + normal[1]*y + normal[2]*z + d = 0
    d = -np.dot(point, normal)
    # make sure that x and y values cover each product with extra offset
    xs, ys = np.meshgrid(
        range(int(products[:, 0].min()) - 100, int(products[:, 0].max()) + 100),
        range(int(products[:, 1].min()) - 100, int(products[:, 1].max()) + 100))
    # for the x and y values above, z values that the surface would take
    zs = -1 * (normal[0] * xs + normal[1] * ys + d) / normal[2]
    return xs, ys, zs

shelf_plane_normal, products = read_calibration(args.calibration_filepath)

print('shelf_plane_normal:\n', shelf_plane_normal)

print('products:\n', products)

plt3d = plt.figure().gca(projection='3d')

plt3d.plot_surface(*surface_points(shelf_plane_normal, products[0]), alpha=0.2)

ax = plt.gca()

ax.scatter(products[:, 0], products[:, 1], products[:, 2])
plt.show()

