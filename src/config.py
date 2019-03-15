import pyrealsense2 as rs

STREAM_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'color_format': rs.format.bgr8,
    'depth_format': rs.format.z16
}

IP, PORT = "localhost", 9999

COLOR_ARRAY_DTYPE = 'uint8'
DEPTH_ARRAY_DTYPE = 'uint16'

MAGIC = b'ABCDEFGHIOJKLMNOPQRSTUVWXYZ'