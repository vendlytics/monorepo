# begin streaming video to a destination (filesystem or address)

import cv2
import numpy as np
import pyrealsense2 as rs

STREAM_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'color_format': rs.format.bgr8,
    'depth_format': rs.format.z16
}

class CameraNode:
    def __init__(self):
        self.color_stream = np.memmap(
            '/tmp/color_stream',
            mode='r', 
            shape=(STREAM_CONFIG['height'], STREAM_CONFIG['width'], 3)
        )
        self.depth_stream = np.memmap(
            '/tmp/depth_stream',
            mode='r', 
            shape=(STREAM_CONFIG['height'], STREAM_CONFIG['width'])
        )
    
    def __del__(self):
        self.color_stream, self.depth_stream = None, None
    
    def get_color_frame(self):
        image = self.color_stream
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    # TODO: Process depth before visualizing
    # def get_depth_frame(self):
    #     depth_values = self.depth_stream
    #     return depth_values 


if __name__ == "__main__":

    color_stream = np.memmap(
        '/tmp/color_stream', 
        mode='w+', 
        shape=(STREAM_CONFIG['height'], STREAM_CONFIG['width'], 3)
        )
    depth_stream = np.memmap(
        '/tmp/depth_stream', 
        mode='w+', 
        shape=(STREAM_CONFIG['height'], STREAM_CONFIG['width'])
        )

    pipeline = rs.pipeline()

    # configure depth and color
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        STREAM_CONFIG['width'], 
        STREAM_CONFIG['height'], 
        STREAM_CONFIG['color_format'], 
        STREAM_CONFIG['fps']
    )
    config.enable_stream(
        rs.stream.depth, 
        STREAM_CONFIG['width'], 
        STREAM_CONFIG['height'], 
        STREAM_CONFIG['depth_format'], 
        STREAM_CONFIG['fps']
    )

    # start stream
    pipeline.start(config)

    try:
        while True:
            # This call waits until a new coherent pair of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            frames = pipeline.wait_for_frames()
            
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()

            if not color or not depth: 
                continue

            color_stream[:] = color.as_frame().get_data()
            depth_stream[:] = depth.as_frame().get_data()
            
            # print(color.profile)
            # print(np.asanyarray(color_stream).shape)
            # print(np.asanyarray(depth_stream).shape)


    except Exception as e:
        print(e)
