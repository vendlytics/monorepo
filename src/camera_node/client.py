# TODO: When multiple cameras are connected, need to identify each one for server

import asyncio

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

IP, PORT = "localhost", 9999


@asyncio.coroutine
def write_numpy(color_array, depth_array, loop):
    _, writer = yield from asyncio.open_connection(IP, PORT, loop=loop)

    print("sending:")
    print(color_array)
    print(depth_array)

    # we split color and depth arrays with newline on every write
    # TODO: Log number of bytes written
    writer.write(color_array.tobytes() + b'\n' + depth_array.tobytes())
    yield from writer.drain()
    writer.close()


if __name__ == "__main__":
    # asyncio
    loop = asyncio.get_event_loop()

    # pipeline = rs.pipeline()

    # # configure depth and color
    # config = rs.config()
    # config.enable_stream(
    #     rs.stream.color,
    #     STREAM_CONFIG['width'], 
    #     STREAM_CONFIG['height'], 
    #     STREAM_CONFIG['color_format'], 
    #     STREAM_CONFIG['fps']
    # )
    # config.enable_stream(
    #     rs.stream.depth, 
    #     STREAM_CONFIG['width'], 
    #     STREAM_CONFIG['height'], 
    #     STREAM_CONFIG['depth_format'], 
    #     STREAM_CONFIG['fps']
    # )

    # # start stream
    # pipeline.start(config)

    try:
        while True:
            # This call waits until a new coherent pair of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            
            # frames = pipeline.wait_for_frames()
            
            # color = frames.get_color_frame()
            # depth = frames.get_depth_frame()

            # if not color or not depth: 
            #     continue

            # color_stream = color.as_frame().get_data()
            # depth_stream = depth.as_frame().get_data()

            color_stream = np.zeros((STREAM_CONFIG['height'], STREAM_CONFIG['width'], 3))
            depth_stream = np.zeros((STREAM_CONFIG['height'], STREAM_CONFIG['width']))

            loop.run_until_complete(write_numpy(color_stream, depth_stream, loop))
            
            # print(color.profile)
            # print(np.asanyarray(color_stream).shape)
            # print(np.asanyarray(depth_stream).shape)


    except Exception as e:
        print(e)

    loop.close()
