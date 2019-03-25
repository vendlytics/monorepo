# TODO: When multiple cameras are connected, need to identify each one for server

import asyncio
from src.config import MAGIC, STREAM_CONFIG, IP, PORT
from src.utils import logger

import cv2
import numpy as np
import pyrealsense2 as rs


def build_payload(color_array, depth_array):
    """Build color and depth bytes payload from raw arrays, using MAGIC separator.
    
    Arguments:
        color_array {np.array} -- Color information
        depth_array {np.array} -- Depth information
    
    Returns:
        bytes -- Color and depth array into bytes.
    """

    color_payload = color_array.tobytes()
    depth_payload = depth_array.tobytes()
    final_payload = color_payload + MAGIC + depth_payload
    
    logger.info('Color payload is {} bytes'.format(len(color_payload)))
    logger.info('Depth payload is {} bytes'.format(len(depth_payload)))
    logger.info('Separator payload is {} bytes'.format(len(MAGIC)))
    logger.info('Final payload is {} bytes'.format(len(final_payload)))

    return final_payload


@asyncio.coroutine
def write_numpy(color_array, depth_array, loop):
    _, writer = yield from asyncio.open_connection(IP, PORT, loop=loop)
    writer.write(build_payload(color_array, depth_array))
    yield from writer.drain()
    writer.close()


if __name__ == "__main__":
    # asyncio
    loop = asyncio.get_event_loop()

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

            # TODO: Uncomment for testing
            # color_stream = np.zeros((STREAM_CONFIG['height'], STREAM_CONFIG['width'], 3))
            # depth_stream = np.zeros((STREAM_CONFIG['height'], STREAM_CONFIG['width']))

            color_stream = np.asarray(color.as_frame().get_data())
            depth_stream = np.asarray(depth.as_frame().get_data())

            loop.run_until_complete(write_numpy(color_stream, depth_stream, loop))
            
            # print(color.profile)


    except Exception as e:
        print(e)

    loop.close()
