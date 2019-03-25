# TODO: Rather than just printing parsed numpy, need to write stream to a file.
# TODO: What's the best way to store and organize numpy arrays/videos?
# TODO: Having logging out to separate file be a coroutine

import asyncio
from datetime import datetime

from src.config import COLOR_ARRAY_DTYPE, DEPTH_ARRAY_DTYPE, MAGIC, STREAM_CONFIG, IP, PORT
from src.utils import logger
import numpy as np


def parse_to_numpy(data, mode='color'):
    """Takes bytes from numpy and parses it back to numpy object.
    
    Arguments:
        data {bytes} -- bytes of from a single read
        mode {str}   -- either color or depth
    """
    if mode == 'color':
        return np.frombuffer(data, dtype=COLOR_ARRAY_DTYPE).reshape(
            (STREAM_CONFIG['height'], STREAM_CONFIG['width'], 3)
        )
    elif mode == 'depth':
        return np.frombuffer(data, dtype=DEPTH_ARRAY_DTYPE).reshape(
            (STREAM_CONFIG['height'], STREAM_CONFIG['width'])
        )
    else:
        raise ValueError


@asyncio.coroutine
def read_numpy(reader, _):
    try:
        data = yield from reader.read()
    except asyncio.streams.LimitOverrunError as e:
        print(e.consumed)

    color_numpy_bytes, depth_numpy_bytes = data.split(MAGIC)
    
    logger.info('Received {} bytes for color'.format(len(color_numpy_bytes)))
    logger.info('Received {} bytes for depth'.format(len(depth_numpy_bytes)))

    parse_to_numpy(color_numpy_bytes, mode='color')
    parse_to_numpy(depth_numpy_bytes, mode='depth')


def write_numpy(array, file_path):
    # TODO: In stream fashion, should write data to filesystem
    # TODO: Streams from the same camera should go to same directory
    pass


@asyncio.coroutine
def flow():
    # chaining coroutines to have reading and writing to file be decoupled
    loop.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    coro = asyncio.start_server(read_numpy, IP, PORT, loop=loop)
    server = loop.run_until_complete(coro)

    print("Serving on:", server.sockets[0].getsockname())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()

