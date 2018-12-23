import pyrealsense2 as rs
import numpy as np


# yields (color np.array, depth np.array)
def read_bag(filepath):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, filepath, repeat_playback=False)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 0)
    align = rs.align(rs.stream.color)
    pipeline_profile = pipeline.start(config)

    # If we don't set this, reading the frames takes as long as the recording duration
    device = pipeline_profile.get_device()
    device.as_playback().set_real_time(False)

    while True:
        success, frames = pipeline.try_wait_for_frames()
        if not success:
            return
        frames = align.process(frames)
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        yield np.array(color_frame.get_data()), np.array(depth_frame.get_data())


if __name__ == '__main__':
    import argparse
    import scipy.misc
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        required=True,
        help="Path to the bag file")
    args = parser.parse_args()
    i = 0
    for color, depth in read_bag(args.filepath):
        i += 1
        if i % 100 == 50:
            print('Current frame:', i)
    print(i, 'frames read')
    scipy.misc.imshow(color)
    scipy.misc.imshow(depth)
