from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate, \
    clips_array

root_path = 'notebooks/data_for_demo_v2/'
preds_path = root_path + 'smooth_predictions.npy'
video1_path = root_path + 'vendgaze/output-test.avi'
video4_path = root_path + 'petstore-demo-rear.mp4'

fps = 24
offset = 0
start_frame = 0
end_frame = 12000
start_sec = offset + start_frame / fps
end_sec = offset + end_frame / fps

idx_to_name = [
    "product 0",
    "product 1",
    "product 2",
    "product 3",
    "product 4",
    "product 5",
    "product 6",
    "product 7"
]

do_downscale = False
out_filename = "overlay_downscaled_with_gaze.mp4" if do_downscale else "overlay_demo.mp4"


def pred_to_text_clip(pred):
    text = "No Prediction"

    if not np.isnan(pred):
        text = idx_to_name[int(pred)]

    return TextClip(text, fontsize=50, bg_color='white', 
                    color='black', stroke_width=3).set_duration(1 / fps)


def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)


def get_video(path, with_audio):
    video = VideoFileClip(path).subclip(start_sec, end_sec)
    if not with_audio:
        video = video.without_audio()
    if do_downscale:
        video = video.resize(0.25)
    return video


preds = np.load(preds_path)[start_frame:end_frame]
video1 = get_video(video1_path, with_audio=True)
video4 = get_video(video4_path, with_audio=False)

stacked_video = clips_array([[video1, video4]])

text_clip = concatenate([pred_to_text_clip(pred)
                         for pred in preds], method="compose")

print("creating compositevideoclip")
result = CompositeVideoClip([stacked_video, text_clip])

# so we don't get permission issues when writing
delete_file_if_exists(out_filename)

print("writing video out to: {}".format(out_filename))
result.write_videofile(out_filename, fps=fps, codec='mpeg4')
