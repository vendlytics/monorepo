from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate, \
    clips_array

preds_path = 'smooth_prediction.npy'
video1_path = 'gaze_overlay_24fps.mp4'
video4_path = 'Camera 4 - Vendlytics Prototype.mp4'

fps = 24
offset = 28.4
start_frame = 0
end_frame = 12000
start_sec = offset + start_frame / fps
end_sec = offset + end_frame / fps

idx_to_name = [
    "Hershey's Reese",
    "Twix Bites",
    "Snickers Bites",
    "Sensations Rice",
    "Uncle Ben's Rice",
    "Purina Dog Treats",
    "Pedigree Dog Treats",
]

do_downscale = False 
out_filename = "overlay_downscaled_with_gaze.mp4" if do_downscale else "overlay_demo.mp4"

def pred_to_text_clip(pred):
    text = idx_to_name[int(pred)]
    return TextClip(text, fontsize=30, color='white').set_duration(1 / fps)

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

text_clip = concatenate([pred_to_text_clip(pred) for pred in preds], method="compose")

result = CompositeVideoClip([stacked_video, text_clip])

# so we don't get permission issues when writing
delete_file_if_exists(out_filename)

result.write_videofile(out_filename, fps=fps, codec='mpeg4')
