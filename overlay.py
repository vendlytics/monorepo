from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate

preds_path = 'notebooks/visualization/notebook_data/smooth_prediction.npy'
video_path = 'notebooks/visualization/notebook_data/videos/Camera 1 - Vendlytics Prototype.mp4'

fps = 24
offset = 28.4
start_frame = 0
end_frame = 600
start_sec = offset + start_frame / fps
end_sec = offset + end_frame / fps

out_filename = "overlay.mp4"

def pred_to_text_clip(pred):
    return TextClip(str(pred), fontsize=70, color='white').set_duration(1 / fps)

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)

preds = np.load(preds_path)[start_frame:end_frame]
video = VideoFileClip(video_path).subclip(start_sec, end_sec)

text_clip = concatenate([pred_to_text_clip(pred) for pred in preds], method="compose")

result = CompositeVideoClip([video, text_clip])

# so we don't get permission issues when writing
delete_file_if_exists(out_filename)
result.write_videofile(out_filename, fps=fps, codec='mpeg4')
