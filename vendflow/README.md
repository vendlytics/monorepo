# Vendflow

Organizes and runs the different models, and processes their outputs to give a single report for one video stream.

Runs in this order:

1. `Vendnet` - for face detection of multiple faces in the frame and their locations in the frame.
2. `Vendgaze` - for the extraction of the Euler angles of each face in the frame.
3. `Vendface` - for the tokenization of each face in the frame into a unique embedding that can be used to identify the same person in multiple frames and videos.
4. `Vendstats` - handles the report generation from the text file that is generated from the single incoming video stream.

`Vendflow` is responsible for the orchestration of all of the above tasks.