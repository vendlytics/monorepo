import json


class ExtractAttention:

    def __init__(self, job_id, calibration_json_path):
        self.job_id = job_id
        self.calibration_json_path = calibration_json_path
        self.calibration_json = None

    def run(self):
        with open(self.calibration_json_path) as f:
            self.calibration_json = json.loads(f)
