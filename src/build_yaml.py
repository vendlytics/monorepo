
"""
Moves photos that match the filename into config/files

Usage:
python3.6 -m src.build_yaml \
    -cp config/baseline_calibration.yml \
    -fd ~/src/github.com/output \
    -td ~/src/github.com/vendlytics/config/files
"""
import argparse
import shutil
import os

from src.calibrate import CalibrationConfig
from src.utils import get_next_frame


parser = argparse.ArgumentParser()
parser.add_argument(
    "-cp",
    "--configpath",
    type=str,
    required=True,
    help="Path to the config file")
parser.add_argument(
    "-fd",
    "--from_dir",
    type=str,
    required=True,
    help="Directory to look for images/csv")
parser.add_argument(
    "-td",
    "--to_dir",
    type=str,
    default="config/files",
    help="Directory to add images/csvs to"
)
args = parser.parse_args()


def read_yaml(path):
    c = CalibrationConfig(yaml_path=path)
    return c.shelf_plane, c.products


if __name__ == "__main__":
    shelf_plane, products = read_yaml(args.configpath)

    for product in products:
        for p, f in product.items():
            for fp in f.values():
                cur_fp = fp
                success = False
                while not success:
                    try:
                        shutil.copy(
                            os.path.join(args.from_dir, cur_fp), 
                            os.path.join(args.to_dir, cur_fp)
                        )
                        success = True
                    except FileNotFoundError:
                        cur_fp = get_next_frame(cur_fp)

    # key should just be `color_path` and `depth_path`
    for key in shelf_plane:
        shutil.copy(
            os.path.join(args.from_dir, shelf_plane[key]),
            os.path.join(args.to_dir, shelf_plane[key])
        )
        
        