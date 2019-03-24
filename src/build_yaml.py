# Read yaml and save all of the files
import argparse
import yaml
import os

from src.calibrate import CalibrationConfig


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
                os.rename(
                    os.path.join(args.from_dir, fp), 
                    os.path.join(args.to_dir, fp)
                )
    