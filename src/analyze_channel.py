"""Outputs each channel"""

import os
import sys
import logging
import argparse
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from src import paths
from src.log.log import initialize_logger

logger = logging.getLogger(__name__)

ANALYSIS_LOG = paths.BASE_DIR / "logs" / "analysis.log"


def load_npy(npy_path):
    return np.load(str(npy_path))


def output_channels(pose_npy) -> None:
    tensor = transforms.ToTensor()(pose_npy).cuda()

    for idx, point in enumerate(tensor):
        logger.info("Channel %s", idx)
        for row in point:
            for col in row:
                print(f"{col} ", end="")
            print()

def output_joint(pose, pose_npy):
    tensor = transforms.ToTensor()(pose_npy).cuda()

    for idx, point in enumerate(tensor):
        rgb_pose = (255 * point)
        print(type(rgb_pose))
        print(rgb_pose.shape)

def main(pose: int) -> int:
    
    try:
        reference_pose = f"reference_pose_{pose}.npy"

        logger.info("Outputting Reference Pose: %s", pose)

        pose_npy = load_npy(npy_path=paths.TARGET_POSE / reference_pose)
        output_joint(pose, pose_npy)

    except Exception as ex:
        logger.exception(ex)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pose", required=True, action="store", 
                        help="Specify the pose you want to output. Between poses 0 and 100")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args, _ = parser.parse_known_args()    

    initialize_logger(log_path=ANALYSIS_LOG, debug=args.debug)

    sys.exit(main(pose=args.pose))
