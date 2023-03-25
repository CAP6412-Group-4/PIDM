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

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from src import paths
from src.log.log import initialize_logger

logger = logging.getLogger(__name__)

ANALYSIS_LOG = paths.BASE_DIR / "logs" / "analysis.log"
POINTS = paths.BASE_DIR

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

def output_joint(dest, pose_npy):
    # (20, 256, 256)
    tensor = transforms.ToTensor()(pose_npy).cuda()

    for idx, point in enumerate(tensor):
        # point = 1 - point
        rgb_pose = (255 * point).cpu().detach().numpy()
        img = Image.fromarray(rgb_pose)
        
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(str(dest / f"point_{idx}.png"))

def save_pose(dest, pose_npy):
    tensor = transforms.ToTensor()(pose_npy).cuda()

    pose = torch.cat([1 - tensor[:3]], -2)
    pose_arr = (255*pose.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()).astype(np.uint8)[0]
    Image.fromarray(pose_arr).save(str(dest / "pose.png"))

def main(pose: int) -> int:
    
    try:
        reference_pose = f"reference_pose_{pose}.npy"
        reference_pose_dir = paths.BASE_DIR / f"reference_pose_{pose}"
        logger.error(reference_pose_dir)

        reference_pose_dir.mkdir(exist_ok=True)

        logger.info("Outputting Reference Pose: %s", pose)

        pose_npy = load_npy(npy_path=paths.TARGET_POSE / reference_pose)

        save_pose(reference_pose_dir, pose_npy)
        output_joint(reference_pose_dir, pose_npy)

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
