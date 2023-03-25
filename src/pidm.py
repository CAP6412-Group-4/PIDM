"""Main script for running the pose generator"""

import glob

import sys
import os
import argparse
import logging
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

from src.predict import Predictor

from src import paths
from src.log.log import initialize_logger

logger = logging.getLogger(__name__)

PIDM_LOG_PATH = paths.BASE_DIR / "logs" / "pidm.log"
PERSON_JPEG_PATH = paths.INPUT_DIR / "person.jpg"
DATASET_URL = "https://drive.google.com/file/d/1VffS0PiGkQhmsbyIVGvqARqRVwwTXFIZ/view?usp=sharing"
ZIP_FILE = paths.BASE_DIR / "checkpoints_data.zip"


def download_dataset() -> None:
    """Downloads the zip file from the google drive."""

    # Checks dirs
    if not (paths.DATA_DIR.is_dir() and paths.CHECKPOINTS_DIR.is_dir()):
        logger.info("Downloading dataset from Google Drive: %s", DATASET_URL)
        
        if paths.DATA_DIR.is_dir():
            paths.DATA_DIR.unlink()
            
        if paths.CHECKPOINTS_DIR.is_dir():
            paths.CHECKPOINTS_DIR.unlink()
            
        os.system(f"gdown --fuzzy {DATASET_URL} && unzip ./checkpoints_data.zip")
        
    if ZIP_FILE.is_file():
        ZIP_FILE.unlink()


def main(image_path: str, debug: bool = False) -> int:
    """
    Main script for running the pose generator

    Args:
        image_path (str): Image to use for pose generation.
        debug: Flag for displaying debug logs

    Returns:
        int: Exit code
    """
    
    exit_code = 0
    
    try:
        initialize_logger(log_path=PIDM_LOG_PATH, debug=debug)
        logger.info("Start of PIDM pose generation")
        
        download_dataset()
        
        predictor = Predictor()
        
        predictor.predict_pose(image=image_path, sample_algorithm="ddim", num_poses=1, nsteps=50)
    except Exception as ex:
        exit_code = 1
        logger.exception(ex)
    finally:
        logger.info("End of PIDM pose generattion: %s", exit_code)    
    
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--image-path", required=False, default=str(PERSON_JPEG_PATH),
                        help="Path to image to use for pose generation.")
    parser.add_argument("-d", "--debug", required=False, default=False, action="store_true",
                        help="Flag for also displaying DEBUG logs. "
                        "This would output more detailed logs.")
    
    args, _ = parser.parse_known_args()
    
    sys.exit(main(image_path=args.image_path, debug=args.debug))
