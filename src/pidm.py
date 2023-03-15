"""Main script for running the pose generator"""

import sys
import os
import argparse
import logging
from pathlib import Path

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

from src.predict import Predictor
from IPython.display import Image

from src import paths
from src.log.log import initialize_logger

logger = logging.getLogger(__name__)

PIDM_LOG_PATH = paths.BASE_DIR / "logs" / "pidm.log"
PERSON_JPEG_PATH = paths.INPUT_DIR / "person.jpg"
DATASET_URL = "https://drive.google.com/file/d/1VffS0PiGkQhmsbyIVGvqARqRVwwTXFIZ/view?usp=sharing"
ZIP_FILE = paths.BASE_DIR / "checkpoints_data.zip"


def download_dataset() -> None:
    """"""
    if not (paths.DATA_DIR.is_dir() and paths.CHECKPOINTS_DIR.is_dir()):
        if paths.DATA_DIR.is_dir():
            paths.DATA_DIR.unlink()
            
        if paths.CHECKPOINTS_DIR.is_dir():
            paths.CHECKPOINTS_DIR.unlink()
            
        os.system(f"gdown --fuzzy {DATASET_URL} && unzip ./checkpoints_data.zip")
        
    if ZIP_FILE.is_file():
        ZIP_FILE.unlink()


def main(image_path: str) -> int:
    """
    Main script for running the pose generator

    Args:
        image_path (str): Image to use for pose generation.

    Returns:
        int: Exit code
    """
    
    exit_code = 0
    
    try:
        initialize_logger(log_path=PIDM_LOG_PATH)
        logger.info("Start of PIDM pose generation")
        
        download_dataset()
        
        predictor = Predictor()
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
    
    args, _ = parser.parse_known_args()
    
    sys.exit(main(image_path=args.image_path))
