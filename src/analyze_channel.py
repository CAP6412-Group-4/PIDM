"""Outputs each channel"""

import sys
import logging
import argparse

import numpy as np
import torchvision.transforms as transforms



def main(pose: int) -> int:
    print(pose)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pose", required=True, action="store", help="Specify the pose you want to output. Between poses 0 and 100")
    args, _ = parser.parse_known_args()    

    sys.exit(main(pose=args.pose))
