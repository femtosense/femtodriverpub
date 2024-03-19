import os
import numpy as np

from femtorun import FemtoRunner
from femtodriverpub import SPURunner, FakeSPURunner

import logging

import argparse
from argparse import RawTextHelpFormatter # allow carriage returns in help strings, for displaying model options

import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
            description="run a pickled FASMIR or FQIR on hardware. Compare with output of FB's SimRunner")

    parser.add_argument("path_to_femtocrux_images",
        help="path with output of femtocrux, the memory images\n")
    parser.add_argument("--debug", default=False, action='store_true',
        help="set debug log level")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    hw_runner = FakeSPURunner(args.path_to_femtocrux_images)
    hw_runner.reset()

