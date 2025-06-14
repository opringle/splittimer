import numpy as np
import random
import logging
import itertools
import pandas as pd
from .interface import SampleGenerator
from utils import get_video_file_path, get_video_metadata
from typing import Set


class RegressionSampleGenerator(SampleGenerator):
    @classmethod
    def add_args(cls, parser):
        """Add command-line arguments specific to this preprocessor."""
        parser.add_argument("--F", type=int, default=50,
                            help="Number of frames per clip (for individual features)")
        parser.add_argument("--add_position_feature", action='store_true',
                            help="Add a feature to each sample for end index")
        parser.add_argument("--add_percent_completion_feature", action='store_true',
                            help="Add a feature to each sample for % completion")

    @staticmethod
    def from_args(args):
        """Create an instance from parsed arguments"""
        return RegressionSampleGenerator(args)

    def __init__(self, args):
        """Initialize the sample generator with arguments."""
        self.args = args

    def compute_and_cache_features(row: dict) -> None:
        pass
