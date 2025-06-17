from abc import ABC, abstractmethod
from argparse import ArgumentParser
from enum import Enum

import numpy as np
from config import Config
from typing import Any, Sequence


class SplitType(str, Enum):
    """Enum for dataset split types."""
    TRAIN = "train"
    VAL = "val"


class SampleGenerator(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        """Add preprocessor-specific arguments to the argument parser."""
        pass

    @staticmethod
    @abstractmethod
    def from_args(args: Any, config: Config) -> 'SampleGenerator':
        """
        Create and configure a Preprocessor instance from parsed arguments, config, and track-to-set mapping.

        Args:
            args: Parsed command-line arguments.
            config: Configuration object containing video metadata.

        Returns:
            Preprocessor: An instance of the implementing Preprocessor subclass.
        """
        pass

    @abstractmethod
    def get_features(self, video_feature_cache: dict, **kwargs) -> Sequence[np.ndarray]:
        """
        Compute features given a dictionary of metadata for an ML sample

        Args:
            row: Dict of sample metadata

        Returns:
            None
        """
        pass
