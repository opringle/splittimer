from abc import ABC, abstractmethod
from argparse import ArgumentParser
from enum import Enum

import pandas
from config import Config
from typing import Any, Set


class SplitType(str, Enum):
    """Enum for dataset split types."""
    TRAIN = "train"
    VAL = "val"


class Preprocessor(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        """Add preprocessor-specific arguments to the argument parser."""
        pass

    @staticmethod
    @abstractmethod
    def from_args(args: Any, config: Config) -> 'Preprocessor':
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
    def generate_training_metadata(self, track_ids: Set[str], split_type: SplitType) -> 'pandas.DataFrame':
        """Generate training metadata using the instance's configuration."""
        pass
