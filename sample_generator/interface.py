from abc import ABC, abstractmethod
from argparse import ArgumentParser
from enum import Enum
from config import Config
from typing import Any


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
    def compute_and_cache_features(self, row: dict) -> None:
        """
        Compute features given a dictionary of metadata for an ML sample

        Args:
            row: Dict of sample metadata

        Returns:
            None
        """
        pass

    @abstractmethod
    def save_batch(self, save_dir: str) -> None:
        pass
