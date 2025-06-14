from abc import ABC, abstractmethod
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from config import Config
from typing import Any, Dict, List, Tuple


class Trainer(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        """Add preprocessor-specific arguments to the argument parser."""
        pass

    @staticmethod
    @abstractmethod
    def from_args(args: Any, dataloader: DataLoader) -> 'Trainer':
        pass

    @abstractmethod
    def save(self, dir: str, checkpoint_idx: int) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load(checkpoint_path: str, device: str) -> Tuple['Trainer', int]:
        pass

    @staticmethod
    @abstractmethod
    def get_dataloader(file_list: List[str], shuffle: bool, num_workers: int, worker_init_fn) -> 'DataLoader':
        pass

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> Dict:
        pass

    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> Dict:
        pass

    @abstractmethod
    def predict_splits(self, config: Config, track_id: str, source_rider_id: str, target_rider_id: str) -> List[str]:
        """
        Return a list of predicted splits 'MM:HH:FF' for the target rider that correspond to the positions of each split for the source rider
        """
        pass
