from typing import Type

from trainer.trainer_regression import RegressionTrainer

from .interface import Trainer
from .trainer_classification import ClassificationTrainer


def get_trainer_class(trainer_type: str) -> Type[Trainer]:
    if trainer_type == "classifier":
        return ClassificationTrainer
    if trainer_type == "regressor":
        return RegressionTrainer
    raise ValueError(f"Unknown trainer type: {trainer_type}")
