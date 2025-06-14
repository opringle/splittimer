from typing import Type

from .interface import SampleGenerator
from .sample_generator_classification import ClassifierSampleGenerator
from .sample_generator_regression import RegressionSampleGenerator


def get_sample_generator_class(generator_type: str) -> Type[SampleGenerator]:
    if generator_type == "classifier":
        return ClassifierSampleGenerator
    elif generator_type == "regressor":
        return RegressionSampleGenerator
    raise ValueError(f"Unknown sample generator type: {generator_type}")
