from typing import Type

from preprocessor.preprocessor_regression import RegressionPreprocessor
from .interface import Preprocessor
from .preprocessor_classification import ClassifierPreprocessor


def get_preprocessor_class(preprocessor_type: str) -> Type[Preprocessor]:
    if preprocessor_type == "classifier":
        return ClassifierPreprocessor
    elif preprocessor_type == "regressor":
        return RegressionPreprocessor
    raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")
