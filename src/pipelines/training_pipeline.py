import pandas as pd
from typing import Dict, Tuple, Any

class TrainingPipeline:
    """
    A pipeline class for training and optimizing machine learning models

    Attributes:
        config (Dict[str, Any]): Configuration dictionary with training parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the TrainingPipeline with the provided configuration.

        Args:
            config (Dict[str, Any]): Full pipeline configuration dictionary.
        """
        self.config: Dict[str, Any] = config['training']

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepares training and testing datasets by applying a target transformation
        and splitting by fraction (no shuffling).
        """
        return x_train, x_test, y_train, y_test

    def tune_hyperparams(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Any:
        """
        Perform hyperparameter tuning using Optuna, then retrain the model
        using the best configuration on the full training data.
        """
        return model

        
    def run(self, df: pd.DataFrame) -> Any:
        """
        Run the full training pipeline:
        """
        x_train, x_test, y_train, y_test = self.prepare_dataset(df)
        model, _ = self.tune_hyperparams(x_train, y_train, x_test, y_test)
        return model