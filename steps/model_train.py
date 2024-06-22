import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    try:
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            print("Data Types in X_train:")
            print(X_train.dtypes)
            trained_model = model.train(X_train, y_train)
            print("Trained model,", trained_model)
            return trained_model  # Return the trained model here
        else:
            raise ValueError(f"Model {config.model_name} not supported")

    except ValueError as ve:
        logging.error(f"ValueError in training model: {ve}")
        raise ve

    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
