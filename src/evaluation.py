import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


class Evaluation(ABC):
    """
    Abstract class defining the strategy for evalation of our models
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE:{mse}")
            return mse
        except Exception as e:
            raise e


class R2(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"MSE:{r2}")
            return r2
        except Exception as e:
            raise e


class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate MSE")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"MSE:{rmse}")
            return rmse
        except Exception as e:
            raise e
