import logging
from zenml import step
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from src.evaluation import MSE, RMSE, R2


@step
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "mse"], Annotated[float, "r2_score"], Annotated[float, "rmse"]
]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)

        r2 = R2()
        r2 = r2.calculate_score(y_test, prediction)

        rmse = RMSE()
        rmse = rmse.calculate_score(y_test, prediction)
        return mse, r2, rmse
    except Exception as e:
        raise e
