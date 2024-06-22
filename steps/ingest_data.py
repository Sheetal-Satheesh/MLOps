import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Ingesting the data
    """

    def __init__(self, data_path: str):
        """
        :param data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        :return: pd.Dataframe
        """
        logging.info(f"Ingesting data from{self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from data_path
    :param data_path: path to the data
    :return: pd.Dataframe: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error While ingesting data: {e}")
        return e
