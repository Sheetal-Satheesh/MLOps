from zenml import pipeline, step
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline
def training_pipelines(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)

    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2_score, rmse = evaluate_model(model, X_test, y_test)
