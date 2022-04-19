from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient
import logging

import mlflow.sklearn
import os
import pandas as pd
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def load_model():
    # mlflow tracking
    mlflow_uri = f"http://{os.getenv('MLFLOW_SERVICE_SERVICE_HOST')}:{os.getenv('MLFLOW_SERVICE_SERVICE_PORT')}"
    mlflow.set_tracking_uri(mlflow_uri)

    # experiment
    experiment_name = "workshop"
    mlflow.set_experiment(experiment_name)
    app.logger.info(
        f"experiment_name: {mlflow.get_experiment_by_name(name=experiment_name)}"
    )

    model_name = "sk-learn-elastic-net-reg-model"

    # get model version
    client = MlflowClient()
    latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
    latest_production_version = latest_version_info[0].version
    app.logger.info(f"model_version: {latest_production_version}")

    # load model
    model_uri = client.get_model_version_download_uri(
        name=model_name, version=latest_production_version
    )
    model = mlflow.sklearn.load_model(model_uri=model_uri)

    return model


model = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    to_predict = pd.DataFrame(request.json, index=[0])
    prediction = model.predict(to_predict)

    return jsonify(json.loads(pd.Series(prediction).to_json(orient="values")))


if __name__ == "__main__":
    app.run()
