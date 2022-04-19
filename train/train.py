#
# train.py
#
#   MLflow model using ElasticNet (sklearn) and Plots ElasticNet Descent Paths
#
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
#   Combines the above with the Lasso Coordinate Descent Path Plot
#       http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
#       Original author: Alexandre Gramfort <alexandre.gramfort@inria.fr>; License: BSD 3 clause
#
#  Usage:
#    python train.py 0.01 0.01
#    python train.py 0.01 0.75
#    python train.py 0.01 1.0
#

import os
import warnings
import sys
import platform

import pandas as pd
import numpy as np
from itertools import cycle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

# Load Diabetes datasets
def get_dataset():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    cols = diabetes.feature_names + ["progression"]
    return pd.DataFrame(d, columns=cols)


# Evaluate metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


mlflow_uri = f"http://{os.getenv('MLFLOW_SERVICE_SERVICE_HOST')}:{os.getenv('MLFLOW_SERVICE_SERVICE_PORT')}"
mlflow.set_tracking_uri(mlflow_uri)
print("tracking_uri:", mlflow.get_tracking_uri())
mlflow.set_registry_uri(mlflow_uri)

experiment_name = "workshop"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
client = mlflow.tracking.MlflowClient()
run = client.create_run(experiment.experiment_id)
run_id = run.info.run_id

with mlflow.start_run(run_id=run_id):
    print("artifact_uri:", mlflow.get_artifact_uri())
    print("experiment_name:", client.get_experiment(run.info.experiment_id).name)
    print("run_id:", run.info.run_id)
    print("experiment_id:", run.info.experiment_id)

    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.python", platform.python_version())
    mlflow.set_tag("version.platform", platform.system())

    warnings.filterwarnings("ignore")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(get_dataset())

    # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    # Run ElasticNet
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("RMSE: %s" % rmse)
    print("MAE: %s" % mae)
    print("R2: %s" % r2)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    model_name = "sk-learn-elastic-net-reg-model"
    artifact_path = "sklearn-model"

    # Log model
    mlflow.sklearn.log_model(sk_model=lr, artifact_path=artifact_path)

    registered_model = mlflow.register_model(
        name=model_name, model_uri=f"runs:/{run.info.run_id}/{artifact_path}"
    )

    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True,
    )
