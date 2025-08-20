import mlflow

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

mlflow.set_tracking_uri('http://localhost:5000')

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.2)

n_estimators = 8
max_depth = 5


with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # logging the evaluation metric of model
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precison", precision)
    mlflow.log_metric("recall", recall)

    # logging the params
    mlflow.log_params({
        "n_estimators" : n_estimators,
        "max_depth" : max_depth
    })

    # logging the artifacts
    mlflow.log_artifact(__file__)

    print("done")