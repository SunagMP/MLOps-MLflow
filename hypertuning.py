import mlflow

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.2)

params = {
    "n_estimators" : [10, 50, 100],
    "max_depth" : [None, 10, 20, 30]
}

mlflow.set_experiment("randam forest with child")
with mlflow.start_run() as parent:
    rf = RandomForestClassifier()

    gs = GridSearchCV(param_grid= params, estimator=rf, verbose= 2, cv=5, n_jobs=-1)
    gs.fit(x_train, y_train)

    for i in range(len(gs.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(gs.cv_results_['params'][i])
            mlflow.log_metric("accuracy",gs.cv_results_['mean_test_score'][i])
            mlflow.sklearn.log_model(rf)

    # logging best parameters
    best_params = gs.best_params_
    mlflow.log_params(best_params)

    # logging the best score
    best_score = gs.best_score_
    mlflow.log_metric("accuracy", best_score)

    #logging the model along with file
    mlflow.sklearn.log_model(rf, "random-forest-best-model")
    mlflow.log_artifact(__file__)

    print("done")