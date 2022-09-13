"""
This module defines the following routines used by the 'train' step of the regression pipeline:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model pipeline.
"""


def estimator_fn():
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    from sklearn.linear_model import LogisticRegression

    with mlflow.start_run(experiment_id="2555133066289073", run_name="logistic_regression") as mlflow_run:
        # AutoML balanced the data internally and use _automl_sample_weight_0dbb to calibrate the probability distribution
        sklr_sample_weight = X_train.loc[:, "_automl_sample_weight_0dbb"].to_numpy()

        model.fit(X_train, y_train, classifier__sample_weight=sklr_sample_weight)

        # Log metrics for the training set
        sklr_training_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_train, y_train, prefix="training_", pos_label=1, sample_weight=sklr_sample_weight)

        # Log metrics for the validation set
        sklr_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_", pos_label=1)

        # Log metrics for the test set
        sklr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_", pos_label=1)

        # Display the logged metrics
        sklr_val_metrics = {k.replace("val_", ""): v for k, v in sklr_val_metrics.items()}
        sklr_test_metrics = {k.replace("test_", ""): v for k, v in sklr_test_metrics.items()}
        display(pd.DataFrame([sklr_val_metrics, sklr_test_metrics], index=["validation", "test"]))
