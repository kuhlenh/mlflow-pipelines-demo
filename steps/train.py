def estimator_fn():
    """Returns an UNFITTED estimator that contains fit() and predict() method.
    Their input and output signatures should be compatible with sklearn estimators.
    """
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(criterion="mse", max_depth=12, random_state=641401879)
