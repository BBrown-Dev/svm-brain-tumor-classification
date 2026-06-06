from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_kernels(X_train, y_train, X_test, y_test, metrics_fn):
    """ Train the kernels """
    kernels = ["linear", "poly", "rbf"]
    results = {}

    # Train and evaluate each kernel
    for k in kernels:
        clf = SVC(kernel=k, C=1.0, gamma="scale")
        clf.fit(X_train, y_train)
        results[k] = metrics_fn(clf, X_test, y_test)

    return results

def tune_rbf(X_train, y_train):
    """ Tune RBF kernels """
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.001]
    }

    # Use GridSearchCV to find the best hyperparameters with 5-fold cross-validation
    grid = GridSearchCV(
        SVC(kernel="rbf"),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # Fit the grid search to the training data
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_