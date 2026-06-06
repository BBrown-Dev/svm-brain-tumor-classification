import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def visualize_decision_boundary(model, X, y):
    """ Visualize the decision boundary of the model using PCA for dimensionality reduction """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Train a new SVM on the 2D PCA-transformed data for visualization
    clf = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma)
    clf.fit(X_2d, y)

    # Create a grid of points to evaluate the model
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    # Generate a grid of points to evaluate the model
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Predict the labels for each point in the grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot the decision boundary and the original data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10)
    plt.title("Decision Boundary (PCA 2D)")
    plt.tight_layout()
    plt.savefig("results/decision_boundary.png")
    plt.show()