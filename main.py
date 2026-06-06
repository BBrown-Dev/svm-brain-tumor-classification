import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocess import load_brain_tumor_dataset
from src.features import extract_hog_features
from src.train import train_kernels, tune_rbf
from src.evaluate import compute_metrics, evaluate_model
from src.visualize import visualize_decision_boundary

def main():
    """ Main function """
    print("Loading dataset...")
    X, y, class_names = load_brain_tumor_dataset("data")

    print("Extracting HOG features...")
    X_hog = extract_hog_features(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hog)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nComparing kernels...")
    kernel_results = train_kernels(
        X_train, y_train, X_test, y_test, compute_metrics
    )
    print(kernel_results)

    print("\nTuning RBF hyperparameters with 5-fold CV...")
    best_model, best_params, best_cv = tune_rbf(X_train, y_train)
    print("Best Params:", best_params)
    print("Best CV Accuracy:", best_cv)

    print("\nEvaluating best model...")
    evaluate_model(best_model, X_test, y_test, class_names)

    print("\nVisualizing decision boundary...")
    visualize_decision_boundary(best_model, X_train, y_train)

if __name__ == "__main__":
    main()