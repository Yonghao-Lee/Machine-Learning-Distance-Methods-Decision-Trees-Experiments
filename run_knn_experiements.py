import numpy as np
from knn import KNNClassifier
import time
import pandas as pd

np.random.seed(0)


def load_data():
    """Loads the train data and test data from the CSV files"""
    print("Loading data...")
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # Extract the features and the labels and store them in separate NumPy arrays
    X_train = train_data[['long', 'lat']].values
    Y_train = train_data['state'].values

    X_test = test_data[['long', 'lat']].values
    Y_test = test_data['state'].values

    print(f"Data loaded successfully: Train shape - {X_train.shape}, Test shape - {X_test.shape}")
    return X_train, Y_train, X_test, Y_test


def run_experiments():
    X_train, Y_train, X_test, Y_test = load_data()

    # Define the hyperparameters like in section 5.1
    k_values = [1, 10, 100, 1000, 3000]
    metrics = ['l1', 'l2']

    # Use a dictionary where keys are l1 and l2 and record the accuracies for each metric and k value
    results = {'l1': [], 'l2': []}

    for metric in metrics:
        for k in k_values:
            start_time = time.time()
            knn = KNNClassifier(k=k, distance_metric=metric)

            # Fit the data
            knn.fit(X_train, Y_train)

            # Predict the labels for the test set
            predictions = knn.predict(X_test)
            accuracy = np.mean(predictions == Y_test)

            time_elapsed = time.time() - start_time

            results[metric].append(accuracy)

    print("\n" + "=" * 42)
    print("RESULTS TABLE")
    print("=" * 42)
    print(f"{'k':<7} | {'L1':<10} | {'L2':<10}")
    print("-" * 42)

    for i, k_value in enumerate(k_values):
        acc_l1 = results['l1'][i]
        acc_l2 = results['l2'][i]
        print(f"{k_value:<7} | {acc_l1:.4f}    | {acc_l2:.4f}")

    # Find the best k value and the worst k value for l1 and l1
    l1_scores = results['l1']
    l2_scores = results['l2']

    best_l1_idx = np.argmax(l1_scores)
    best_l2_idx = np.argmax(l2_scores)

    worst_l1_idx = np.argmin(l1_scores)
    worst_l2_idx = np.argmin(l2_scores)

    best_k_l1 = k_values[best_l1_idx]
    best_k_l2 = k_values[best_l2_idx]

    worst_k_l1 = k_values[worst_l1_idx]
    worst_k_l2 = k_values[worst_l2_idx]

    print(f"\nBest k value for l1: {best_k_l1}")
    print(f"Worst k value for l1: {worst_k_l1}")
    print(f"Best k value for l2: {best_k_l2}")
    print(f"Worst k value for l2: {worst_k_l2}")


if __name__ == "__main__":
    run_experiments()
