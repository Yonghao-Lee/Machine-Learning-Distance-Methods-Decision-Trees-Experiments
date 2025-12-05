import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from knn import KNNClassifier

np.random.seed(0)


def load_data():
    train_data = pd.read_csv("train.csv")
    ad_test_data = pd.read_csv("AD_test.csv")

    X_train = train_data[['long', 'lat']].values
    Y_train = train_data['state'].values

    X_ad = ad_test_data[['long', 'lat']].values

    return X_train, Y_train, X_ad


def run_anomaly_detection():
    X_train, Y_train, X_ad = load_data()
    knn = KNNClassifier(k=5, distance_metric='l2')
    knn.fit(X_train, Y_train)

    dists, _ = knn.knn_distance(X_ad)

    sum_anomaly_scores = np.sum(dists, axis=1)

    # Find the top 50 scores, these will be referred to as anomalous points while the rest as normal points
    top_50_indices = np.argsort(sum_anomaly_scores)[-50:]  # This takes the last 50 indices in the sorted array

    is_anomalous = np.zeros(len(X_ad), dtype=bool)
    is_anomalous[top_50_indices] = True

    anomalies = X_ad[is_anomalous]
    normal = X_ad[~is_anomalous]

    print(f"Anomalies found: {len(anomalies)}")
    print(f"Normal points found: {len(normal)}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c='black', alpha=0.01, label='Training Data')

    plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', s=30, label='Anomalies')
    plt.scatter(normal[:, 0], normal[:, 1], c='blue', s=20, label='Normal')

    plt.title("Anomaly Detection (k=5)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_anomaly_detection()

