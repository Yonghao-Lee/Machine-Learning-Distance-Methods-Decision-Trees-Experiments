import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from knn import KNNClassifier
from helpers import plot_decision_boundaries
from run_knn_experiements import load_data

def run_visualization():
    X_train, Y_train, X_test, Y_test = load_data()

    # These results are from section 5.1
    k_max = 1
    k_min = 3000

    print(f"Plotting for k_max = {k_max} and k_min = {k_min}")

    print(f"1. Plotting L2 with k={k_max}")
    knn_l2_best = KNNClassifier(k=k_max, distance_metric='l2')
    knn_l2_best.fit(X_train, Y_train)
    plot_decision_boundaries(knn_l2_best, X_test, Y_test, title=f"L2 Metric, k={k_max} (best k-value)")

    print(f"2. Plotting L2 with k={k_min}")
    knn_l2_worst = KNNClassifier(k=k_min, distance_metric='l2')
    knn_l2_worst.fit(X_train, Y_train)
    plot_decision_boundaries(knn_l2_worst, X_test, Y_test, title=f"L2 Metric, k={k_min} (worst k-value)")

    # Plot for L1
    knn_l1_best = KNNClassifier(k=k_max, distance_metric='l1')
    knn_l1_best.fit(X_train, Y_train)
    plot_decision_boundaries(knn_l1_best, X_test, Y_test, title=f"L1 Metric, k={k_max} (best k-value)")

if __name__ == "__main__":
    run_visualization()

