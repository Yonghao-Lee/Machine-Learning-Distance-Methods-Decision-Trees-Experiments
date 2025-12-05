import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from helpers import plot_decision_boundaries

def load_data():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    X_train = train_data[['long', 'lat']].values
    Y_train = train_data['state'].values

    X_test = test_data[['long', 'lat']].values
    Y_test = test_data['state'].values

    return X_train, Y_train, X_test, Y_test

def run_tree_visualization():
    X_train, Y_train, X_test, Y_test = load_data()

    # Here note that the best tree is of depth 20 and leaves 1000
    print("Print Best Tree: Depth = 20, Leaves = 1000")
    tree_best = DecisionTreeClassifier(max_depth=20, max_leaf_nodes=1000, random_state=0)
    tree_best.fit(X_train, Y_train)
    plot_decision_boundaries(tree_best, X_test, Y_test, title="Best Tree: Depth = 20, Leaves = 1000")

    # For question 6.2.5
    # We look at the best tree with leaves=50, from the table, we can choose depth=20
    best_tree_50_leaves = DecisionTreeClassifier(max_depth=20, max_leaf_nodes=50, random_state=0)
    best_tree_50_leaves.fit(X_train, Y_train)
    plot_decision_boundaries(best_tree_50_leaves, X_test, Y_test, title="Best Tree: Depth = 20, Leaves = 50")

    # For question 6.2.6
    # We look at the best tree with max depth 6, from the table, we can choose leaves=1000 or 50 or 100, we choose 1000
    best_tree_6_depth = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=1000, random_state=0)
    best_tree_6_depth.fit(X_train, Y_train)
    plot_decision_boundaries(best_tree_6_depth, X_test, Y_test, title="Best Tree: Depth = 6, Leaves = 1000")


if __name__ == "__main__":
    run_tree_visualization()
