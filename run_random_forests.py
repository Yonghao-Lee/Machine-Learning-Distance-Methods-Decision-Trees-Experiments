import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from helpers import plot_decision_boundaries

# 1. Set Seed
np.random.seed(0)

def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    X_train = train_df[['long', 'lat']].values
    Y_train = train_df['state'].values
    X_test = test_df[['long', 'lat']].values
    Y_test = test_df['state'].values
    return X_train, Y_train, X_test, Y_test

def run_random_forests_experiments():
    X_train, Y_train ,X_test, Y_test = load_data()
    print("Training Random Forests...")
    print("300 Trees with Depth 6")

    # Initialize the random forest
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0, n_jobs=-1)

    rf_model = rf_model.fit(X_train, Y_train)
    acc = rf_model.score(X_test, Y_test)
    print(f"Test Accuracy: {acc:.4f}")

    print("Plotting The Cat!")
    plot_decision_boundaries(rf_model, X_test, Y_test, title="Random Forests: Depth = 6, 300 Trees")

if __name__ == "__main__":
    run_random_forests_experiments()
