import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
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


def run_xgboost_experiments():
    X_train, Y_train, X_test, Y_test = load_data()
    print("Training XGBoost with 300 Trees and Depth 6...")

    xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1)

    xgb_model = xgb_model.fit(X_train, Y_train)

    acc = xgb_model.score(X_test, Y_test)
    print(f"Test Accuracy: {acc:.4f}")

    print("Drawing Another Cat!")
    plot_decision_boundaries(xgb_model, X_test, Y_test, title="XGBoost: Depth = 6, 300 Trees")


if __name__ == "__main__":
    run_xgboost_experiments()
