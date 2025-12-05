import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

np.random.seed(0)


def load_data():
    print("Loading data...")
    # Load all three datasets
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('validation.csv')
    test_df = pd.read_csv('test.csv')

    X_train = train_df[['long', 'lat']].values
    Y_train = train_df['state'].values

    X_val = val_df[['long', 'lat']].values
    Y_val = val_df['state'].values

    X_test = test_df[['long', 'lat']].values
    Y_test = test_df['state'].values

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def run_tree_experiments():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()

    max_depths = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes = [50, 100, 1000]

    print("Running tree experiments...")
    print(f"{'Depth':<6} | {'Leaves':<6} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10}")
    print("#" * 60)

    best_val_acc = 0
    best_params = None
    best_tree_test_acc = 0

    for max_depth in max_depths:
        for max_leaf_node in max_leaf_nodes:
            tree_classifier = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_node, random_state=0)
            tree_classifier.fit(X_train, Y_train)
            train_acc = tree_classifier.score(X_train, Y_train)
            val_acc = tree_classifier.score(X_val, Y_val)
            test_acc = tree_classifier.score(X_test, Y_test)

            print(f"{max_depth:<6} | {max_leaf_node:<6} | {train_acc:<10.2f} | {val_acc:<10.2f} | {test_acc:<10.2f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (max_depth, max_leaf_node)
                best_tree_test_acc = test_acc

    print("\n")
    print("#" * 60)
    print("Best Tree Results from Validation Set")
    print("#" * 60)
    print(f"Best Params: Depth={best_params[0]}, Leaves={best_params[1]}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Corresponding Test Accuracy: {best_tree_test_acc:.4f}")


if __name__ == "__main__":
    run_tree_experiments()

