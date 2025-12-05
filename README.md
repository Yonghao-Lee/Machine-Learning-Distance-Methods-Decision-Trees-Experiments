# Machine Learning: Distance Methods & Decision Trees

A comprehensive implementation and analysis of K-Nearest Neighbors (KNN) and Decision Tree algorithms for geographic classification, using US state prediction from longitude/latitude coordinates.

## Overview

This project explores various machine learning algorithms for classifying geographic locations (US states) based on longitude and latitude coordinates. The implementation includes:

- **K-Nearest Neighbors (KNN)** with L1 and L2 distance metrics
- **Decision Trees** with hyperparameter tuning
- **Random Forests** ensemble method
- **XGBoost** gradient boosting
- **Anomaly Detection** using distance-based methods

## Key Findings

- **KNN Performance**: Achieved 96.8% accuracy with k=1 and L2 metric; performance degrades significantly as k increases
- **Decision Trees**: Best model (Depth=20, Leaves=1000) achieved 97.74% test accuracy
- **Random Forests**: 300 trees with Depth=6 achieved 80.46% accuracy
- **XGBoost**: Achieved 96.50% accuracy, demonstrating the power of boosting over bagging

## Project Structure

```
.
├── knn.py                      # KNN classifier implementation using FAISS
├── decision_tree.py            # Decision tree experiments
├── run_knn_experiements.py     # KNN hyperparameter tuning
├── anomaly_detection.py        # Distance-based anomaly detection
├── visualize_tree.py           # Decision tree visualization
├── visulaize_knn.py            # KNN decision boundary visualization
├── run_random_forests.py       # Random forest experiments
├── xgboost_experiments.py      # XGBoost experiments
├── helpers.py                  # Utility functions for visualization
└── report.pdf                  # Detailed analysis report
```

## Requirements

```bash
numpy
pandas
matplotlib
scikit-learn
faiss-cpu  # or faiss-gpu for GPU acceleration
xgboost
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-distance-methods.git
cd ml-distance-methods
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn faiss-cpu xgboost
```

3. Ensure you have the required CSV files:
   - `train.csv`
   - `test.csv`
   - `validation.csv`
   - `AD_test.csv`

## Usage

### K-Nearest Neighbors Experiments
Run comprehensive KNN analysis with different k values and distance metrics:
```bash
python run_knn_experiements.py
```

### KNN Visualization
Visualize decision boundaries for different k values:
```bash
python visulaize_knn.py
```

### Decision Tree Analysis
Perform hyperparameter tuning for decision trees:
```bash
python decision_tree.py
```

### Decision Tree Visualization
Generate decision boundary visualizations:
```bash
python visualize_tree.py
```

### Anomaly Detection
Run distance-based anomaly detection:
```bash
python anomaly_detection.py
```

### Random Forests
Train and evaluate random forest models:
```bash
python run_random_forests.py
```

### XGBoost
Run XGBoost experiments:
```bash
python xgboost_experiments.py
```

## Results Summary

### KNN Performance (Test Accuracy)

| Metric | k=1   | k=10  | k=100 | k=1000 | k=3000 |
|--------|-------|-------|-------|--------|--------|
| L1     | 0.967 | 0.962 | 0.923 | 0.745  | 0.402  |
| L2     | 0.968 | 0.958 | 0.920 | 0.742  | 0.398  |

### Decision Tree Performance

| Configuration | Test Accuracy |
|---------------|---------------|
| Best (Depth=20, Leaves=1000) | 0.9774 |
| Limited Leaves (Depth=20, Leaves=50) | 0.84 |
| Limited Depth (Depth=6, Leaves=1000) | 0.58 |

### Ensemble Methods

| Method | Configuration | Test Accuracy |
|--------|---------------|---------------|
| Random Forest | 300 Trees, Depth=6 | 0.8046 |
| XGBoost | 300 Trees, Depth=6 | 0.9650 |

## 🔍 Key Insights

### Bias-Variance Tradeoff
- **Small k (k=1)**: High variance, low bias - captures fine details but sensitive to noise
- **Large k (k=3000)**: Low variance, high bias - smooth predictions but loses local patterns

### Distance Metrics
- **L2 (Euclidean)**: Produces circular equidistant contours and smooth, curved boundaries
- **L1 (Manhattan)**: Creates diamond-shaped contours with boundaries aligned to 45° angles

### Decision Tree Limitations
- Decision trees create axis-aligned rectangular partitions
- Complex non-rectangular shapes (like state borders) require many leaves to approximate
- Depth and leaf constraints directly limit model expressiveness

### Ensemble Learning
- **Random Forests** (Bagging): Reduces variance but cannot overcome high bias from shallow trees
- **XGBoost** (Boosting): Sequentially corrects errors, reducing both bias and variance

## Visualizations

The project generates several visualizations including:
- KNN decision boundaries with different k values and metrics
- Decision tree partitioning with various hyperparameter constraints
- Random Forest and XGBoost classification maps
- Anomaly detection scatter plots

## Dataset

The project uses geographic data with longitude and latitude coordinates mapped to US states:
- **Training set**: Large dataset of US city coordinates with state labels
- **Test set**: Held-out evaluation data
- **Validation set**: For hyperparameter tuning
- **Anomaly Detection set**: Special test set for outlier detection

## Implementation Details

### KNN Classifier
- Uses FAISS library for efficient nearest neighbor search
- Supports both L1 and L2 distance metrics
- Implements majority voting for classification

### Anomaly Detection
- Computes sum of distances to k nearest neighbors
- Flags top 50 points with highest distance scores as anomalies
- Visualizes normal vs anomalous points geographically

## Report

For detailed analysis, methodology, and visualizations.

## Contributing

This is an academic project, but suggestions and improvements are welcome. Feel free to open issues or submit pull requests.

## License

This project is part of an academic assignment. Please respect academic integrity policies if you're using this for learning purposes.

## Author

**Yonghao Lee**

## Acknowledgments

- AI Tool Usage: Gemini 2.5 Pro was used for coding assistance, conceptual clarification, and report formatting
- Libraries: scikit-learn, FAISS, XGBoost, matplotlib
- Dataset: US geographic coordinate data

