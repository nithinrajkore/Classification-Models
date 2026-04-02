# Classification Models

![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Python](https://img.shields.io/badge/Python-3.10.9-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A collection of **7 supervised classification algorithms** implemented in Python using scikit-learn. Each algorithm is demonstrated on the same dataset, making it easy to compare approaches and understand how different classifiers partition feature space. Includes decision boundary visualizations for every model.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Common Preprocessing Pipeline](#common-preprocessing-pipeline)
- [Algorithms](#algorithms)
  - [1. Logistic Regression](#1-logistic-regression)
  - [2. K-Nearest Neighbors](#2-k-nearest-neighbors)
  - [3. Support Vector Machine (Linear)](#3-support-vector-machine-linear)
  - [4. Kernel SVM (RBF)](#4-kernel-svm-rbf)
  - [5. Naive Bayes](#5-naive-bayes)
  - [6. Decision Tree](#6-decision-tree)
  - [7. Random Forest](#7-random-forest)
- [Algorithm Comparison](#algorithm-comparison)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Key Concepts Glossary](#key-concepts-glossary)
- [References](#references)

---

## Repository Structure

```
Classification-Models/
├── Decision Trees/
│   ├── Decision Tree.ipynb
│   └── Social_Network_Ads.csv
├── K-Nearest Neighbors/
│   ├── K-Nearest Neighbors.ipynb
│   └── Social_Network_Ads.csv
├── Kernel SVM/
│   ├── Kernel SVM.ipynb
│   └── Social_Network_Ads.csv
├── Logistic Regression/
│   ├── Logistic Regression.ipynb
│   └── Social_Network_Ads.csv
├── Naive Bayes/
│   ├── Naive Bayes.ipynb
│   └── Social_Network_Ads.csv
├── Random Forest/
│   ├── Random Forest Classification.ipynb
│   └── Social_Network_Ads.csv
├── Support Vector Machines/
│   ├── Support Vector Machines - Classification.ipynb
│   └── Social_Network_Ads.csv
├── Classification_Pros_Cons.pdf
└── README.md
```

| Folder | Algorithm | Notebook File |
|---|---|---|
| `Logistic Regression/` | Logistic Regression | `Logistic Regression.ipynb` |
| `K-Nearest Neighbors/` | K-Nearest Neighbors | `K-Nearest Neighbors.ipynb` |
| `Support Vector Machines/` | Support Vector Machine (Linear) | `Support Vector Machines - Classification.ipynb` |
| `Kernel SVM/` | Kernel SVM (RBF) | `Kernel SVM.ipynb` |
| `Naive Bayes/` | Naive Bayes | `Naive Bayes.ipynb` |
| `Decision Trees/` | Decision Tree | `Decision Tree.ipynb` |
| `Random Forest/` | Random Forest | `Random Forest Classification.ipynb` |

---

## Dataset

All 7 notebooks use the same dataset: **`Social_Network_Ads.csv`**

| Column | Type | Description |
|---|---|---|
| `Age` | int | Age of the user |
| `EstimatedSalary` | int | Estimated annual salary (USD) |
| `Purchased` | int (0 or 1) | Target: whether the user purchased the product |

| Property | Value |
|---|---|
| Total rows | 400 |
| Features used | `Age`, `EstimatedSalary` |
| Target | `Purchased` (binary: 0 = No, 1 = Yes) |
| Train / Test split | 300 / 100 (75% / 25%) |
| Random seed | `random_state = 0` |

A sample single prediction is run in every notebook:

```python
# Predict whether a 30-year-old with $87,000 salary will purchase
print(classifier.predict(sc.transform([[30, 87000]])))
# Output: [0]  (will NOT purchase)
```

---

## Common Preprocessing Pipeline

Every notebook follows the **exact same preprocessing pipeline** before fitting the classifier:

```python
import numpy as np
import pandas as pd

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Feature matrix and target vector
X = dataset.iloc[:, :-1].values   # Age, EstimatedSalary
y = dataset.iloc[:, -1].values    # Purchased

# Train / Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

**Decision boundary visualization** (same code used in every notebook):

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# For both Training Set and Test Set:
X_set, y_set = X_train, y_train  # or X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
)

plt.contourf(X1, X2,
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Algorithm Name - Training Set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

---

## Algorithms

---

### 1. Logistic Regression

**Theory:**
Logistic Regression models the probability that an input belongs to a class using the **sigmoid function**. Despite its name, it is a classification — not regression — algorithm.

$$P(y=1 \mid X) = \sigma(w^TX + b) = \frac{1}{1 + e^{-(w^TX + b)}}$$

The decision boundary is **linear** in feature space.

| Parameter | Value | Description |
|---|---|---|
| `random_state` | `0` | Random seed for reproducibility |

**Implementation:**

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
```

**Evaluation:**

```python
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
```

**Decision Boundary:** Linear — draws a straight line separating the two classes.

---

### 2. K-Nearest Neighbors

**Theory:**
KNN is a **non-parametric, instance-based** algorithm. To classify a new point, it finds the **K nearest training samples** (using a distance metric) and assigns the majority class.

$$\hat{y} = \text{mode}\{y_i : x_i \in N_K(x)\}$$

where $N_K(x)$ is the set of $K$ nearest neighbors of $x$.

**Minkowski distance** (generalizes Manhattan and Euclidean):

$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

With $p=2$ this is Euclidean distance.

| Parameter | Value | Description |
|---|---|---|
| `n_neighbors` | `5` | Number of nearest neighbors |
| `metric` | `'minkowski'` | Distance metric |
| `p` | `2` | Power parameter (2 = Euclidean distance) |

**Implementation:**

```python
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
```

**Decision Boundary:** Non-linear, irregular — can capture complex shapes depending on K.

---

### 3. Support Vector Machine (Linear)

**Theory:**
SVM finds the **optimal hyperplane** that maximizes the margin between the two classes. Support vectors are the data points closest to the decision boundary.

$$\underset{w, b}{\text{minimize}} \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \geq 1 \;\; \forall i$$

With a linear kernel, the decision boundary is a straight line (in 2D).

| Parameter | Value | Description |
|---|---|---|
| `kernel` | `'linear'` | Linear kernel — no feature transformation |
| `random_state` | `0` | Random seed for reproducibility |

**Implementation:**

```python
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
```

**Decision Boundary:** Linear — a straight line that maximizes the class margin.

---

### 4. Kernel SVM (RBF)

**Theory:**
Kernel SVM extends the standard SVM by projecting data into a higher-dimensional space using a **kernel function**, making non-linearly separable data linearly separable in that higher space.

The **RBF (Radial Basis Function) kernel** is:

$$K(x, x') = e^{-\gamma \|x - x'\|^2}$$

where $\gamma$ controls the radius of influence of each support vector.

| Parameter | Value | Description |
|---|---|---|
| `kernel` | `'rbf'` | Radial Basis Function kernel |
| `random_state` | `0` | Random seed for reproducibility |

**Implementation:**

```python
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
```

**Decision Boundary:** Non-linear, smooth curves — can separate complex, non-linearly separable data.

---

### 5. Naive Bayes

**Theory:**
Naive Bayes applies **Bayes' theorem** with the "naive" assumption that all features are conditionally independent given the class label.

$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)} \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

**Gaussian Naive Bayes** assumes each feature follows a normal distribution:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

| Parameter | Value | Description |
|---|---|---|
| *(none)* | — | GaussianNB uses default priors estimated from training data |

**Implementation:**

```python
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

**Decision Boundary:** Non-linear, Gaussian-shaped — smooth curves based on feature distributions.

---

### 6. Decision Tree

**Theory:**
A Decision Tree recursively splits the feature space using axis-aligned thresholds to minimize impurity in the resulting partitions. **Entropy** measures the impurity at each node:

$$H(S) = -\sum_{c \in C} p_c \log_2(p_c)$$

**Information Gain** drives each split:

$$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

| Parameter | Value | Description |
|---|---|---|
| `criterion` | `'entropy'` | Splitting criterion: Information Gain |
| `random_state` | `0` | Random seed for reproducibility |

**Implementation:**

```python
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
```

**Decision Boundary:** Rectangular / step-shaped — axis-aligned splits create box-like regions.

---

### 7. Random Forest

**Theory:**
Random Forest is an **ensemble** of Decision Trees trained on random subsets of the training data (bagging) and random subsets of features at each split. Final prediction is by **majority vote** across all trees.

$$\hat{y} = \text{mode}\{h_1(x), h_2(x), \ldots, h_T(x)\}$$

where $h_t(x)$ is the prediction of the $t$-th tree. The ensemble reduces **variance** while keeping bias low compared to a single deep tree.

| Parameter | Value | Description |
|---|---|---|
| `n_estimators` | `10` | Number of trees in the forest |
| `criterion` | `'entropy'` | Splitting criterion: Information Gain |
| `random_state` | `0` | Random seed for reproducibility |

**Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
```

**Decision Boundary:** Non-linear, irregular — aggregation of many rectangular regions produces complex boundaries.

---

## Algorithm Comparison

| # | Algorithm | Class | Key Parameters | Decision Boundary | Needs Scaling | Interpretability | Handles Non-linearity | Best Use Case |
|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | `LogisticRegression` | `random_state=0` | Linear | Yes | High | No | Binary classification, linearly separable data |
| 2 | K-Nearest Neighbors | `KNeighborsClassifier` | `n_neighbors=5`, `metric='minkowski'`, `p=2` | Non-linear (irregular) | Yes | Medium | Yes | Small datasets, non-linear boundaries |
| 3 | SVM (Linear) | `SVC` | `kernel='linear'`, `random_state=0` | Linear | Yes | Medium | No | High-dimensional, clear margin of separation |
| 4 | Kernel SVM (RBF) | `SVC` | `kernel='rbf'`, `random_state=0` | Non-linear (smooth) | Yes | Low | Yes | Complex boundaries, medium-sized datasets |
| 5 | Naive Bayes | `GaussianNB` | *(none)* | Non-linear (Gaussian) | No | High | Yes | Text classification, fast baseline |
| 6 | Decision Tree | `DecisionTreeClassifier` | `criterion='entropy'`, `random_state=0` | Rectangular (step) | No | High | Yes | Explainable models, feature importance |
| 7 | Random Forest | `RandomForestClassifier` | `n_estimators=10`, `criterion='entropy'`, `random_state=0` | Non-linear (irregular) | No | Low | Yes | General-purpose, robust, high accuracy |

---

## Tech Stack

| Tool / Library | Version | Purpose |
|---|---|---|
| Python | 3.10.9 | Programming language |
| NumPy | latest | Numerical computation, array operations |
| Pandas | latest | Dataset loading and manipulation |
| Matplotlib | latest | Decision boundary and scatter plot visualizations |
| scikit-learn | latest | All ML models, preprocessing, evaluation metrics |
| Jupyter Notebook | latest | Interactive development environment |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nithinrajkore/Classification-Models.git
cd Classification-Models
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Or using a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Navigate to any algorithm folder and open the `.ipynb` file.

### 4. Run All Cells

In Jupyter: **Kernel → Restart & Run All**

Each notebook is fully self-contained and will:
1. Load `Social_Network_Ads.csv`
2. Preprocess the data (split + scale)
3. Train the classifier
4. Make a single prediction for `[30, 87000]`
5. Generate the confusion matrix and accuracy score
6. Plot decision boundaries for training and test sets

---

## Key Concepts Glossary

| Term | Definition |
|---|---|
| **Classification** | Supervised learning task that assigns discrete class labels to input samples |
| **Decision Boundary** | The surface in feature space that separates different class regions |
| **Feature Scaling** | Normalizing feature ranges so no single feature dominates (StandardScaler: zero mean, unit variance) |
| **Train/Test Split** | Partitioning data into a training set (to learn) and a test set (to evaluate generalization) |
| **Confusion Matrix** | 2×2 table showing True Positives, True Negatives, False Positives, False Negatives |
| **Accuracy** | Fraction of correctly classified samples: $(TP + TN) / (TP + TN + FP + FN)$ |
| **Entropy** | Measure of impurity / disorder in a node: $H = -\sum p_c \log_2 p_c$ |
| **Information Gain** | Reduction in entropy achieved by splitting on a feature |
| **Kernel Trick** | Implicitly projecting data to higher dimensions using a kernel function (avoids explicit transformation) |
| **Bagging** | Bootstrap Aggregating — training multiple models on random data subsets and averaging predictions |
| **Support Vectors** | Training samples closest to the decision hyperplane in SVM |
| **Margin** | Distance between the decision hyperplane and the nearest support vectors |
| **Overfitting** | When a model memorizes training data and performs poorly on unseen data |
| **Bias-Variance Tradeoff** | The tension between model simplicity (high bias) and sensitivity to training data (high variance) |

---

## References

- **`Classification_Pros_Cons.pdf`** — included in the repo root; summarizes the advantages and disadvantages of each classification algorithm covered here.
- [scikit-learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
