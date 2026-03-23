# ML Data Preprocessing — Étude comparative du scaling

A mini-project exploring the impact of feature scaling on logistic regression performance, using the **Breast Cancer Wisconsin** dataset from scikit-learn.

---

## Objective

Demonstrate how different scaling strategies affect:
- Model **accuracy** (5-fold cross-validation)
- **Convergence** of the lbfgs optimizer
- Result **stability** (standard deviation across folds)

---

## 📁 Project Structure

```
ml-data-preprocessing/
├── preprocessing-miniprojet.ipynb   # Main notebook
├── figures/
│   ├── fig1_boxplot_scaling.png         # Feature distribution before/after scaling
│   ├── fig2_outliers_std_vs_robust.png  # Outlier sensitivity comparison
│   └── fig3_comparaison_accuracy.png    # Accuracy across the 4 scenarios
└── test-1/                              # Early experiment outputs
```

---

## Experiment — 4 Scaling Scenarios

All scenarios use `LogisticRegression(max_iter=1000)` evaluated with 5-fold cross-validation.

| Scenario | Scaler | Mean Accuracy | Std |
|---|---|---|---|
| S1 | None *(no scaling)* | 0.9543 | 0.0128 |
| S2 | `StandardScaler` | **0.9807** | 0.0065 |
| S3 | `MinMaxScaler` | 0.9613 | 0.0042 |
| S4 | `RobustScaler` | 0.9789 | 0.0089 |

### Key findings

- **S1 (no scaling)** triggers 5 `ConvergenceWarning`s — lbfgs fails to converge due to large feature scale differences. Accuracy is degraded and less stable.
- **S2 (StandardScaler)** achieves the best mean accuracy and a low std. Ideal when data has no significant outliers.
- **S3 (MinMaxScaler)** is the most stable (lowest std) but sensitive to outliers, which compress useful variance.
- **S4 (RobustScaler)** is a strong alternative when outliers are present — uses median/IQR instead of mean/std.

---

## 🗂️ Dataset

**Breast Cancer Wisconsin** — available directly via `sklearn.datasets.load_breast_cancer()`

- 569 observations, 30 features
- Binary classification: `0` = malignant (212), `1` = benign (357)

---

## ⚙️ Setup

**Requirements:** Python 3.x, scikit-learn, numpy, matplotlib

```bash
pip install scikit-learn numpy matplotlib
```