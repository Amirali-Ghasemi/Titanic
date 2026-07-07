# Titanic - Advanced Ensemble Learning Framework

<p align="center">
  <img src="assets/titanic.jpg" alt="Titanic Ship" width="900">
</p>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Pipeline & Methodology](#pipeline--methodology)
- [Algorithms & Optimization](#algorithms--optimization)
- [Installation & Reproduction](#installation--reproduction)


An end-to-end Machine Learning pipeline utilizing sophisticated feature engineering, state-of-the-art tree-based algorithms, hyperparameter optimization, and custom Ensemble methods (Voting & Stacking Classifiers) to predict survival rates for the classic Kaggle Titanic competition.

## Project Overview

This repository showcases a structured, production-ready machine learning framework. Developed as a milestone project, it incorporates rigorous exploratory data analysis (EDA), custom pre-processing pipelines, robust cross-validation strategies, and advanced ensemble models to maximize predictive performance on the historical Titanic passenger dataset.

## Project Architecture

The project structure is organized to maintain a clean separation between raw data, research notebooks, and serialized models:

- `data/`: Contains the training and testing datasets.
- `notebooks/`: Comprehensive research and training workflow.
- `models/`: Serialized final ensemble model (`.pkl`).
- `requirements.txt`: Environment dependency specifications.

### Technologies Used

- Kaggle
- Python 3.11
- Scikit-Learn
- XGBoost

- `submission.csv`: Final predictions generated for Kaggle.

## Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA)

- **Feature Correlation:** Investigated relationships between demographic variables (Gender, Age, Class) and the `Survived` target.
- **Data Quality:** Addressed skewness in `Fare` distributions and managed significant missing value clusters.
- **Contextual Insights:** Analyzed the influence of family size and socioeconomic indicators on survival probability.

### 2. Feature Engineering & Extraction

- **Title Extraction:** Isolated titles (e.g., `Mr`, `Mrs`, `Miss`, `Master`) and condensed rare titles (e.g., `Lady`, `Capt`, `Dr`) into a unified `Rare` category to reduce cardinality.
- **Family Dimensions:** Computed `FamilySize = SibSp + Parch + 1` and created an `IsAlone` boolean feature to measure passenger independence.
- **Deck Extraction:** Engineered structural deck assignments (e.g., `A`, `B`, `C`) derived from raw `Cabin` strings.
- **Financial Metrics:** Calculated `FarePerPerson` to standardize costs.
- **Intelligent Imputation:** Applied median-based imputation for numerical data and mode-based imputation for categorical features.

### 3. Pre-Processing Pipeline

To ensure robust performance and prevent data leakage, we utilized Scikit-Learn’s `ColumnTransformer` and `Pipeline` architecture:

- **Numerical Features:** Standardized via `StandardScaler`.
- **Categorical Features:** Encoded via `OneHotEncoder` with `handle_unknown="ignore"` to maintain stability on unseen test sets.

## Algorithms & Optimization

We evaluated models through a rigorous 5-Fold Cross-Validation approach. Hyperparameters were fine-tuned using `GridSearchCV`.

| Model Name | Cross-Validation Score (Accuracy) |
|-----------|------------------------------------|
| Voting Classifier (Soft Ensemble) | Top Performer |
| Stacking Classifier | High Performer |
| XGBoost Classifier (Tuned) | High Performer |
| Random Forest Classifier (Tuned) | Solid Performer |
| Logistic Regression (Baseline) | Baseline |

### Custom Ensemble Architectures

- **Voting Classifier:** Integrates predictions from Tuned Random Forest, Tuned XGBoost, and Logistic Regression using a soft-voting scheme, which averages the predicted probabilities.
- **Stacking Classifier:** Implements a meta-learning approach where the outputs of base models (Random Forest and XGBoost) serve as features for a final meta-regressor (Logistic Regression), effectively optimizing meta-weights for the final decision.

## Installation & Reproduction

### Prerequisites

- Python 3.11 or higher

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/titanic-survivor-prediction.git
cd titanic-survivor-prediction
