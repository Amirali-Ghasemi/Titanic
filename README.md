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
- [Streamlit Live Demo](#streamlit-live-demo)
- [Installation & Reproduction](#installation--reproduction)
- [Screenshots](#screenshots)
- [Notes](#notes)

---

An end-to-end Machine Learning pipeline that combines **feature engineering**, **tree-based algorithms**, **hyperparameter optimization**, and **custom ensemble methods** (Voting & Stacking Classifiers) to predict survival rates for the classic Kaggle Titanic competition.

Besides the offline pipeline, the project now includes a **deployed Streamlit app** that makes the model accessible as an interactive web demo.

---

## Project Overview

This repository showcases a structured, production-oriented machine learning workflow.
It was developed as a milestone project and includes:

- Rigorous **Exploratory Data Analysis (EDA)**
- Custom **pre-processing pipelines**
- Robust **cross-validation strategies**
- Advanced **ensemble models** (soft Voting & Stacking)
- A **Streamlit dashboard** for real-time inference

The goal is to maximize predictive performance on the historical Titanic passenger dataset while keeping the codebase clear, reproducible, and extensible.

---

## Project Architecture

The project structure is organized to maintain a clean separation between raw data, research notebooks, serialized models, and the web app:

- `data/`
  Contains the training and testing datasets.

- `notebooks/`
  Jupyter notebooks covering EDA, feature engineering, model training, and evaluation.

- `models/`
  Serialized final ensemble model (`.pkl`) used both in offline experiments and by the Streamlit app.

- `assets/`
  Static assets such as the Titanic cover image and Streamlit app screenshots.

- `app.py`
  Streamlit application that loads the trained model and exposes an interactive UI for survival prediction.

- `requirements.txt`
  Environment dependency specifications.

- `submission.csv`
  Final predictions generated for Kaggle.

### Technologies Used

- Kaggle
- Python 3.11
- Scikit-Learn
- XGBoost
- Streamlit

---

## Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA)

Key analysis steps include:

- **Feature Correlation**
  Investigating relationships between demographic variables (`Sex`, `Age`, `Pclass`, etc.) and the `Survived` target.

- **Data Quality**
  Handling skewed distributions (e.g., `Fare`) and addressing clusters of missing values.

- **Contextual Insights**
  Studying the impact of family size and socioeconomic indicators on survival probability.

---

### 2. Feature Engineering & Extraction

To extract more signal from the raw data, several engineered features are introduced:

- **Title Extraction**
  Parsing passenger names to isolate titles (e.g., `Mr`, `Mrs`, `Miss`, `Master`) and collapsing rare titles (e.g., `Lady`, `Capt`, `Dr`) into a unified `Rare` category.

- **Family Dimensions**
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone` (boolean feature) to capture whether a passenger traveled alone.

- **Deck Extraction**
  Deriving structural deck assignments (e.g., `A`, `B`, `C`) from raw `Cabin` strings.

- **Financial Metrics**
  `FarePerPerson` to normalize ticket cost per individual.

- **Intelligent Imputation**
  - Median-based imputation for numerical features
  - Mode-based imputation for categorical features

---

### 3. Pre-Processing Pipeline

To prevent data leakage and keep transformations consistent, the project relies on Scikit-Learn’s `ColumnTransformer` and `Pipeline`:

- **Numerical Features**
  Standardized via `StandardScaler`.

- **Categorical Features**
  Encoded via `OneHotEncoder` with `handle_unknown="ignore"` to ensure robustness on unseen data.

All preprocessing steps are encapsulated inside the pipeline so that training, validation, and inference share the exact same transformations.

---

## Algorithms & Optimization

Models are evaluated with a **5-Fold Cross-Validation** strategy, and hyperparameters are tuned using `GridSearchCV`.

| Model Name                        | Cross-Validation Performance         |
|-----------------------------------|--------------------------------------|
| Voting Classifier (Soft Ensemble) | Top Performer                        |
| Stacking Classifier               | High Performer                       |
| XGBoost Classifier (Tuned)        | High Performer                       |
| Random Forest Classifier (Tuned)  | Solid Performer                      |
| Logistic Regression (Baseline)    | Baseline / Reference Model           |

### Custom Ensemble Architectures

- **Voting Classifier (Soft Voting)**
  Combines tuned Random Forest, tuned XGBoost, and Logistic Regression by averaging predicted probabilities for more stable decisions.

- **Stacking Classifier**
  Uses the outputs of base learners (Random Forest, XGBoost) as features for a meta-learner (Logistic Regression), effectively learning optimal meta-weights for the final prediction.

---

## Streamlit Live Demo

The trained ensemble model is exposed via a **Streamlit web application**:

- Interactive input controls for key passenger features
- Real-time prediction of survival probability
- Simple, clean UI suitable for demonstrations and portfolio use

You can try the live demo here:

> 🔗 **Live App:**
> https://amirali-ghasemi-titanic-app-3nm9aa.streamlit.app/

> 💻 **Source Code (this repository):**
> https://github.com/Amirali-Ghasemi/Titanic

---

## Installation & Reproduction

### Prerequisites

- Python 3.11 or higher
- `pip` (or `conda`) for package management

### 1. Clone the Repository
```bash
git clone https://github.com/Amirali-Ghasemi/Titanic.git
cd Titanic
