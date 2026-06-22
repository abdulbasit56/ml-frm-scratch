# Student Success Predictor

> Course 1 Project · [Andrew Ng's Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) · Built from scratch with NumPy

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](YOUR_STREAMLIT_URL)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Built%20with-NumPy-013243?logo=numpy)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?logo=streamlit)](YOUR_STREAMLIT_URL)

---

## What This Is

Two supervised learning models built **entirely from scratch** — no scikit-learn, no black boxes. Every cost function, gradient, and weight update is hand-coded in NumPy to match the mathematical theory from Andrew Ng's MLS Course 1, then deployed together as a live two-tab web app.

| Model | Type | Task | Result |
|---|---|---|---|
| Graduate Admission Predictor | Linear Regression | Predict admission probability (0–1) | R² = 0.82, MAE = 0.042 |
| Student Pass/Fail Classifier | Logistic Regression | Classify pass or fail (binary) | ~95%+ accuracy |

**[→ Try the live app](YOUR_STREAMLIT_URL)**

---

## Project Structure

```
student-success-predictor/
├── app.py                          # Streamlit app — two tabs, one for each model
├── requirements.txt
├── proj_utils.py                   # Shared z-score normalization utility
├── Data/
│   ├── Admission_Predict_Ver1.1.csv
│   └── pass-fail.csv
├── Linear Regression/
│   ├── regressor.py                # LinearRegression class, train, predict, save, load
│   ├── train_save.py               # Run once to train and persist weights
│   ├── eda.ipynb                   # Exploratory data analysis
│   └── README.md                   # ← Implementation details for this model
└── Logistic Regression/
    ├── classifier.py               # LogisticRegression class, train, predict, save, load
    ├── train_save.py
    ├── visualize.py                # Decision boundary + probability distribution plots
    ├── decision_boundary.png
    └── README.md                   # ← Implementation details for this model
```

---

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train both models (run once)**
```bash
cd "Linear Regression" && python train_save.py && cd ..
cd "Logistic Regression" && python train_save.py && cd ..
```

**3. Launch the app**
```bash
streamlit run app.py
```

---

## What's Under the Hood

Both models share the same architecture pattern — a Python class with three core methods (`compute_cost`, `compute_gradient`, `gradient_descent`), wrapped in standalone `train()`, `predict_new()`, `save_model()`, and `load_model()` functions so they're cleanly importable by the Streamlit app.

Model weights are persisted as `.npy` files so the deployed app loads pre-trained parameters instantly without re-running gradient descent on every user interaction.

For implementation details, math, and key learnings from each model:

- [Linear Regression README](Linear%20Regression/README.md)
- [Logistic Regression README](Logistic%20Regression/README.md)

---

## Why From Scratch

Using scikit-learn's `LinearRegression().fit()` would take two lines. The point of this project is to understand what those two lines actually do — how gradient descent navigates a cost surface, why feature scaling matters for convergence, why MSE breaks logistic regression, and what regularization is actually solving. That understanding is what the implementation demonstrates.

---

## Part of

This project is part of the [`ml-from-scratch`](https://github.com/YOUR_USERNAME/ml-from-scratch) repository, which documents my full implementation of Andrew Ng's Machine Learning Specialization — one project per concept, built from the ground up.

---

## Dependencies

```
numpy
pandas
streamlit
matplotlib
```
