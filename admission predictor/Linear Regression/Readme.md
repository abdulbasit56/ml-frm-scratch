# Graduate Admission Predictor — Linear Regression

> Part of the [ml-from-scratch](https://github.com/abdulbasit56/ml-from-scratch) repository · Course 1 of Andrew Ng's Machine Learning Specialization

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](YOUR_STREAMLIT_URL)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Built%20with-NumPy-013243?logo=numpy)](https://numpy.org/)

---

## Overview

A **multivariate linear regression model** built entirely from scratch using NumPy to predict a student's probability of graduate school admission (0–1 scale). No scikit-learn, every component including the cost function, gradient computation, and gradient descent loop is hand-implemented to match the theory taught in Andrew Ng's MLS Course 1.

The model is trained on the [Graduate Admissions dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) (500 samples, 7 features) and deployed as a live Streamlit web app.

---

## Results

| Metric | Value |
|---|---|
| R² Score | 0.82 |
| MAE | 0.042 |
| RMSE | 0.060 |
| Bias | ~0.000 |
| Within ±0.10 | 91.8% |

The model explains **82% of the variance** in admission probability. Bias of ~0 confirms gradient descent converged to a genuine minimum with no systematic over- or under-prediction.

---

## Dataset

**Graduate Admissions** — Mohan S Acharya, Kaggle

| Feature | Description | Range |
|---|---|---|
| GRE Score | Graduate Record Exam score | 260–340 |
| TOEFL Score | English proficiency score | 0–120 |
| University Rating | Institution prestige | 1–5 |
| SOP | Statement of purpose strength | 1.0–5.0 |
| LOR | Letter of recommendation strength | 1.0–5.0 |
| CGPA | Undergraduate GPA | 0–10 |
| Research | Research experience (binary) | 0 / 1 |

Target: **Chance of Admit** (continuous, 0–1)

---

## Implementation

All three core components are implemented from scratch in pure NumPy with no ML library dependencies:

### Cost Function — Mean Squared Error
```
J(w,b) = (1/2m) Σ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

### Gradient Computation
```
∂J/∂wⱼ = (1/m) Σ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾
∂J/∂b  = (1/m) Σ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾)
```

### Gradient Descent
```
wⱼ := wⱼ - α · ∂J/∂wⱼ
b  := b  - α · ∂J/∂b
```

Features are normalized using **Z-score normalization** before training to ensure gradient descent converges efficiently across features with different scales.

**Hyperparameters used:**

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.01 |
| Iterations | 1000 |
| Feature scaling | Z-score normalization |

---

## File Structure

```
Linear Regression/
├── regressor.py            # LinearRegression class + train/predict/save/load functions
├── train_save.py           # Run once to train and persist model weights
├── eda.ipynb               # Exploratory data analysis notebook
├── admission_model_w.npy   # Trained weight vector
├── admission_model_b.npy   # Trained bias
├── admission_model_mu.npy  # Feature means (for normalization at inference)
└── admission_model_sigma.npy  # Feature std devs (for normalization at inference)
```

---

## How to Run

**1. Train the model (run once):**
```bash
cd "Linear Regression"
python train_save.py
```

**2. Run the Streamlit app from the project root:**
```bash
streamlit run app.py
```

**3. Or import directly in your own code:**
```python
from regressor import load_model, predict_new

w, b, mu, sigma = load_model(prefix='admission_model')
chance = predict_new([320, 105, 3, 3.5, 3.5, 8.5, 1], w, b, mu, sigma)
print(f"Admission chance: {chance:.2%}")
```

---

## Key Learnings

- Implemented the full gradient descent loop from scratch, observing how cost evolves per iteration and verifying convergence by testing multiple learning rates (0.001 → 0.1)
- Discovered that Z-score normalization doesn't change the shape of feature distributions — it rescales axes without altering relationships, which is its actual purpose (enabling consistent gradient steps across features)
- Identified that dropping 3 features (University Rating, SOP, LOR) was artificially capping R² — adding them back improved performance, demonstrating that feature completeness matters more than polynomial complexity when signal is missing
- Confirmed the model reached the true cost minimum (J = 0.0018) by verifying the same final cost across all tested learning rates

---

## Dependencies

```
numpy
pandas
streamlit
```
