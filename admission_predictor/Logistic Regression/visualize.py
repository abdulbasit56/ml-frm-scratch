import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from proj_utils import z_normalize
from classifier import load_model, load_data

# --- load data and model ---
x_train, y_train = load_data('../Data/pass-fail.csv')
x_train_norm, mu, sigma = z_normalize(x_train)
w, b, mu, sigma = load_model(prefix='admission_model')

# --- compute predictions and accuracy ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

probs = sigmoid(x_train_norm @ w + b)
preds = (probs >= 0.5).astype(int)
accuracy = np.mean(preds == y_train)

# --- pick two strongest features for the 2D boundary plot ---
# attendance_pct (col 0) vs study_hours_per_week (col 3)
feat_x_idx, feat_y_idx = 0, 3
feat_x_name, feat_y_name = 'Attendance (%)', 'Study Hours / Week'

feat_x_raw = x_train[:, feat_x_idx]
feat_y_raw = x_train[:, feat_y_idx]

# --- build mesh grid in raw feature space ---
x_min, x_max = feat_x_raw.min() - 5, feat_x_raw.max() + 5
y_min, y_max = feat_y_raw.min() - 1, feat_y_raw.max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

# fix the other two features at their mean (homework_pct, midterm_score)
mean_homework = x_train[:, 1].mean()
mean_midterm  = x_train[:, 2].mean()

grid_points = np.column_stack([
    xx.ravel(),
    np.full(xx.size, mean_homework),
    np.full(xx.size, mean_midterm),
    yy.ravel()
])

grid_norm = (grid_points - mu) / sigma
grid_probs = sigmoid(grid_norm @ w + b).reshape(xx.shape)

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Logistic Regression — Student Pass/Fail Classifier', fontsize=13, fontweight='bold', y=1.01)

# PLOT 1: decision boundary
ax = axes[0]
ax.contourf(xx, yy, grid_probs, levels=[0, 0.5, 1],
            colors=['#f28b82', '#86c5a0'], alpha=0.35)
ax.contour(xx, yy, grid_probs, levels=[0.5],
           colors=['#333333'], linewidths=1.8, linestyles='--')

pass_mask = y_train == 1
fail_mask = y_train == 0
ax.scatter(feat_x_raw[pass_mask], feat_y_raw[pass_mask],
           color='#1a7a4a', edgecolors='white', s=55, alpha=0.8, label='Pass (1)', zorder=3)
ax.scatter(feat_x_raw[fail_mask], feat_y_raw[fail_mask],
           color='#c0392b', edgecolors='white', s=55, alpha=0.8, label='Fail (0)', zorder=3)

ax.set_xlabel(feat_x_name, fontsize=11)
ax.set_ylabel(feat_y_name, fontsize=11)
ax.set_title(f'Decision Boundary\n(other features fixed at mean)', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.5)

boundary_patch = mpatches.Patch(facecolor='none', edgecolor='#333333',
                                linestyle='--', label='Decision boundary (p=0.5)')
ax.legend(handles=[
    mpatches.Patch(color='#1a7a4a', label='Pass (1)'),
    mpatches.Patch(color='#c0392b', label='Fail (0)'),
    boundary_patch
], fontsize=9, loc='upper left')

# PLOT 2: probability distribution
ax2 = axes[1]
ax2.hist(probs[y_train == 1], bins=25, color='#1a7a4a', alpha=0.6,
         label=f'Pass (n={pass_mask.sum()})', edgecolor='white')
ax2.hist(probs[y_train == 0], bins=25, color='#c0392b', alpha=0.6,
         label=f'Fail (n={fail_mask.sum()})', edgecolor='white')
ax2.axvline(0.5, color='#333333', linestyle='--', linewidth=1.8, label='Threshold (0.5)')
ax2.set_xlabel('Predicted Probability of Pass', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title(f'Predicted Probability Distribution\nAccuracy: {accuracy*100:.1f}%', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Accuracy: {accuracy*100:.2f}%")