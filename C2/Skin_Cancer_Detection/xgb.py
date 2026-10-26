import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
RANDOM_STATE = 42


# df = pd.read_csv("metadata_cleaned.csv")
# df = df.set_index('img_id')
# print(df.shape)
# var = [x for x in df.columns if x not in 'diagnostic']

import pandas as pd
import numpy as np

X_tab_train = pd.read_csv("splits/X_tab_train.csv")
X_tab_cv    = pd.read_csv("splits/X_tab_cv.csv")
X_tab_test  = pd.read_csv("splits/X_tab_test.csv")
y_train     = np.load("splits/y_train.npy")
y_cv        = np.load("splits/y_cv.npy")
y_test      = np.load("splits/y_test.npy")



xgb_model = XGBClassifier(n_estimators = 300, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE, early_stopping_rounds = 50)
xgb_model.fit(X_tab_train,y_train, eval_set = [(X_tab_cv,y_cv)])

# print(xgb_model.best_iteration)

print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_tab_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_tab_cv),y_cv):.4f}")
# print(accuracy_score(xgb_model.predict(X_tab_test), y_test))

cv_probs = xgb_model.predict_proba(X_tab_cv)[:, 1]
cv_preds = (cv_probs > 0.5).astype(int)

test_probs = xgb_model.predict_proba(X_tab_test)[:, 1]
test_preds = (test_probs > 0.5).astype(int)

# print("AUC:", roc_auc_score(y_cv, test_probs))
# print(classification_report(y_cv, test_preds))
print(confusion_matrix(y_cv, cv_preds))
print (f"Accuracy score: CV = {accuracy_score(y_cv, cv_preds)} , Test = {accuracy_score(y_test, test_preds)}")

for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    t_preds = (cv_probs > t).astype(int)
    cm = confusion_matrix(y_cv, t_preds)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(f"threshold={t}: recall={recall:.3f}, precision={precision:.3f}, missed cases={fn}, false alarms={fp}")


np.save("splits/tree_probs_cv.npy", cv_probs)
np.save("splits/tree_probs_test.npy", test_probs)

# saved for app.py
import os
os.makedirs("models", exist_ok=True)
xgb_model.save_model("models/xgb_model.json")

import matplotlib.pyplot as plt
plt.hist(cv_probs, bins=30)
plt.xlabel("Predicted probability")
plt.show()