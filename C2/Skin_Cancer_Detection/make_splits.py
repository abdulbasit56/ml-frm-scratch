import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 42 

df = pd.read_csv("metadata_cleaned.csv")
X = np.load("images_greyscale.npy")
y = df['diagnostic'].to_numpy()

indices = np.arange(len(df))

X_train, X_, y_train, y_, idx_train, idx_ = train_test_split(
    X, y, indices, test_size=0.30, random_state=RANDOM_STATE
)
X_cv, X_test, y_cv, y_test, idx_cv, idx_test = train_test_split(
    X_, y_, idx_, test_size=0.50, random_state=RANDOM_STATE
)

feature_cols = [c for c in df.columns if c not in ['img_id', 'diagnostic']]

X_tab_train = df.iloc[idx_train][feature_cols]
X_tab_cv    = df.iloc[idx_cv][feature_cols]
X_tab_test  = df.iloc[idx_test][feature_cols]


os.makedirs("splits", exist_ok=True)

# images + labels (npy — arrays)
np.save("splits/X_train.npy", X_train)
np.save("splits/X_cv.npy", X_cv)
np.save("splits/X_test.npy", X_test)

# labels are common to both
np.save("splits/y_train.npy", y_train)
np.save("splits/y_cv.npy", y_cv)
np.save("splits/y_test.npy", y_test)

#tabular splits 
X_tab_train.to_csv("splits/X_tab_train.csv", index=False)
X_tab_cv.to_csv("splits/X_tab_cv.csv", index=False)
X_tab_test.to_csv("splits/X_tab_test.csv", index=False)

print("Splits saved. Row counts:", len(y_train), len(y_cv), len(y_test))