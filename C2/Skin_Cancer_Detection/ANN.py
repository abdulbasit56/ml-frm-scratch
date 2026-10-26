import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np 
import pandas as pd 
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RandomFlip, RandomRotation, RandomZoom, Flatten
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
RANDOM_STATE = 42


import numpy as np

X_train = np.load("splits/X_train.npy")
X_cv    = np.load("splits/X_cv.npy")
X_test  = np.load("splits/X_test.npy")
y_train = np.load("splits/y_train.npy")
y_cv    = np.load("splits/y_cv.npy")
y_test  = np.load("splits/y_test.npy")



def eval_err(y, y_hat):

    m = len(y)
    neg = 0

    for i in range(m):
        if y[i] != y_hat[i]:
            neg += 1
        
    err = neg/m
    return err



model = Sequential(
    [
    Dense(128, activation='relu', kernel_regularizer= tf.keras.regularizers.l2(0.0001), name="L1"),
    Dense(64, activation='relu', kernel_regularizer= tf.keras.regularizers.l2(0.0001), name= "L2"),
    Dense(1, activation='sigmoid', name= "L3")
    ]
    , name= "Simple_ANN"
 )


model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer= tf.keras.optimizers.Adam(0.001),
    metrics = ['accuracy']
)

es = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data = (X_cv, y_cv),
    epochs= 50,
    callbacks = [es]


)


y_pred_probs = model.predict(X_train)
y_pred = np.where(y_pred_probs >= 0.5,1,0)
print(eval_err(y_train, y_pred))

acc = accuracy_score(y_train, y_pred)
print(f"Calculated Test Accuracy: {acc * 100:.2f}%")



from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

ann_val_probs = model.predict(X_cv).flatten()   # ANN's sigmoid outputs on validation set
ann_val_preds = np.where(ann_val_probs > 0.5,1, 0)

print("AUC:", roc_auc_score(y_cv, ann_val_probs))
print(f"Accuracy (CV){accuracy_score(y_cv,ann_val_preds )}")
print(classification_report(y_cv, ann_val_preds))
print(confusion_matrix(y_cv, ann_val_preds))

np.save("splits/ann_probs_cv.npy", ann_val_probs)
np.save("splits/ann_probs_test.npy", model.predict(X_test).flatten())

# saved for app.py
os.makedirs("models", exist_ok=True)
model.save("models/ann_model.keras")

ann_test_prob = model.predict(X_test)
ann_test_preds = np.where(ann_test_prob>= 0.5, 1,0)

print(f"Test Data Accuracy: {accuracy_score(y_test, ann_test_preds)}")



# import matplotlib.pyplot as plt
# plt.hist(ann_val_probs, bins=30)
# plt.xlabel("Predicted probability")
# plt.title("ANN probability distribution")
# plt.show()

















