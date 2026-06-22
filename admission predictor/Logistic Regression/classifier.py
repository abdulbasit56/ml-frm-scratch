import numpy as np
import pandas as pd
import copy
import math
import sys
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent
# 1. Find the folder one level up from this file
parent_folder = str(Path(__file__).resolve().parent.parent)

# 2. Add that parent folder to Python's search radar
sys.path.append(parent_folder)

from proj_utils import z_normalize


class LogisticRegression:
    
    def __init__(self):
        pass

    def compute_cost(self,X,y,w,b, lambda_ =1):
        total_cost = 0
        epsilon = 1e-15
        m,n = X.shape
        for i in range(m):
            z_i = np.dot(X[i], w) + b
            f_wb_i = 1 / (1+np.exp(-z_i) )
            f_wb_i = np.clip(f_wb_i, epsilon, 1 - epsilon) # this part is to prevent division by 0.
            total_cost += (-y[i] * np.log(f_wb_i)) - (1- y[i]) * np.log(1- f_wb_i)

        total_cost = total_cost / m
        reg_cost =0
        for j in range(n): # regularization step
            reg_cost += (w[j])**2
        reg_cost = reg_cost * (lambda_ / (2 * m))

        total_cost += reg_cost
        return total_cost
            
    

    def compute_gradient(self,X,y,w,b, lambda_ =1):
        m, n = X.shape
        dj_dw = np.zeros(n)
        dj_db =0
        for i in range(m):
            z_i = (np.dot(X[i], w)+ b) 
            err = (1 / (1+ np.exp(-z_i))) - y[i]
            for j in range(n):
                dj_dw[j]  =  dj_dw[j] + err * X[i][j]
            dj_db += err

        
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        for j in range(n):
            dj_dw[j] += w[j] * (lambda_/m)
        
        return dj_dw, dj_db


    def gradient_descent(self, X, y, w_ini, b_ini, aplha, num_iters):
        
        J_hist = []

        w = copy.deepcopy(w_ini)
        b = b_ini
        for i in range(num_iters):

            dj_dw, dj_db = self.compute_gradient(X, y, w, b)

            w = w - aplha * dj_dw
            b = b - aplha * dj_db

            if i < 10000:
                J_hist.append(self.compute_cost(X, y,w,b))
            if i% math.ceil(num_iters / 10) == 0:
                print (f"Iteration: {i: 4d} Cost: {J_hist[-1]: 8.4f}")

        return w,b, J_hist
    


def load_data(csv_path):

    #loading the data and creating np arrays
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Selects all rows, and columns from index 1 up to (but excluding) the last column
    x_train = df.iloc[:, 1:-1].to_numpy()
    y_train = df['pass'].to_numpy()

    return x_train, y_train


def train(csv_path, alpha=0.01, iters=1000):

    x_train, y_train = load_data(csv_path)
    x_train, mu, sigma = z_normalize(x_train)

    w_init = np.zeros(x_train.shape[1])
    b_init = 0

    model = LogisticRegression()
    w_final, b_final, J_hist = model.gradient_descent(x_train, y_train, w_init, b_init, alpha, iters)

    return w_final, b_final, mu, sigma, J_hist


def predict_new(features_raw, w, b, mu, sigma):

    features_raw = np.array(features_raw, dtype=float)
    features_scaled = (features_raw - mu) / sigma
    z = np.dot(features_scaled, w) + b
    pred = 1/ (1+ np.exp(-z))
    if (pred >= 0.5):
        return 1
    else:
        return 0



def save_model(w, b, mu, sigma, prefix='model'):

    np.save(MODEL_DIR/f'{prefix}_w.npy', w)
    np.save(MODEL_DIR/f'{prefix}_b.npy', np.array([b]))
    np.save(MODEL_DIR/f'{prefix}_mu.npy', mu)
    np.save(MODEL_DIR/f'{prefix}_sigma.npy', sigma)

def load_model(prefix='model'):
    w = np.load(MODEL_DIR/f'{prefix}_w.npy')
    b = np.load(MODEL_DIR/f'{prefix}_b.npy')[0]
    mu = np.load(MODEL_DIR/f'{prefix}_mu.npy')
    sigma = np.load(MODEL_DIR/f'{prefix}_sigma.npy')
    return w, b, mu, sigma


