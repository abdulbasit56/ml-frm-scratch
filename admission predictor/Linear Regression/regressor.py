import numpy as np
import pandas as pd
import copy
import math
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from proj_utils import z_normalize



class LinearRegression:
    
    def __init__(self):
        pass

    def compute_cost(self,X,y,w,b):
        total_cost = 0
        m = X.shape[0]
        for i in range(m):
            f_wb_i = np.dot(X[i], w) + b
            cost = (f_wb_i - y[i])**2
            total_cost += cost
        total_cost = total_cost / (2 * m)
        return total_cost
            
    

    def compute_gradient(self,X,y,w,b):
        m, n = X.shape
        dj_dw = np.zeros(n)
        dj_db =0
        for i in range(m):
            err = (np.dot(X[i], w)+ b) - y[i]
            for j in range(n):
                dj_dw[j]  =  dj_dw[j] + err * X[i][j]
            dj_db += err
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        
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
    df = pd.read_csv("../Data/Admission_Predict_Ver1.1.csv")
    df.columns = df.columns.str.strip()

    # Selects all rows, and columns from index 1 up to (but excluding) the last column
    x_train = df.iloc[:, 1:-1].to_numpy()
    y_train = df['Chance of Admit'].to_numpy()
    
    return x_train, y_train


def train(csv_path, alpha=0.01, iters=1000):

    x_train, y_train = load_data(csv_path)
    x_train, mu, sigma = z_normalize(x_train)

    w_init = np.zeros(x_train.shape[1])
    b_init = 0

    model = LinearRegression()
    w_final, b_final, J_hist = model.gradient_descent(x_train, y_train, w_init, b_init, alpha, iters)

    return w_final, b_final, mu, sigma, J_hist


def predict_new(features_raw, w, b, mu, sigma):

    features_raw = np.array(features_raw, dtype=float)
    features_scaled = (features_raw - mu) / sigma
    return np.dot(features_scaled, w) + b


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

    
