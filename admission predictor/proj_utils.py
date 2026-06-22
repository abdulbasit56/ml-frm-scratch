import numpy as np

def z_normalize(X):

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    x_norm = (X - mu) / sigma

    return x_norm, mu, sigma


def evaluate_regression(self, y_true, y_pred):
        m = len(y_true)
        
        # 1. Mean Absolute Error
        mae = np.sum(np.abs(y_pred - y_true)) / m
        
        # 2. Mean Squared Error
        mse = np.sum((y_pred - y_true) ** 2) / m
        
        # 3. Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # 4. R-squared (Coefficient of Determination)
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print("--- Model Performance Metrics ---")
        print(f"Mean Absolute Error (MAE):      {mae:.4f}")
        print(f"Mean Squared Error (MSE):       {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared Score (R2):           {r2:.4f} ({r2 * 100:.2f}%)")
        
        return mae, mse, rmse, r2