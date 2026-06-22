from regressor import train, save_model

w_final, b_final, mu, sigma, J_hist = train("../Data/Admission_Predict_Ver1.1.csv", alpha=0.01, iters=1000)

save_model(w_final, b_final, mu, sigma, prefix='admission_model')

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final}")