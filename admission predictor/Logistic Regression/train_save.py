from classifier import train, save_model

w_final, b_final, mu, sigma, J_hist = train("../Data/pass-fail.csv", alpha=0.5, iters=2000)

save_model(w_final, b_final, mu, sigma, prefix='student_model')

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final}")