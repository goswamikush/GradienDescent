import numpy as np
import pandas as pd

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_gradient(x, y, w, b):
    dw, db = 0, 0
    m = x.shape[0]

    for i in range(m):
        dw += (w*x[i] + b - y[i]) * x[i] / m
        db += (w*x[i] + b - y[i]) / m

    return dw, db

def gradient_descent(x, y, initial_w, intial_b, alpha, compute_gradient, num_iters):
    w, b = initial_w, intial_b
    for i in range(num_iters):
        dw, db = compute_gradient(x, y, w, b)

        w = w - alpha * dw
        b = b - alpha * db

    return w, b

w_final, b_final = gradient_descent(x_train, y_train, 0, 0, 1.0e-2, compute_gradient, 10000)
print(w_final, b_final)
