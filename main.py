from typing import Any

import pyKRLST
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import norm
from sklearn.model_selection import train_test_split


def f(x: np.ndarray) -> np.ndarray:
    return np.sinc(x)


def get_training_data(size: int, left: float, right: float) -> [np.ndarray, np.ndarray]:
    data = np.linspace(left, right, size)
    X, _ = train_test_split(data, test_size=0.95)
    return np.atleast_2d(X).T, np.atleast_2d(data).T


def evaluate(kernel, m, X, x, noise: np.ndarray = None) -> [Any, np.ndarray, np.ndarray, float, float]:
    c, l = 1e-5, 0.999
    mode = "B2P"

    krlst = pyKRLST.KRLST(kernel=RBF(kernel), l=l, c=c, M=m, forgetmode=mode)

    y = f(X).ravel()
    if noise is not None:
        y = y + noise

    for t, a, b in zip(np.arange(10), X, y):
        krlst.observe(a, b, t)

    y_pred, y_std = krlst.predict(x)

    mae = MAE(y_pred, f(x))
    rmse = math.sqrt(MSE(f(x), y_pred))
    return krlst, y_pred, y_std, rmse, mae


def evaluate_vector(X: np.ndarray, x: np.ndarray, kernel, isNoisy: bool = False) -> [Any, np.ndarray, np.ndarray, float, float, float]:
    size, index, min_rmse = 50, 0, 100
    models, preds, stds, rmses, maes = [], [], [], [], []
    noise = None

    if isNoisy:
        noise = norm.rvs(scale=0.1, size=len(X))

    for m in range(size):
        results = evaluate(kernel, m, X, x, noise)
        models.append(results[0])
        preds.append(results[1])
        stds.append(results[2])
        rmses.append(results[3])
        maes.append(results[4])

        if min_rmse > rmses[m]:
            min_rmse = rmses[m]
            index = m

    return models[index], preds[index], stds[index], rmses[index], maes[index], index


def evaluate_kernel(X, x) -> [Any, np.ndarray, np.ndarray, float, float, float, float]:
    models, preds, stds, rmses, maes, ms = [], [], [], [], [], []

    index, min_rmse, opt_kernel = 0, 100, 0
    for kernel in np.linspace(0.001, 1.1, 10):
        results = evaluate_vector(X, x, kernel)
        models.append(results[0])
        preds.append(results[1])
        stds.append(results[2])
        rmses.append(results[3])
        maes.append(results[4])
        ms.append(results[5])

        if min_rmse > rmses[len(rmses) - 1]:
            index = len(rmses) - 1
            min_rmse = rmses[index]
            opt_kernel = kernel

    print(rmses)
    draw_graph(rmses, "Dependence between RMSE and kernel parameters", np.linspace(0.001, 1.1, 10))
    return models[index], preds[index], stds[index], rmses[index], maes[index], ms[index], opt_kernel


def draw_graph(values: list, title: str, x_label: str, y_label: str, data: np.ndarray = None) -> None:
    if data is not None:
        plt.plot(data, values)
    else:
        plt.plot(values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def draw_krlst(X, x, y_pred, y_std, krlst: Any, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x, f(x), 'r:', label=r'$f(x) = sinc(x)$')
    plt.plot(krlst.Xb, krlst.mu, 'k*', markersize=20, label="Dictionary Elements")
    plt.plot(X, f(X), 'r.', markersize=15, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * y_std,
                             (y_pred + 1.9600 * y_std)[::-1]]),
             alpha=.25, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-2, 2)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


def main() -> None:
    epoch_count, size = 50, 200
    X, x = get_training_data(200, -2, 2)

    # KLST-regression without noise
    model, y_pred, y_std, rmse, mae, m, kernel = evaluate_kernel(X, x)
    print("KLST-regression without noise")
    print(f"Kernel param = {kernel}")
    print(f"Support vectors = {m}")
    print(f"MAE = {mae}, RMSE = {rmse}")
    draw_krlst(X, x, y_pred, y_std, model, "KLST-regression without noise")

    # KLST-regression with noise
    models, preds, stds, rmses, maes, ms = [], [], [], [], [], []

    for epoch in range(epoch_count):
        results = evaluate_vector(X, x, kernel, True)
        models.append(results[0])
        preds.append(results[1])
        stds.append(results[2])
        rmses.append(results[3])
        maes.append(results[4])
        ms.append(results[5])

    draw_graph(rmses, "RMSE values with using a noise")

    index = np.array(rmses).argmin()
    print("\nKLST-regression with noise and minimum RMSE")
    print(f"Kernel param = {kernel}")
    print(f"Support vectors = {ms[index]}")
    print(f"MAE = {maes[index]}, RMSE = {rmses[index]}")
    print(f"\nMean RMSE = {np.mean(rmses)}")
    draw_krlst(X, x, preds[index], stds[index], models[index], "KLST-regression with noise and minimum RMSE")


main()