import pyKRLST
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from typing import Any


def f(x: np.ndarray) -> np.ndarray:
    return np.sinc(x)


def get_training_data(size: int, left: float, right: float, train_size) -> [np.ndarray, np.ndarray]:
    data = np.linspace(left, right, size)
    X, _ = train_test_split(data, train_size=train_size)
    return np.atleast_2d(X).T, np.atleast_2d(data).T


def evaluate(kernel: Any, m: int, l: float, X: np.ndarray, x: np.ndarray, noise: np.ndarray = None) ->\
        [Any, np.ndarray, np.ndarray, float, float, float]:
    c = 1e-5
    mode = "B2P"

    krlst = pyKRLST.KRLST(kernel=RBF(kernel), l=l, c=c, M=m, forgetmode=mode)

    y = f(X).ravel()
    if noise is not None:
        y = y + noise

    for t, a, b in zip(np.arange(10), X, y):
        krlst.observe(a, b, t)

    y_pred, _ = krlst.predict(x)

    mae = MAE(y_pred, f(x))
    rmse = math.sqrt(MSE(f(x), y_pred))
    return krlst, y_pred, y, rmse, mae, krlst.m


def evaluate_vector(X: np.ndarray, x: np.ndarray, kernel: Any, l: float, isNoisy: bool = False) ->\
        [Any, np.ndarray, np.ndarray, float, float, float]:
    max_vector_count, index, min_rmse = 15, 0, 100
    models, preds, ys, rmses, maes, ms = [], [], [], [], [], []
    noise = None

    if isNoisy:
        noise = norm.rvs(scale=0.5, size=len(X))

    for m in range(max_vector_count):
        results = evaluate(kernel, m, l, X, x, noise)
        models.append(results[0])
        preds.append(results[1])
        ys.append(results[2])
        rmses.append(results[3])
        maes.append(results[4])
        ms.append(results[5])

        if min_rmse > rmses[m]:
            min_rmse = rmses[m]
            index = m

    return models[index], preds[index], ys[index], rmses[index], maes[index], ms[index]


def evaluate_kernel(X: np.ndarray, x: np.ndarray, isNoisy: bool = False) ->\
        [Any, np.ndarray, np.ndarray, float, float, float, float]:
    models, preds, ys, rmses, maes, ms = [], [], [], [], [], []
    index, min_rmse, opt_kernel, opt_l = 0, 100, 0, 0
    kernel_list = np.linspace(0.01, 2, 20)
    l_list = np.linspace(0.001, 1, 15)

    for kernel in kernel_list:
        for l in l_list:
            results = evaluate_vector(X, x, kernel, l, isNoisy)
            models.append(results[0])
            preds.append(results[1])
            ys.append(results[2])
            rmses.append(results[3])
            maes.append(results[4])
            ms.append(results[5])

            if min_rmse > rmses[len(rmses) - 1]:
                index = len(rmses) - 1
                min_rmse = rmses[index]
                opt_kernel = kernel
                opt_l = l

    if not isNoisy:
        kernel_rmses, l_rmses = [], []
        for kernel in kernel_list:
            results = evaluate_vector(X, x, kernel, opt_l)
            kernel_rmses.append(results[4])
        draw_graph(kernel_rmses, "Dependence between kernel parameters and RMSE", "Kernel parameters", "RMSE", kernel_list)

        for l in l_list:
            results = evaluate_vector(X, x, opt_kernel, l)
            l_rmses.append(results[4])
        draw_graph(l_rmses, "Dependence between threshold parameters and RMSE", "Threshold parameters", "RMSE", l_list)

    return models[index], preds[index], ys[index], rmses[index], maes[index], ms[index], opt_kernel, opt_l


def vector_dependence(X: np.ndarray, x: np.ndarray, kernel: Any, l: float) -> None:
    rmses, ms = [], []
    max_vector_count = 15
    #l_list = np.linspace(0.1, 0.999, 15)

    for m in range(max_vector_count):
        results = evaluate(kernel, m, l, X, x)
        rmses.append(results[3])
    draw_graph(rmses, "Dependence between support vector count and RMSE", "Support vectors", "RMSE")

    """for el in l_list:
        results = evaluate_vector(X, x, kernel, el)
        ms.append(results[5])
    draw_graph(ms, "Dependence between sparsity parameter and support vector count", "Sparsity parameter",
               "Support vectors", np.array([1 - l for l in l_list]))"""


def draw_graph(values: list, title: str, x_label: str, y_label: str, data: np.ndarray = None) -> None:
    if data is not None:
        plt.semilogy(data, values)
    else:
        plt.semilogy(values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def draw_krlst(X: np.ndarray, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray, krlst: Any, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x, f(x), 'r:', label=r'$f(x) = sinc(x)$')
    indexes = [i for i in range(len(X)) if X[i] in krlst.Xb]
    plt.plot(krlst.Xb, [y[i] for i in indexes], 'k*', markersize=20, label="Dictionary Elements")
    plt.plot(X, y, 'm.', markersize=10, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-2, 2)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


def main() -> None:
    epoch_count, size, left, right, percent = 15, 200, -2, 2, 0.5
    rmses = []
    X, x = get_training_data(size, left, right, percent)

    # KRLS-regression without noise
    model, y_pred, y, rmse, mae, m, kernel, l = evaluate_kernel(X, x)
    rmses.append(rmse)
    print("KRLS-regression without noise")
    print(f"Kernel parameter = {kernel}")
    print(f"Support vectors = {m}")
    print(f"Sparsity parameter = {l}")
    print(f"MAE = {mae}, RMSE = {rmse}")
    draw_krlst(X, x, y_pred, y, model, "KRLS-regression without noise")

    vector_dependence(X, x, kernel, l)

    # KRLS-regression with noise
    models, preds, ys, rmses, maes, ms, kernels, ls = [], [], [], [], [], [], [], []

    for epoch in range(epoch_count):
        results = evaluate_kernel(X, x, True)
        models.append(results[0])
        preds.append(results[1])
        ys.append(results[2])
        rmses.append(results[3])
        maes.append(results[4])
        ms.append(results[5])
        kernels.append(results[6])
        ls.append(results[7])

    draw_graph(rmses, "RMSE values with using a noise", "Iteration", "RMSE")

    index = np.array(rmses).argmin()
    print("\nKRLS-regression with noise and minimum RMSE")
    print(f"Kernel parameter = {kernels[index]}")
    print(f"Support vectors = {ms[index]}")
    print(f"Sparsity parameter = {ls[index]}")
    print(f"MAE = {maes[index]}, RMSE = {rmses[index]}")
    print(f"\nMean RMSE = {np.mean(rmses)}")
    draw_krlst(X, x, preds[index], ys[index], models[index], "KRLS-regression with noise and minimum RMSE")


main()
