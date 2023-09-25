import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from utils import c1

tf.config.experimental.enable_tensor_float_32_execution(False)


def calc_rho_nDFT(model, profiles, T=1.0, plot=False, maxiter=10000):
    zs = profiles["z"]
    muloc = profiles["muloc"]

    def plotInit(zs, rho, rho_sim=None):
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_ylim(0, 4)
        if rho_sim is not None:
            ax.plot(zs, rho_sim)
        rhoGraph, = ax.plot(zs, rho)
        plt.show()
        return fig, rhoGraph

    def plotUpdate(fig, rhoGraph, rho):
        rhoGraph.set_ydata(rho)
        fig.canvas.draw()
        fig.canvas.flush_events()

    rho = np.zeros_like(zs)
    rho_new = np.zeros_like(zs)
    valid = np.isfinite(muloc)
    rho[valid] = 0.5
    if plot:
        fig, rhoGraph = plotInit(zs, rho, rho_sim=profiles["rho"])
    alpha = 0.00001
    i = 0
    while True:
        if i > 10:
            alpha = 0.0001
        if i > 20:
            alpha = 0.001
        if i > 50:
            alpha = 0.01
        if i > 100:
            alpha = 0.05
        if i > 300:
            alpha = 0.01
        if i > 5000:
            alpha = 0.001
        if plot:
            plotUpdate(fig, rhoGraph, rho)
        rho_new[valid] = np.exp(muloc / T + c1(model, rho))[valid]
        rho = (1 - alpha) * rho + alpha * rho_new
        delta = np.max(np.abs(rho_new - rho))
        relative_error = delta / np.max(rho)
        print(i)
        print(delta)
        i += 1
        if delta < 1e-5 or relative_error < 1e-5:
            print(f"Converged after {i} iterations (delta = {delta})")
            return rho
        if i > maxiter:
            print(f"Not converged after {i} iterations (delta = {delta})")
            return None


def do_all_test():
    model = keras.models.load_model("models/HS")
    simData = np.load("data/HS.npy", allow_pickle=True).item()
    for key, profiles in simData["test"].items():
        print(f"Doing {key}")
        rho_nDFT = calc_rho_nDFT(model, profiles, plot=True)
        # Do what you want with rho_nDFT here, e.g. save it to disk for further analysis


do_all_test()

