import itertools
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.integrate import simps
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

dz = 0.01

def generateWindows(a, bins):
    aPadded = np.pad(a, bins, mode="wrap")
    aWindows = np.empty((len(a), 2*bins+1))
    for i in range(len(a)):
        aWindows[i] = aPadded[i:i+2*bins+1]
    return aWindows


def c1(model, rho, c2=False):
    """
    Infer the one-body direct correlation profile c1 from a given density profile with a given neural correlation functional.

    model: The neural correlation functional
    rho: The density profile
    c2: If False, only return c1. If True, return both c1 as well as the corresponding two-body direct correlation function c2(x, x') which is obtained via autodifferentiation. If 'unstacked', give c2 as a function of x and x-x', i.e. as obtained naturally from the model.
    """
    inputBins = model.layers[0].input_shape[0][1]
    windowBins = (inputBins - 1) // 2
    rhoWindows = generateWindows(rho, windowBins).reshape(rho.shape[0], inputBins, 1)
    if c2:
        rhoWindows = tf.Variable(rhoWindows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rhoWindows)
            result = model(rhoWindows)
        jacobiWindows = tape.batch_jacobian(result, rhoWindows).numpy().squeeze() / dz
        c1_result = result.numpy().flatten()
        if c2 == "unstacked":
            return c1_result, jacobiWindows
        c2_result = np.row_stack([np.roll(np.pad(jacobiWindows[i], (0,rho.shape[0]-inputBins)), i-windowBins) for i in range(rho.shape[0])])
        return c1_result, c2_result
    return model.predict_on_batch(rhoWindows).flatten()


def Fexc(model, rho):
    alphas = np.linspace(0, 1, 30)
    integrandMap = {}
    for alpha in alphas:
        integrandMap[alpha] = np.sum(rho * c1(model, alpha * rho)) * dz
    Fexc = -simps(list(integrandMap.values()), list(integrandMap.keys()))
    return Fexc


class DataGenerator(keras.utils.Sequence):
    def __init__(self, simData, batch_size=32, shuffle=True, inputKeys=["rho"], outputKeys=["c1"], windowSigma=2.0):
        self.simData = simData
        self.inputKeys = inputKeys
        self.outputKeys = outputKeys
        self.windowSigma = windowSigma
        firstSimData = list(self.simData.values())[0]
        self.dz = 2 * firstSimData["z"][0]
        self.simDataBins = len(firstSimData["z"])
        self.windowBins = int(round(self.windowSigma/self.dz))
        self.inputData = {}
        self.outputData = {}
        self.validBins = {}
        for simId in self.simData.keys():
            valid = np.full(self.simDataBins, True)
            for k in self.outputKeys:
                valid = np.logical_and(valid, ~np.isnan(self.simData[simId][k]))
            self.validBins[simId] = np.flatnonzero(valid)
            self.inputData[simId] = structured_to_unstructured(np.pad(self.simData[simId][self.inputKeys], self.windowBins, mode="wrap"))
            self.outputData[simId] = structured_to_unstructured(np.pad(self.simData[simId][self.outputKeys], self.windowBins, mode="wrap"))
        self.batch_size = batch_size
        self.inputShape = (2*self.windowBins+1, len(self.inputKeys))
        self.outputShape = (len(self.outputKeys),)
        self.shuffle = shuffle
        self.on_epoch_end()
        print(f"Initialized DataGenerator from {len(self.simData)} simulations which will yield up to {len(self.indices)} input/output samples in batches of {self.batch_size}")

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        ids = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.inputShape))
        y = np.empty((self.batch_size, *self.outputShape))
        for b, (simId, i) in enumerate(ids):
            i += self.windowBins
            X[b] = self.inputData[simId][i-self.windowBins:i+self.windowBins+1]
            y[b] = self.outputData[simId][i]
        return X, y

    def on_epoch_end(self):
        self.indices = []
        for simId in self.simData.keys():
            self.indices.extend(list(itertools.product([simId], list(self.validBins[simId]))))
        if self.shuffle == True:
            np.random.default_rng().shuffle(self.indices)


class BulkCorrelationEvaluator:
    """
    This class provides utilities to calculate c2 and c3 of the bulk fluid.
    """
    def __init__(self, model, dx, rhob=0.7):
        self.model = model
        self.dx = dx
        self.rhob = rhob
        self.inputBins = self.model.layers[0].input_shape[0][1]
        self.channels = self.model.layers[0].input_shape[0][2]
        rhobWindow = np.full(self.inputBins, self.rhob)
        inputBinsHalf = self.inputBins // 2
        self.xWindow = self.dx * np.linspace(-inputBinsHalf, inputBinsHalf, self.inputBins)
        if self.channels == 1:
            inputWindow = rhobWindow.reshape(1, self.inputBins, 1)
        elif self.channels == 2:
            inputWindow = np.c_[self.xWindow, rhobWindow].reshape(1, self.inputBins, 2)
        self.inputWindow = tf.Variable(inputWindow)

    def c2(self):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.inputWindow)
            result = self.model(self.inputWindow)
        grad = tape.gradient(result, self.inputWindow)
        return tf.squeeze(grad).numpy() / self.dx

    def c3(self):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
            tape1.watch(self.inputWindow)
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
                tape2.watch(self.inputWindow)
                result = self.model(self.inputWindow)
            c2 = tape2.gradient(result, self.inputWindow)
        c3 = tape1.jacobian(c2, self.inputWindow)
        return tf.squeeze(c3).numpy() / self.dx**2


class PlotBulkCorrelationCallback(keras.callbacks.Callback):
    """
    This class implements a callback which can be used during training to monitor the quality of bulk correlations.
    """
    def __init__(self, when="on_batch_end", every=1, options=None):
        super().__init__()
        setattr(self, when, self._doPlot)
        self.every = every
        self.options = options
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.c2Graph, = self.ax.plot(0)

    def _doPlot(self, index, logs=None):
        if index % self.every != 0:
            return
        c2 = self.testProfilesEvaluator.c2()
        self.c2Graph.set_data(self.testProfilesEvaluator.xWindow, c2)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_begin(self, index, logs=None):
        self.testProfilesEvaluator = BulkCorrelationEvaluator(self.model, **self.options)
