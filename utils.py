import itertools
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import tensorflow as tf
from tensorflow import keras


def generateWindows(a, bins):
    aPadded = np.pad(a, bins, mode="wrap")
    aWindows = []
    for i in range(len(a)):
        i += bins
        aWindows.append(aPadded[i-bins:i+bins+1])
    return np.array(aWindows)


def c1(model, rho, c2=False, c2_unstacked=False):
    inputBins = model.layers[0].input_shape[0][1]
    windowBins = (inputBins - 1) // 2
    rhoWindows = generateWindows(rho, windowBins).reshape(rho.shape[0], inputBins, 1)
    if c2:
        rhoWindows = tf.Variable(rhoWindows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rhoWindows)
            result = model(rhoWindows)
        jacobiWindows = tape.batch_jacobian(result, rhoWindows).numpy().squeeze()
        c1_result = result.numpy().flatten()
        if c2_unstacked:
            return c1_result, jacobiWindows
        c2_result = np.row_stack([np.roll(np.pad(jacobiWindows[i], (0,rho.shape[0]-inputBins)), i-windowBins) for i in range(rho.shape[0])])
        return c1_result, c2_result
    return model.predict_on_batch(rhoWindows).flatten()


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

