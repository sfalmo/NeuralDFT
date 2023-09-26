import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from utils import DataGenerator

tf.config.experimental.enable_tensor_float_32_execution(False)


inputKeys = ["rho"]
outputKeys = ["c1"]

# Read simulation data from the prepared files
simData = np.load("data/HS.npy", allow_pickle=True).item()

# simData is now a dict with the keys "training", "validation" and "test".
# Each of these sets is a dict with the following structure: {simId1: profiles1, simId2: profiles2, ...}
# simIdX is a unique identifier of the respective simulation and profileX is a numpy structured array which contains the position "z", the density profile "rho", the local chemical potential "muloc" and the one-body direct correlation function "c1".

generatorOptions = dict(batch_size=128, windowSigma=2.56, inputKeys=inputKeys, outputKeys=outputKeys)
trainingGenerator = DataGenerator(simData["training"], **generatorOptions)
validationGenerator = DataGenerator(simData["validation"], **generatorOptions)

# Define the model
inputs = keras.Input(shape=trainingGenerator.inputShape, name="_".join(inputKeys))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(512, activation="softplus")(x)
x = keras.layers.Dense(512, activation="softplus")(x)
x = keras.layers.Dense(512, activation="softplus")(x)
outputs = keras.layers.Dense(trainingGenerator.outputShape[0], name="_".join(outputKeys))(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()]
)
model.summary()

# Specify a learning rate schedule which exponentially decreases the learning rate after the first few epochs.
def lrschedule(epoch, lr):
    if epoch > 5:
        lr *= 0.95
    return lr

# Do the training
fitHistory = model.fit(
    trainingGenerator,
    validation_data=validationGenerator,
    epochs=100,
    callbacks=[
        keras.callbacks.LearningRateScheduler(lrschedule),
        keras.callbacks.ModelCheckpoint(filepath="models/currentBest", monitor="val_mean_absolute_error", save_best_only=True),
    ]
)
