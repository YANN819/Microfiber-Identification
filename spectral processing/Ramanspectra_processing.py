

import time
import numpy as np
import random
import os
import distort
import csv
import importData as io
import matplotlib.pyplot as plt
import outGraphs as out
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from scipy.optimize import nnls
from globals import SPECLENGTH
from copy import copy
from Reconstruction import prepareSpecSet, Reconstructor, getVCNN, getDenseReconstructor, getConvReconstructor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, Callback # type: ignore
import tensorflow as tf

os.chdir(os.path.dirname(os.getcwd()))

noiseLevel: float = 0.25
fracValid: float = 0.2
numVariations: int = 50
useConvNetwork: bool = False
randomShuffle: bool = True
specTypesTotal: int = 15
plot = False

def lr_scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch != 0:
        lr = lr * 0.5
    return lr

experimentTitle = f"Augumentation of Raman Data of Microfibers"
print(experimentTitle)

class TrainingHistoryLogger(Callback):
    def __init__(self, filename):
        self.filename = filename
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epoch'] = epoch
        logs['learning_rate'] = self.model.optimizer.lr.numpy()
        self.history.append(logs)
    
    def on_train_end(self, logs=None):
        keys = self.history[0].keys()
        with open(self.filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.history)

t0 = time.time()
path = r'D:\Anaconda3\pythonwork\ramanspec_file\data\Ramandata_csv'
specNames, spectra = io.load_specCSVs_from_directory(path, None, 1e6)
wavenums = spectra[:, 0].copy()
specs: np.ndarray = spectra[:, 1:]
dbSpecs: np.ndarray = specs.copy()
dbNames: list[str] = copy(specNames)

#specs = normalizeSpecSet(specs)

scaler = MinMaxScaler()
specs = scaler.fit_transform(specs)

print(specNames)
print(wavenums)
print(specs.shape)
print(f'Loading Raman spectra took {round(time.time()-t0)} seconds')


if randomShuffle:
    specs = np.tile(specs, (1, numVariations))
    numSpecs = specs.shape[1]
    random.seed(32)
    valIndices = random.sample(range(numSpecs), int(round(numSpecs * fracValid)))
    trainIndices = [i for i in range(numSpecs) if i not in valIndices]
    trainSpectra: np.ndarray = specs[:, trainIndices]
    testSpectra: np.ndarray = specs[:, valIndices]
    allSpecNames = specNames*numVariations
    testNames: list[str] = [allSpecNames[i] for i in valIndices]
    trainNames = [allSpecNames[i] for i in trainIndices]
else:
    numTestSpectra = int(round(fracValid * specTypesTotal))
    numTrainSpectra = specTypesTotal - numTestSpectra
    trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariations))
    testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariations))
    trainNames: list[str] = specNames[:numTrainSpectra] * numVariations
    testNames: list[str] = specNames[numTrainSpectra:] * numVariations

print(trainSpectra.shape)
print(testSpectra.shape)

t0 = time.time()
train_seed = 25
test_seed = 30

noisyTrainSpectra1 = distort.add_noise(trainSpectra, level=noiseLevel, seed=train_seed, ramanMode=True)
noisyTestSpectra1 = distort.add_noise(testSpectra, level=noiseLevel, seed=test_seed, ramanMode=True)

noisyTrainSpectra1 = distort.add_periodic_interferences_raman(noisyTrainSpectra1, seed=train_seed)
noisyTestSpectra1 = distort.add_periodic_interferences_raman(noisyTestSpectra1, seed=test_seed)

levelRange = (0.2, 1.0)
noisyTrainSpectra1 = distort.add_fluorescence(noisyTrainSpectra1, levelRange=levelRange, seed=train_seed)
noisyTestSpectra1 = distort.add_fluorescence(noisyTestSpectra1, levelRange=levelRange, seed=test_seed)

noisyTrainSpectra1 = distort.add_cosmic_ray_peaks(noisyTrainSpectra1, numRange=(1.0, 2.0), seed=train_seed)
noisyTestSpectra1 = distort.add_cosmic_ray_peaks(noisyTestSpectra1, numRange=(1.0, 2.0), seed=test_seed)



trainSpectra = prepareSpecSet(trainSpectra, addDimension=useConvNetwork)
testSpectra = prepareSpecSet(testSpectra, addDimension=useConvNetwork)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, addDimension=useConvNetwork)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, addDimension=useConvNetwork)

if useConvNetwork:
    rec: Reconstructor = getVCNN()
else:
    rec: Reconstructor = getDenseReconstructor(dropout=0.1 if randomShuffle else 0.30)

early_stop_callback = EarlyStopping(monitor='loss', patience=10)
lr_callback = LearningRateScheduler(lr_scheduler)
history_logger = TrainingHistoryLogger(r'D:\Anaconda3\pythonwork\ramanspec_file\dataprocessing_history.csv')

t0 = time.time()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs = 80, 
                  validation_data = (noisyTestSpectra, testSpectra),
                  batch_size = 32, 
                  shuffle = True,
                  callbacks =  [early_stop_callback, lr_callback, history_logger]
                  )
print(f"Training took {round(time.time()-t0, 2)} seconds.")

rec.save_weights(r'D:\Anaconda3\pythonwork\ramanspec_file\data_processing\AutoEncoder_modelweights_20250114.h5')

t0 = time.time()
reconstructedtrainSpecs = rec.call(noisyTrainSpectra)
reconstructedtestSpecs = rec.call(noisyTestSpectra)
print(f'reconstruction took {round(time.time()-t0, 2)} seconds')

train_output_file = r'D:\Anaconda3\pythonwork\ramanspec_file\data_training\rec_train_spectra0515.csv'
with open(train_output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Wavenumbers'] + wavenums.tolist())
    for name, spectrum in zip(trainNames, reconstructedtrainSpecs.numpy()):
        row = [name] + spectrum.tolist()
        csvwriter.writerow(row)

test_output_file = r'D:\Anaconda3\pythonwork\ramanspec_file\data_training\rec_test_spectra0515.csv'
with open(test_output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Wavenumbers'] + wavenums.tolist())
    for name, spectrum in zip(testNames, reconstructedtestSpecs.numpy()):
        row = [name] + spectrum.tolist()
        csvwriter.writerow(row)

print(f'Reconstructed train spectra saved to {train_output_file}')
print(f'Reconstructed test spectra saved to {test_output_file}')

if plot:

    histplot = out.getHistPlot(history.history, annotate=False)
    specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedtestSpecs,
                                                includeSavGol=True,
                                                wavenumbers=wavenums,
                                                randomIndSeed= None
                                                )
    specPlot.subplots_adjust(top=0.94, bottom=0.05, left=0.2, right=0.8, hspace=0.32, wspace=0.05)
    plt.show()

mse_AE = mean_squared_error(testSpectra, reconstructedtestSpecs)
testSpectra_np = testSpectra.numpy()
reconstructedtestSpecs_np = reconstructedtestSpecs.numpy()
r_AE = pearsonr(testSpectra_np.flatten(), reconstructedtestSpecs_np.flatten())[0]
r2_AE = r2_score(testSpectra, reconstructedtestSpecs)
signal_energy_AE = np.sum(np.square(testSpectra))
noise_energy_AE = np.sum(np.square(testSpectra - reconstructedtestSpecs))
snr_AE = 10 * np.log10(signal_energy_AE / noise_energy_AE)


print(f"Autoencoder MSE: {mse_AE}, Pearson r: {r_AE}, R^2: {r2_AE}, SNR: {snr_AE}")
