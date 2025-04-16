
import os
import numpy as np
from typing import List, Tuple
import h5py
from functions import getNMostDifferentSpectra, reduceSpecsToNWavenumbers, remapSpectrumToWavenumbers


def load_reference_Raman_spectra() -> np.ndarray:
    specs = []
    file = h5py.File(r"RamanReferenceSpectra/Raman reference spectra.h5")
    skipIndices: List[int] = [7, 16]  # indices of spectra that are not clean..
    for i, sample in enumerate(file['Samples'].keys()):
        if i in skipIndices:
            continue
        sample = file['Samples'][str(sample)]
        spec = sample['Spectra']
        data = spec[str(list(spec.keys())[0])]
        origSpec = data['SpectralData<p:ArbitrarySpacedOriginalSpectrum>']

        spec = np.array(origSpec)
        spec = (spec - spec.min()) / (spec.max() - spec.min())

        if i == 0:  # Estimate wavenumbers, I manually took the values from a PET spectrum (it's the last one)
            peak1, peak2 = 1750, 3100
            ind1, ind2 = 690, 1311
            wavenums = np.arange(len(spec))
            wavenums -= ind1
            wavenums = wavenums * (peak2 - peak1) / (ind2 - ind1)
            wavenums += peak1
            specs.append(wavenums)

        specs.append(spec)

    return np.array(specs).transpose()


def load_specCSVs_from_directory(path: str, fixName: str = None, maxSpectra=1e8) -> Tuple[List[str], np.ndarray]:
    """
    Reads Spectra from CSV files in path. If given, a fix name is assigned to each spectrum
    :param path: Directory path
    :param fixName: If None, each spectrum has the filename as name, otherwise the indicated fixName
    :param maxSpectra: Max number of spectra to take.
    :return: Tuple[Assignment List, spectra array]
    """
    spectra: np.ndarray = None
    names: list = []
    for file in os.listdir(path):
        if file.lower().endswith('.csv'):
            curSpec: list = []
            specName = fixName if fixName is not None else file.lower().split('.csv')[0]
            names.append(specName)

            with open(os.path.join(path, file), 'r') as fp:
                if spectra is None:
                    wavenumbers = []
                    # for index, row in enumerate(reader):
                    for line in fp.readlines():
                        wavenum, intensity = get_numbers_from_line(line)
                        curSpec.append(intensity)
                        wavenumbers.append(wavenum)

                    spectra = np.array(wavenumbers)
                else:
                    tmpSpec = []
                    tmpWavenumber = []
                    for line in fp.readlines():
                        wavenum, intensity = get_numbers_from_line(line)
                        tmpSpec.append(intensity)
                        tmpWavenumber.append(wavenum)

                    tmpSpec = np.array(tmpSpec)
                    tmpWavenumber = np.array(tmpWavenumber)
                    for number in spectra[:, 0]:
                        index = np.argmin(np.abs(tmpWavenumber - number))
                        curSpec.append(tmpSpec[index])

                if len(spectra.shape) == 1:
                    spectra = np.append(spectra[:, np.newaxis], np.array(curSpec)[:, np.newaxis], axis=1)
                else:
                    spectra = np.append(spectra, np.array(curSpec)[:, np.newaxis], axis=1)

    numSpectra = spectra.shape[1] - 1
    
    if numSpectra > maxSpectra:
        names, spectra = getNMostDifferentSpectra(names, spectra, maxSpectra)

    return names, spectra


def get_numbers_from_line(line: str) -> Tuple[float, float]:
    """
    Takes a line from a csv or txt document and checks for delimiter and decimal separator to yield exactly
    two float numbers
    :param line:
    :return: the two float numbers as Tuple
    """
    origline = line.strip()
    try:
        line = origline.split(';')
        assert len(line) == 2
        numbers: Tuple[float, float] = float(line[0].replace(',', '.')), float(line[1].replace(',', '.'))
    except AssertionError:
        line = origline.split(',')
        assert len(line) == 2
        numbers: Tuple[float, float] = float(line[0]), float(line[1])
    except AssertionError as e:
        print(e)
        raise
    return numbers
