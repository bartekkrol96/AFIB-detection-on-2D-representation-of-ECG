import tensorflow as tf
import wfdb
import pandas as pd
from scipy.signal import spectrogram, cwt, ricker, get_window
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
from tqdm import tqdm
import matplotlib.cm as cm
from tensorflow.keras import datasets, layers, models, utils
import wfdb
from scipy.signal import spectrogram, cwt, ricker, get_window
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kde
from tqdm import tqdm
import os
import pickle as pkl
from matplotlib import image
from PIL import Image
plt.ioff()

AFIB_THRESHOLD = 0.5
WINDOW = 5
GLOBAL_NR_AFIB = 0
GLOBAL_NR_nonAFIB = 0


class TrainProbeGenerator:
    def __init__(self, signal_path=None):
        if signal_path:
            self.signal_path = signal_path
            self.ann = wfdb.rdann(self.signal_path, 'atr')
            self.ann_samples = self.ann.sample
            self.fs = self.ann.fs
            self.record = wfdb.rdrecord(self.signal_path)
            ecg = self.record.p_signal
            dataframe = pd.DataFrame(ecg, columns=['ECG1', 'ECG2'])
            self.dataframe = dataframe.assign(rhythm='NOISE', segment=0)
            self.__update_rhythms()
            self.__keys = ['index', 'ratio', 'rhythm', 'chunk']

            self.window_size = WINDOW * self.fs
            self.list_of_segments = self.split_dataframe_into_smaller(self.dataframe, chunk_size=self.window_size)

    def __update_rhythms(self):
        print('update rhythms')
        for index, rhytm in enumerate(self.ann.aux_note):
            if index + 1 >= len(self.ann_samples):
                new_df = pd.DataFrame({'rhythm': rhytm, 'segment': index + 1},
                                      index=range(self.ann_samples[index], len(self.dataframe)))
            else:
                new_df = pd.DataFrame({'rhythm': rhytm, 'segment': index + 1},
                                      index=range(self.ann_samples[index], self.ann_samples[index + 1]))
            self.dataframe.update(new_df)

    @staticmethod
    def split_dataframe_into_smaller(df, chunk_size=3000):
        list_of_df = list()
        number_chunks = len(df) // chunk_size + 1
        for i in range(number_chunks):
            chunk = df[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) == chunk_size:
                ratio = len(chunk.loc[chunk.rhythm == "(AFIB"]) / chunk_size
                if ratio >= AFIB_THRESHOLD:
                    rhythm = 'AFIB'
                else:
                    rhythm = 'NORMAL'
                list_of_df.append({'index': i, 'ratio': ratio, 'rhythm': rhythm, 'chunk': chunk})
        return list_of_df

    @staticmethod
    def show_spectogram(segment):
        window = get_window('hamming', 128)
        f, t, Sxx = spectrogram(segment, fs=250.0, window=window)
        spect = plt
        spect.figure()
        spect.pcolormesh(t, f, 10 * np.log10(Sxx), cmap=cm.gray)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        spect.gca().set_axis_off()
        spect.margins(0, 0)
        spect.gca().xaxis.set_major_locator(plt.NullLocator())
        spect.gca().yaxis.set_major_locator(plt.NullLocator())
        return spect

    @staticmethod
    def show_scalogram(segment):
        scalo = plt
        scalo.figure()
        widths = np.arange(1, 31)
        cwt_for_segment = cwt(segment, ricker, widths)
        scalo.imshow(cwt_for_segment, extent=[-1, 1, 1, 31], cmap=cm.gray, aspect='auto',
                   vmax=abs(cwt_for_segment).max(), vmin=-abs(cwt_for_segment).max())
        scalo.gca().set_axis_off()
        scalo.margins(0, 0)
        scalo.gca().xaxis.set_major_locator(plt.NullLocator())
        scalo.gca().yaxis.set_major_locator(plt.NullLocator())
        return scalo

    @staticmethod
    def show_attractor(segment):
        # todo:
        # sprawdzic od ktorej probki te opoznienia, czy na pewno 66(1/3 cyklu) i czy nie powinno byc fltracji
        # OPTIMAL WYSZLO 7?
        tau = 220
        x = np.array(segment[2 * tau:])
        y = np.array(segment[tau:-tau])
        z = np.array(segment[:-2 * tau])

        v = (x + y - 2 * z) / np.sqrt(6)
        w = (x - y) / np.sqrt(2)

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        try:
            nbins = 300
            k = kde.gaussian_kde([v, w])
            xi, yi = np.mgrid[v.min():v.max():nbins * 1j, w.min():w.max():nbins * 1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            atr = plt
            atr.figure()
            atr.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cm.gray)
            atr.gca().set_axis_off()
            atr.margins(0, 0)
            atr.gca().xaxis.set_major_locator(plt.NullLocator())
            atr.gca().yaxis.set_major_locator(plt.NullLocator())
            return atr
        except:
            pass


directory = os.fsencode('/Users/projekt/Desktop/files')
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file.decode())
    if filename.endswith(".dat"):
        print(filename)
        signal_path = '/Users/projekt/Desktop/files/'+filename[:-4]
        my_visu = TrainProbeGenerator(signal_path)
        all_segments = my_visu.list_of_segments
        for segment in tqdm(all_segments):
            ecg1 = segment['chunk']['ECG1']
            label = segment['rhythm']

            try:
                # specto = my_visu.show_spectogram(ecg1)
                # scalo = my_visu.show_scalogram(ecg1)
                atr = my_visu.show_attractor(ecg1)
                if label=='AFIB':
                    atr.savefig("/Volumes/Backup Plus/MGR_DATASET/ATRACTOR/AFIB/{}.jpg".format(GLOBAL_NR_AFIB), bbox_inches='tight',
                            pad_inches=0, dpi=15)
                    atr.close()
                    GLOBAL_NR_AFIB += 1
                else:
                    atr.savefig("/Volumes/Backup Plus/MGR_DATASET/ATRACTOR/nonAFIB/{}.jpg".format(GLOBAL_NR_nonAFIB), bbox_inches='tight',
                                   pad_inches=0, dpi=15)
                    atr.close()
                    GLOBAL_NR_nonAFIB += 1
            except:
                continue
