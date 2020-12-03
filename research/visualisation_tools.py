import wfdb
import pandas as pd
from scipy.signal import spectrogram, cwt, ricker, get_window, butter, lfilter
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
from tqdm import tqdm
import matplotlib.cm as cm

AFIB_THRESHOLD = 0.5
WINDOW = 5


class SegmentVisualization(object):
    """
    Creating object of this class enables to visualize all three features.


    Attributes
    ----------
    signal_path : str
        The path of the recording chosen for visualization

    Example
    -------
    my_visu = SegmentVisualization('datasets/mitbih_afdb/04015')


    """

    def __init__(self, signal_path):
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
        self.lables = []

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
                ratio = len(chunk.loc[chunk.rhythm == "(AFIB"]) / 3000
                if ratio >= AFIB_THRESHOLD:
                    rhythm = 'AFIB'
                else:
                    rhythm = 'NORMAL'
                list_of_df.append({'index': i, 'ratio': ratio, 'rhythm': rhythm, 'chunk': chunk})
        return list_of_df

    @staticmethod
    def show_segment(segment):
        plt.figure()
        plt.plot(segment)
        plt.show()
        plt.xlabel('Probes')
        plt.ylabel('ECG(mV)')

    @staticmethod
    def show_spectogram(segment):
        window = get_window('hamming', 128)
        f, t, Sxx = spectrogram(segment, fs=250.0, window=window)
        plt.figure()
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), cmap=cm.gray)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

    @staticmethod
    def show_scalogram(segment):
        plt.figure()
        widths = np.arange(1, 256)
        cwt_for_segment = cwt(segment, ricker, widths)
        plt.imshow(cwt_for_segment, extent=[-1, 1, 1, 31], cmap='BrBG', aspect='auto',
                   vmax=abs(cwt_for_segment).max(), vmin=-abs(cwt_for_segment).max())

    @staticmethod
    def show_attractor(segment):
        # todo:
        #sprawdzic od ktorej probki te opoznienia, czy na pewno 66(1/3 cyklu) i czy nie powinno byc fltracji
        # OPTIMAL WYSZLO 7?
        tau = 220
        x = np.array(segment[2 * tau:])
        y = np.array(segment[tau:-tau])
        z = np.array(segment[:-2 * tau])

        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, label="attractor reconstruction (Tekens' delay)")
        ax.legend()
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')


        plt.show()
        plt.figure()
        plt.plot(y, x)
        plt.xlabel('x')
        plt.ylabel('y')


        # 2d attractor
        # plot x, y czy v, w?
        v = (x + y - 2 * z) / np.sqrt(6)
        w = (x - y) / np.sqrt(2)
        plt.figure()
        plt.plot(v, w)
        plt.xlabel('v')
        plt.ylabel('w')

        x = v
        y = w
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        nbins = 300
        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # Make the plot
        plt.figure()
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
        plt.show()
        plt.xlabel('v')
        plt.ylabel('w')

        # Change color palette
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
        plt.show()

    def search(self, key='rhythm', value='AFIB', output=None):
        if key in self.__keys:
            if output and output in self.__keys:
                return [element[output] for element in self.list_of_segments if element[key] == value]
            return [element for element in self.list_of_segments if element[key] == value]
        else:
            print('Unknown key')


signal_path = 'datasets/mitbih_afdb/04015'
my_visu = SegmentVisualization(signal_path)

all_segments = my_visu.search(key='rhythm', value='NORMAL')
segment = all_segments[100]['chunk']['ECG1']
my_visu.show_segment(segment)
# my_visu.show_spectogram(segment)
# my_visu.show_scalogram(segment)
my_visu.show_attractor(segment)