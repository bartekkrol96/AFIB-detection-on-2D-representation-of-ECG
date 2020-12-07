import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, cwt, ricker, get_window


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
