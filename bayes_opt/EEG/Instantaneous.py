import numpy as np
from scipy.signal import firwin, filtfilt, hilbert
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

class peak_calc:
    def __init__(self, srate):
        self.srate = srate
        self.orders = np.round(np.linspace(srate * 0.01, srate * 0.4, 10)).astype(int)
    
    def column_has_value_above_threshold(self, column, min, max):
            return np.any((column > max) | (column < min))
    ##Applies plateau band pass for theta and alpha (error)
    def filter(self, sample):
        filt_order = round(3 *(self.srate/4))
        ny = self.srate/2
        Theta_b = firwin(filt_order, [4/ny, 8/ny], pass_zero=False)
        Theta_filt = filtfilt(Theta_b, [1.0], sample)

        Alp_b = firwin(filt_order, [8/ny, 14/ny], pass_zero=False)
        Alpha_filt = filtfilt(Alp_b, [1.0], sample)
        return Alpha_filt, Theta_filt
    ###Follows the method used in Cohen (2014) to calculate IF
    def InsFreq_calc(self, Alpha_filt, Theta_filt):

        Alpha_angle = np.unwrap(np.angle(hilbert(Alpha_filt)))
        Theta_angle = np.unwrap(np.angle(hilbert(Theta_filt)))

        instantaneous_frequency_alp = (self.srate * np.diff(Alpha_angle) / (2.0*np.pi))
        instantaneous_frequency_the = (self.srate * np.diff(Theta_angle) / (2.0*np.pi))



        return instantaneous_frequency_alp, instantaneous_frequency_the
    ###Applies a median filter to reject noise present in the IF data, it is then the average peak is calculated
    def median_filt(self, instantaneous_frequency_alp, instantaneous_frequency_the):
        aphasemed = [None] * len(self.orders)
        tphasemed = [None] * len(self.orders)
        freqslide_filt_alpha = [None] * len(instantaneous_frequency_alp)
        freqslide_filt_theta = [None] * len(instantaneous_frequency_the)
        for Ei in range(len(instantaneous_frequency_alp)):
            for oi in range(len(self.orders)):
                aphasemed[oi] = median_filter(instantaneous_frequency_alp[Ei], self.orders[oi])
                tphasemed[oi] = median_filter(instantaneous_frequency_the[Ei], self.orders[oi])

            freqslide_filt_alpha[Ei] = np.median(np.array(aphasemed), axis=0)
            freqslide_filt_theta[Ei] = np.median(np.array(tphasemed), axis=0)

        columns_to_delete = [
            i for i in range(np.array(freqslide_filt_alpha).shape[1]) if self.column_has_value_above_threshold(np.array(freqslide_filt_alpha)[:, i], 8, 14)
            ]
        freqslide_filt_alpha = np.delete(freqslide_filt_alpha, columns_to_delete, axis=1)

        columns_to_delete = [
            i for i in range(np.array(freqslide_filt_theta).shape[1]) if self.column_has_value_above_threshold(np.array(freqslide_filt_theta)[:, i], 4, 8)
            ]
        freqslide_filt_theta = np.delete(freqslide_filt_theta, columns_to_delete, axis=1)

        mean_alpha = np.mean(freqslide_filt_alpha, axis = 1) #7 long variable now

        mean_theta = np.mean(freqslide_filt_theta, axis = 1)
        return mean_alpha, mean_theta
    ###Final code to convert LSL epoch to mean alpha and theta peaks
    def mean_peaks(self, sample):
        Alpha_filt, Theta_filt = self.filter(sample)

        instantaneous_frequency_alp, instantaneous_frequency_the = self.InsFreq_calc(Alpha_filt, Theta_filt)

        mean_alpha, mean_theta = self.median_filt(instantaneous_frequency_alp, instantaneous_frequency_the)
        return mean_alpha, mean_theta


