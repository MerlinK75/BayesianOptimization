import mne
import numpy as np
from scipy.signal import butter, lfilter
from meegkit.asr import ASR
import matplotlib.pyplot as plt

###Code with filtering so ASR is fitted for online data collection

class filtering:
    def __init__(self, srate):
        self.srate = srate
        ##Removes any values outside of threshold range
        
    def column_has_value_above_threshold(self, column, thresh):
        return np.any((column > thresh) | (column < -thresh))

    # Define the bandpass filter function
    def butter_bandpass(self, lowcut, highcut, order=4):
        nyquist = 0.5 * self.srate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # Define the online filtering function
    def online_bandpass_filter(self, data, lowcut, highcut):
        b, a = self.butter_bandpass(lowcut, highcut)
        filtered_data = lfilter(b, a, data)
        return filtered_data
    
    ###Main function
    def baseline(self, tmin, tmax, plot, path):
        #path = r"C:\Users\user\OneDrive - University College London\Desktop\Study_!\P23\Baseline_02.edf" #Place filename here and recording name
        time_range = [tmin, tmax] 
        data=mne.io.read_raw_edf(path,# + "_baseline.edf", (error)
                                preload=True)
        EEG_data = data.get_data()
        
        filtered_eeg = self.online_bandpass_filter(EEG_data, 1, 40)
        #Plots data to determine if this region is clean, need to fix the orientation
        if plot:
            #['UNI 01', 'UNI 03', 'UNI 04', 'UNI 05', 'UNI 06', 'UNI 07', 'UNI 08']

            fig, axs = plt.subplots(2, 5)
            axs[0, 0].plot(filtered_eeg[6, time_range[0]*self.srate:time_range[1]*self.srate])
            axs[0, 0].set_title('F8')
            axs[0, 1].plot(filtered_eeg[1, time_range[0]*self.srate:time_range[1]*self.srate])
            axs[0, 1].set_title('FP2')
            axs[0, 3].plot(filtered_eeg[0, time_range[0]*self.srate:time_range[1]*self.srate])
            axs[0, 3].set_title('FP2')
            axs[0, 4].plot(filtered_eeg[2, time_range[0]*self.srate:time_range[1]*self.srate])
            axs[0, 4].set_title('F7')
            axs[1, 1].plot(filtered_eeg[5, time_range[0]*self.srate:time_range[1]*self.srate])
            axs[1, 1].set_title('F4')
            axs[1, 2].plot(filtered_eeg[4, time_range[0]*self.srate:time_range[1]*self.srate])
            axs[1, 2].set_title('Fz')
            axs[1, 3].plot(filtered_eeg[3, time_range[0]*self.srate:time_range[1]*self.srate])
            axs[1, 3].set_title('F3')
            plt.show()
        else: ###Produces the asr class
            asr = ASR(sfreq=self.srate, cutoff=20, method="euclid")
            train_idx = np.arange(time_range[0] * self.srate, time_range[1] * self.srate, dtype=int) #Arbritraily chosen, could have a better method of selecting the train index
            asr.fit(filtered_eeg[:7, train_idx]) #Set to 7, the number of channels
            print("ASR generated")

            return asr
