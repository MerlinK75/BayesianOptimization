import numpy as np
from pylsl import StreamInlet, resolve_stream
from Filter_func import filtering
from Instantaneous import peak_calc
import matplotlib.pyplot as plt

###This py file calculates the current theta and alpha peaks to maximise nonharmonic locking
def Framework(tmin, tmax, baseline_path) ->None:
    print("Framework_start")
    ##Extracts the EEG data from the LSL layer and calculates srate
    nirs_stream = resolve_stream('type','NIRS')
    nirs_inlet = StreamInlet(nirs_stream[0])
    srate = nirs_inlet.info().nominal_srate()

    ##Fits the ASR to the baseline recording for preprocessing
    filt = filtering(srate)
    asr = filt.baseline(tmin, tmax, False, baseline_path) ##Participant, session, Tmin and Tmax at the end in seconds and if a plot is needed

    ##Prepares the epoch buffer and the feature extraction object
    epoch_buffer = []
    
    ifcalc = peak_calc(srate)
    


    while True:
        
        sample, timestamp = nirs_inlet.pull_sample()
        

        #####Filtering
        filtered_eeg = filt.online_bandpass_filter(sample, 1, 40)
        epoch_buffer.append(filtered_eeg)

        ##If epoch of 5s is formed
        if (len(epoch_buffer) >= int(srate*5)):
            
            epoch_buffer = epoch_buffer[len(epoch_buffer)-int(srate*5):]

            #preprocessing
            clean = asr.transform(np.array(epoch_buffer).T)
            columns_to_delete = [
                i for i in range(clean.shape[1]) if filt.column_has_value_above_threshold(clean[:, i], 100)
                ]
            processed_data = np.delete(clean, columns_to_delete, axis=1)
         
            ##Calculate desired alpha range from mean theta, additionally send mean and desired alpha to RLS class
            mean_alpha, mean_theta = ifcalc.mean_peaks(processed_data)

            if not Cur_Queue.empty():
                if Cur_Queue.get(): 
                    child_current.send(mean_alpha)
                    print(f"mean_alpha: {mean_alpha}")

            Desired_A = des_alpha(mean_theta)

            if not Des_Queue.empty():
                if Des_Queue.get():
                    child_desired.send(Desired_A)

            #Remove last second
            epoch_buffer = epoch_buffer[len(epoch_buffer)-int(srate*4):]
