import numpy as np
import matplotlib.pyplot as plt
#import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
from HelpFunctions import *

ALL_SIMULATED_DATA_MAT = "real_signals_nifeadb_mat"
WINDOWED_SIMULATED_SIGNAL = "real_signals_nifeadb_1024windows_NR_10_20_Dec_22"

dir_path = os.path.dirname(os.path.realpath(__file__))
save_mat_dir = os.path.join(dir_path,ALL_SIMULATED_DATA_MAT)
window_sim_dir = os.path.join(dir_path,WINDOWED_SIMULATED_SIGNAL)


if __name__ == '__main__':

    # Merging maternal ecg + noise + fetal ecg
    # dividing into windows and saving
    files = os.listdir(save_mat_dir).copy()
    for filename in os.listdir(save_mat_dir):
        if "NR_10" not in filename:
            continue
        print(filename)
        files.remove(filename)
        name = filename[:len(filename)-4]
        #name = filename
        signal = loadmat(os.path.join(save_mat_dir, name))['data']
        signal = signal.reshape(-1)

        # FECG-SYN is samples at 250Hz while NIFEA-DB at 1KHz
        window_size = 1024*4
        number_of_window = len(signal)//window_size
        #num_of_signal_to_remove = 248
        #sig1 = loadmat(os.path.join(save_mat_dir, signals[0]))['data'][num_of_signal_to_remove:]


        for i in range(number_of_window):
            data = signal[i*window_size:(i+1)*window_size]
            abs_signal = [abs(data[i]) for i in range(len(data))]
            max_signal = max(abs_signal)
            signal_new = [sample / max_signal for sample in data]
            new_signal = signal_new[0:1024]
            for j in range(1024):
                new_signal[j] = signal_new[j*4]

            signal_new = new_signal
            sio.savemat(os.path.join(window_sim_dir, name + '_win_' + str(i)) + '.mat', {'data': signal_new})
            #sio.savemat(os.path.join(window_sim_dir, name[:len(filename)-4] + str(i)) + '.mat', {'data': data})

