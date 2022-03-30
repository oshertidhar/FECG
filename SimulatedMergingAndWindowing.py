import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
from HelpFunctions import *

ALL_SIMULATED_DATA_MAT = "simulated_signals_mat"
#ALL_SIMULATED_DATA_MAT = "SimulatedDatabaseMat"
WINDOWED_SIMULATED_SIGNAL = "simulated_windows_noise_without_c3"

if not os.path.exists(WINDOWED_SIMULATED_SIGNAL):
    os.mkdir(WINDOWED_SIMULATED_SIGNAL)

dir_path = os.path.dirname(os.path.realpath(__file__))
save_mat_dir = os.path.join(dir_path,ALL_SIMULATED_DATA_MAT)
window_sim_dir = os.path.join(dir_path,WINDOWED_SIMULATED_SIGNAL)


if __name__ == '__main__':
    # Merging Data,dividing into windows and saving
    files = os.listdir(save_mat_dir).copy()
    for filename in os.listdir(save_mat_dir):
        if 'fecg1' not in filename:
            continue
        files.remove(filename)
        name = filename[:len(filename)-6]
        signals = [filename]
        for elem in files:
            tmp = '_'.join(str.split(elem,'_')[:-1])
            if tmp == name:
                if 'fecg2' in elem:
                    continue
                else:
                    signals.append(elem)
        size = len(signals)
        print(signals)

        window_size = 1024
        number_of_window = 73
        num_of_signal_to_remove = 248

        if size == 2:
            sig1 = loadmat(os.path.join(save_mat_dir, signals[0]))['data'][num_of_signal_to_remove:]
            sig2 = loadmat(os.path.join(save_mat_dir, signals[1]))['data'][num_of_signal_to_remove:]
            for i in range(number_of_window):
                record = [a + b  for a, b in zip(sig1[i*window_size:(i+1)*window_size], sig2[i*window_size:(i+1)*window_size])]
                sio.savemat(os.path.join(window_sim_dir, name + '_noise0' + '_mix' + str(i)), {'data': record})
                for elem in signals:
                    if ('fecg1' in elem or 'mecg' in elem):
                        data = loadmat(os.path.join(save_mat_dir, elem))['data'][i*window_size:(i+1)*window_size]
                        sio.savemat(os.path.join(window_sim_dir, elem + str(i)), {'data':data})

        elif size == 3:
            sig1 = loadmat(os.path.join(save_mat_dir, signals[0]))['data'][num_of_signal_to_remove:]
            sig2 = loadmat(os.path.join(save_mat_dir, signals[1]))['data'][num_of_signal_to_remove:]
            sig3 = loadmat(os.path.join(save_mat_dir, signals[2]))['data'][num_of_signal_to_remove:]
            for i in range(number_of_window):
                record = [a + b + c  for a, b, c in
                          zip(sig1[i * window_size:(i + 1) * window_size], sig2[i * window_size:(i + 1) * window_size],
                              sig3[i * window_size:(i + 1) * window_size])]
                sio.savemat(os.path.join(window_sim_dir, name + '_noise1' +'_mix' + str(i)), {'data': record})
                for elem in signals:
                    if ('fecg1' in elem or 'mecg' in elem):
                        data = loadmat(os.path.join(save_mat_dir, elem))['data'][i * window_size:(i + 1) * window_size]
                        sio.savemat(os.path.join(window_sim_dir, elem + str(i)), {'data': data})
        elif size == 4:
            sig1 = loadmat(os.path.join(save_mat_dir,signals[0]))['data'][num_of_signal_to_remove:]
            sig2 = loadmat(os.path.join(save_mat_dir,signals[1]))['data'][num_of_signal_to_remove:]
            sig3 = loadmat(os.path.join(save_mat_dir,signals[2]))['data'][num_of_signal_to_remove:]
            sig4 = loadmat(os.path.join(save_mat_dir,signals[3]))['data'][num_of_signal_to_remove:]
            for i in range(number_of_window):
                record = [a + b + c + d for a, b, c, d in zip(sig1[i*window_size:(i+1)*window_size], sig2[i*window_size:(i+1)*window_size], sig3[i*window_size:(i+1)*window_size], sig4[i*window_size:(i+1)*window_size])]
                sio.savemat(os.path.join(window_sim_dir, name + '_noise2' + '_mix' + str(i)), {'data': record})
                for elem in signals:
                    if ('fecg1' in elem or 'mecg' in elem):
                        data = loadmat(os.path.join(save_mat_dir, elem))['data'][i * window_size:(i + 1) * window_size]
                        sio.savemat(os.path.join(window_sim_dir, elem + str(i)),{'data': data})

        elif (size == 5):
            sig1 = loadmat(os.path.join(save_mat_dir, signals[0]))['data'][num_of_signal_to_remove:]
            sig2 = loadmat(os.path.join(save_mat_dir, signals[1]))['data'][num_of_signal_to_remove:]
            sig3 = loadmat(os.path.join(save_mat_dir, signals[2]))['data'][num_of_signal_to_remove:]
            sig4 = loadmat(os.path.join(save_mat_dir, signals[3]))['data'][num_of_signal_to_remove:]
            sig5 = loadmat(os.path.join(save_mat_dir, signals[4]))['data'][num_of_signal_to_remove:]
            for i in range(number_of_window):
                record = [a + b + c + d + e for a, b, c, d, e in zip(sig1[i*window_size:(i+1)*window_size], sig2[i*window_size:(i+1)*window_size], sig3[i*window_size:(i+1)*window_size], sig4[i*window_size:(i+1)*window_size], sig5[i*window_size:(i+1)*window_size])]
                # print(len(record))
                sio.savemat(os.path.join(window_sim_dir, name + '_noise3' + '_mix' + str(i)), {'data': record})
                for elem in signals:
                    if ('fecg1' in elem or 'mecg' in elem):
                        data = loadmat(os.path.join(save_mat_dir, elem))['data'][i * window_size:(i + 1) * window_size]
                        # print(len(data))
                        sio.savemat(os.path.join(window_sim_dir, elem + str(i)),{'data':data})

        else:
            print(signals)
            print('ciao')
