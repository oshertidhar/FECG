import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
import random
from HelpFunctions import *

ALL_SIMULATED_DATA_MAT = "simulated_signals_mat"
WINDOWED_SIMULATED_SIGNAL = "RefactorDataset"

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
        signals.sort()
        print(signals)
        window_size = 1024
        number_of_window = 73
        sigma = 0.5 #random.uniform(0.2,0.5)

        if size == 2:
            for i in range(number_of_window):
                fecg = loadmat(os.path.join(save_mat_dir, signals[0]))['data'][i*window_size:(i+1)*window_size]
                mecg = loadmat(os.path.join(save_mat_dir, signals[1]))['data'][i*window_size:(i+1)*window_size]
                mix = [a + b for a, b in zip(fecg, mecg)]
                abs_signal = [abs(mecg[i] + sigma * fecg[i]) for i in range(len(mecg))]
                max_signal = max(abs_signal)
                fecg_new = [(sample / max_signal) * sigma for sample in fecg]
                mecg_new = [sample / max_signal for sample in mecg]
                mix_new = [a + b for a, b in zip(fecg_new,mecg_new)]
                sio.savemat(os.path.join(window_sim_dir, signals[0] + str(i)), {'data':fecg_new})
                sio.savemat(os.path.join(window_sim_dir, signals[1] + str(i)), {'data':mecg_new})
                sio.savemat(os.path.join(window_sim_dir, name + '_noise0' + '_mix' + str(i)), {'data':mix_new})
                """fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1)
                ax1.plot(mecg)
                ax1.plot(mecg_new)
                ax2.plot(fecg)
                ax2.plot(fecg_new)
                ax3.plot(mix)
                ax4.plot(mix_new)
                plt.show()
                plt.close()"""

        elif size == 4:
            for i in range(number_of_window):
                fecg = loadmat(os.path.join(save_mat_dir, signals[0]))['data'][i * window_size:(i + 1) * window_size]
                mecg = loadmat(os.path.join(save_mat_dir, signals[1]))['data'][i * window_size:(i + 1) * window_size]
                noise1 = loadmat(os.path.join(save_mat_dir, signals[2]))['data'][i * window_size:(i + 1) * window_size]
                noise2 = loadmat(os.path.join(save_mat_dir, signals[3]))['data'][i * window_size:(i + 1) * window_size]
                mix = [a + b + c + d for a, b, c, d in zip(fecg, mecg, noise1, noise2)]
                abs_signal = [abs(mecg[i] + sigma * fecg[i] + noise1[i] + noise2[i]) for i in range(len(mecg))]
                max_signal = max(abs_signal)
                fecg_new = [(sample / max_signal) * sigma for sample in fecg]
                mecg_new = [sample / max_signal for sample in mecg]
                noise1_new = [(sample / max_signal) for sample in noise1]
                noise2_new = [(sample / max_signal) for sample in noise2]
                mix_new = [a + b + c + d for a, b, c, d in zip(fecg_new,mecg_new,noise1_new,noise2_new)]
                sio.savemat(os.path.join(window_sim_dir, signals[0] + str(i)), {'data': fecg_new})
                sio.savemat(os.path.join(window_sim_dir, signals[1] + str(i)), {'data': mecg_new})
                #sio.savemat(os.path.join(window_sim_dir, signals[2] + str(i)), {'data': noise1_new})
                #sio.savemat(os.path.join(window_sim_dir, signals[3] + str(i)), {'data': noise2_new})
                sio.savemat(os.path.join(window_sim_dir, name + '_noise2' + '_mix' + str(i)),
                            {'data': mix_new})
                """fig, (ax1, ax2, ax3, ax4,ax5,ax6) = plt.subplots(6, 1)
                ax1.plot(mecg)
                ax1.plot(mecg_new)
                ax2.plot(fecg)
                ax2.plot(fecg_new)
                ax3.plot(mix)
                ax4.plot(mix_new)
                ax5.plot(noise1)
                ax5.plot(noise1_new)
                ax6.plot(noise2)
                ax6.plot(noise2_new)
                plt.show()
                plt.close()"""


        elif (size == 5):
            for i in range(number_of_window):
                fecg = loadmat(os.path.join(save_mat_dir, signals[0]))['data'][i * window_size:(i + 1) * window_size]
                mecg = loadmat(os.path.join(save_mat_dir, signals[1]))['data'][i * window_size:(i + 1) * window_size]
                noise1 = loadmat(os.path.join(save_mat_dir, signals[2]))['data'][i * window_size:(i + 1) * window_size]
                noise2 = loadmat(os.path.join(save_mat_dir, signals[3]))['data'][i * window_size:(i + 1) * window_size]
                noise3 = loadmat(os.path.join(save_mat_dir, signals[4]))['data'][i * window_size:(i + 1) * window_size]
                mix = [a + b + c + d + e for a, b, c, d, e in zip(fecg, mecg, noise1, noise2,noise3)]
                abs_signal = [abs(mecg[i] + sigma * fecg[i] + noise1[i] + noise2[i] + noise3[i]) for i in range(len(mecg))]
                max_signal = max(abs_signal)
                fecg_new = [(sample / max_signal) * sigma for sample in fecg]
                mecg_new = [sample / max_signal for sample in mecg]
                noise1_new = [(sample / max_signal) for sample in noise1]
                noise2_new = [(sample / max_signal) for sample in noise2]
                noise3_new = [(sample / max_signal) for sample in noise2]
                mix_new = [a + b + c + d + e for a, b, c, d,e in zip(fecg_new, mecg_new, noise1_new, noise2_new,noise3_new)]
                sio.savemat(os.path.join(window_sim_dir, signals[0] + str(i)), {'data': fecg_new})
                sio.savemat(os.path.join(window_sim_dir, signals[1] + str(i)), {'data': mecg_new})
                #sio.savemat(os.path.join(window_sim_dir, signals[2] + str(i)), {'data': noise1_new})
                #sio.savemat(os.path.join(window_sim_dir, signals[3] + str(i)), {'data': noise2_new})
                #sio.savemat(os.path.join(window_sim_dir, signals[4] + str(i)), {'data': noise3_new})
                sio.savemat(os.path.join(window_sim_dir, name + '_noise3' + '_mix' + str(i)),{'data': mix_new})
                """fig, (ax1, ax2, ax3, ax4, ax5, ax6,ax7) = plt.subplots(7, 1)
                ax1.plot(mecg)
                ax1.plot(mecg_new)
                ax2.plot(fecg)
                ax2.plot(fecg_new)
                ax3.plot(mix)
                ax4.plot(mix_new)
                ax5.plot(noise1)
                ax5.plot(noise1_new)
                ax6.plot(noise2)
                ax6.plot(noise2_new)
                ax7.plot(noise3)
                ax7.plot(noise3_new)
                plt.show()
                plt.close()"""


        else:
            print(signals)
            print('ciao')
