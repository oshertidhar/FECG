import os
from tkinter import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import fft,dct,fftfreq,rfft
import numpy as np

DATA_FOLDER = "real_signals_nifecgdb_2048_windows_07_Mar_102_abd3"
ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Osher_TestReal/real_signals_nifecgdb_2048_windows_07_Mar_102_abd3")

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, ECG_OUTPUTS_TEST_REAL)

# w = evt.widget
# index = int(w.curselection()[0])
# value = w.get(index)
for filename in os.listdir(ECG_OUTPUTS_TEST_REAL):  # present the fecg outputs
    if "label_ecg1" in filename and "2" in filename and not "mat" in filename:
        print(filename)
        path_label = os.path.join(ECG_OUTPUTS_TEST_REAL, filename)
        number_file = filename.index("g") + 1
        end_path = filename[number_file:]
        path = os.path.join(ECG_OUTPUTS_TEST_REAL, "ecg" + end_path)
        mecg_label = os.path.join(ECG_OUTPUTS_TEST_REAL, "mecg" + end_path)
        fecg_label = os.path.join(ECG_OUTPUTS_TEST_REAL, "fecg" + end_path)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        ax1.plot(np.load(path)[0])
        ax1.set_ylabel("ECG")
        ax2.plot(np.load(path_label)[0])
        ax2.set_ylabel("LABEL ECG")
        ax3.plot(np.load(mecg_label)[0])
        ax3.set_ylabel("MECG")
        ax4.plot(np.load(fecg_label)[0])
        ax4.set_ylabel("FECG")
        fig.suptitle(filename)
        plt.show()
        plt.close()





