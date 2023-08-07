import os
from tkinter import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import fft,dct,fftfreq,rfft
import numpy as np
import torch

ECG_OUTPUTS_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTrainReal/NR_02/best_model_LOSS/ContTrain")

for filename in os.listdir(ECG_OUTPUTS_REAL):  # present the fecg outputs
    #if "mecg1" in filename and "epoch" in filename:
    if "ecg_all" in filename:
        print(filename)
        number_file = filename.index("g") + 1
        end_path = filename[number_file + 4:]

        ecg_all = os.path.join(ECG_OUTPUTS_REAL, "ecg_all" + end_path)
        mecg_label = os.path.join(ECG_OUTPUTS_REAL, "label_m" + end_path)
        mecg = os.path.join(ECG_OUTPUTS_REAL, "mecg" + end_path)
        fecg = os.path.join(ECG_OUTPUTS_REAL, "fecg" + end_path)
        fecg_shifted = os.path.join(ECG_OUTPUTS_REAL, "fecg_shifted" + end_path)

        #fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
        ax1.plot(np.load(mecg_label))
        ax1.set_ylabel("TECG (in)")
        ax2.plot(np.load(mecg))
        ax2.set_ylabel("MECG (out)")
        ax3.plot(np.load(ecg_all))
        ax3.set_ylabel("AECG (in)")
        ax4.plot(np.load(fecg_shifted))
        ax4.set_ylabel("FECG (out)")
        #ax5.plot(np.load(fecg))
        #ax5.set_ylabel("FECG")
        plt.show()
        plt.close()










