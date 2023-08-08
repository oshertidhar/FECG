import os
from tkinter import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import fft,dct,fftfreq,rfft
import numpy as np

LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses/Overfit/NR_10/TrainAllSet")
#LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses/Ch0/250Epochs")

for filename in os.listdir(LOSSES):  # present the fecg outputs
    path = os.path.join(LOSSES, filename)
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(np.load(path))
    ax1.set_ylabel(filename)
    plt.show()
    plt.close()










