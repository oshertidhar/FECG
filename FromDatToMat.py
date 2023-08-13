import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
from HelpFunctions import *

PHYSIONET_PATH = "../physionet.org/files/fecgsyndb/1.0.0"
ALL_SIMULATED_DATA_MAT = "simulated_signals_mat_l1to2_baseline_and_c0to2_11_Oct_22_Ch19_21_23"

dir_path = os.path.dirname(os.path.realpath(__file__))
physionet_path = os.path.join(dir_path,PHYSIONET_PATH)
save_mat_dir = os.path.join(dir_path,ALL_SIMULATED_DATA_MAT)


if __name__ == '__main__':
    # Transforming from dat to mat
    #channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    channels=[19,21,23]
    for dirname in os.listdir(physionet_path):
        if ".html"  in dirname:
            continue
        path_1 = os.path.join(physionet_path,dirname)
        for subdirname in os.listdir(path_1):
            if ".html" in dirname:
                continue
            if ".html" in subdirname:
                continue
            path_2 = os.path.join(path_1,subdirname)
            for filename in os.listdir(path_2):
                if not(filename.endswith(".dat")):
                    continue
                if "l1" not in filename and "l2" not in filename:
                    continue
                if "c3" in filename or "c4" in filename or "c5" in filename:
                    continue

                print(filename[:len(filename)-4])
                print(filename[:19])
                for i in channels:
                    print(i)
                    record,fields = wfdb.rdsamp(os.path.join(path_2,filename[:len(filename)-4]), channels=[i]) #  TODO: change to channel 28
                    #print(fields)
                    sio.savemat(os.path.join(save_mat_dir,filename[:19]+'ch'+str(i)+filename[19:len(filename)-4]),{'data': record})
