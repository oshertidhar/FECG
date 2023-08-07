from scipy.io import savemat
import numpy as np
import glob
import os
import scipy.io

def npy2mat():
    #npzFiles = glob.glob("./ECGOutputsTrainReal/NR_10/TrainAllSet/*.npy")
    npzFiles = glob.glob("../MasterAECG+MECG/ECGOutputs/Ch_19_21_23/m2IsNoisedAndShiftedBy1/310_320/label*.npy")
    for f in npzFiles:
       fm = os.path.splitext(f)[0]+'.mat'
       d = np.load(f)
       savemat(fm, {"data":d})
       print('generated ', fm, 'from', f)


def mat2npy():
    shifted_mat = scipy.io.loadmat('./ECGOutputsTrainReal/NR_02/best_model_LOSS/ContTrain/fecg_shifted_win0_batch1.mat')['shifted_mat']
    np.save('./ECGOutputsTrainReal/fecg0_batch0_shifted.npy', shifted_mat)


npy2mat()
#mat2npy()
