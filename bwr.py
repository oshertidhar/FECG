from math import sqrt
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from SignalPreprocessing.data_preprocess_function import *

# Daubechies 4 Constant
c0 = (1+sqrt(3))/(4*sqrt(2))
c1 = (3+sqrt(3))/(4*sqrt(2))
c2 = (3-sqrt(3))/(4*sqrt(2))
c3 = (1-sqrt(3))/(4*sqrt(2))

def conv(x, h):
    """ Perform the convolution operation between two input signals. The output signal length
    is the sum of the lenght of both input signal minus 1."""
    length = len(x) + len(h) - 1
    y = [0]*length

    for i in range(len(y)):
        for j in range(len(h)):
            if i-j >= 0 and i-j < len(x):
                y[i] += h[j] * x[i-j]

    return y

def db4_dec(x, level):
    """ Perform the wavelet decomposition to signal x with Daubechies order 4 basis function as many as specified level"""

    # Decomposition coefficient for low pass and high pass
    lpk = [c0, c1, c2, c3]
    hpk = [c3, -c2, c1, -c0]

    result = [[]]*(level+1)
    x_temp = x[:]
    for i in range(level):
        lp = conv(x_temp, lpk)
        hp = conv(x_temp, hpk)

        # Downsample both output by half
        lp_ds=[0]*int((len(lp)/2))
        hp_ds=[0]*int((len(hp)/2))
        for j in range(len(lp_ds)):
            lp_ds[j] = lp[2*j+1]
            hp_ds[j] = hp[2*j+1]

        result[level-i] = hp_ds
        x_temp = lp_ds[:]

    result[0] = lp_ds
    return result

def db4_rec(signals, level):
    """ Perform reconstruction from a set of decomposed low pass and high pass signals as deep as specified level"""

    # Reconstruction coefficient
    lpk = [c3, c2, c1, c0]
    hpk = [-c0, c1, -c2, c3]

    cp_sig = signals[:]
    for i in range(level):
        lp = cp_sig[0]
        hp = cp_sig[1]

        # Verify new length
        length = 0
        if len(lp) > len(hp):
            length = 2*len(hp)
        else:
            length = 2*len(lp)

        # Upsampling by 2
        lpu = [0]*(length+1)
        hpu = [0]*(length+1)
        index = 0
        for j in range(length+1):
            if j%2 != 0:
                lpu[j] = lp[index]
                hpu[j] = hp[index]
                index += 1

        # Convolve with reconstruction coefficient
        lpc = conv(lpu, lpk)
        hpc = conv(hpu, hpk)

        # Truncate the convolved output by the length of filter kernel minus 1 at both end of the signal
        lpt = lpc[3:-3]
        hpt = hpc[3:-3]

        # Add both signals
        org = [0]*len(lpt)
        for j in range(len(org)):
            org[j] = lpt[j] + hpt[j]

        if len(cp_sig) > 2:
            cp_sig = [org]+cp_sig[2:]
        else:
            cp_sig = [org]

    return cp_sig[0]

def calcEnergy(x):
    """ Calculate the energy of a signal which is the sum of square of each points in the signal."""
    total = 0
    for i in x:
        total += i*i
    return total

def bwr(raw):
    """ Perform the baseline wander removal process against signal raw. The output of this method is signal with correct baseline
    and its baseline """
    en0 = 0
    en1 = 0
    en2 = 0
    n = 0

    curlp = raw[:]
    num_dec = 0
    last_lp = []
    count = 0
    while True:
        count +=1
        #print('Iterasi ke' + str(num_dec+1))
        #print(len(curlp))

        # Decompose 1 level
        [lp, hp] = db4_dec(curlp,1)

        # Shift and calculate the energy of detail/high pass coefficient
        en0 = en1
        en1 = en2
        en2 = calcEnergy(hp)
        #print(en2)

        # Check if we are in the local minimum of energy function of high-pass signal
        if ((en0 > en1 and en1 < en2) or (count > 16)):
            last_lp = curlp
            break

        curlp = lp[:]
        num_dec = num_dec+1

    # Reconstruct the baseline from this level low pass signal up to the original length
    base = last_lp[:]
    for i in range(num_dec):
        base = db4_rec([base,[0]*len(base)], 1)

    # Correct the original signal by subtract it with its baseline
    ecg_out = [0]*len(raw)
    for i in range(len(raw)):
        ecg_out[i] =  raw[i] - base[i]

    return (base, ecg_out)

if __name__ == '__main__':
    SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabase")
    SIMULATED_DATASET_NOISE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabaseMat")
    BWR_SIGNALS = "bwr_signals"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bwr_path = os.path.join(dir_path, BWR_SIGNALS)

    if not os.path.exists(BWR_SIGNALS):
        os.mkdir(BWR_SIGNALS)
    for filename in os.listdir(SIMULATED_DATASET):
        if 'noise3' not in filename:
            continue
        print(filename)
        current_signal = loadmat(os.path.join(SIMULATED_DATASET, filename))['data']
        path = filename[:filename.find('noise')]
        mecg_path = path + 'mecg'
        fecg_path = path + 'fecg1'
        noise3_path = path + 'noise3'
        noise2_path = path + 'noise2'
        noise1_path = path + 'noise1'
        index = int(filename[filename.find('x') + 1:])
        print(index)
        print(noise3_path)
        print(mecg_path)
        print(fecg_path)

        mecg = loadmat(os.path.join(SIMULATED_DATASET_NOISE, mecg_path))['data'][index * 1024: (index + 1) * 1024]
        fecg = loadmat(os.path.join(SIMULATED_DATASET_NOISE, fecg_path))['data'][index * 1024: (index + 1) * 1024]
        noise3 = loadmat(os.path.join(SIMULATED_DATASET_NOISE, noise3_path))['data'][index * 1024: (index + 1) * 1024]
        noise1 = loadmat(os.path.join(SIMULATED_DATASET_NOISE, noise1_path))['data'][index * 1024: (index + 1) * 1024]
        noise2 = loadmat(os.path.join(SIMULATED_DATASET_NOISE, noise2_path))['data'][index * 1024: (index + 1) * 1024]
        yf1, freq1, t = transformation('fft', noise1)
        yf2, freq2, t = transformation('fft', noise2)
        yf3, freq3, t = transformation('fft', noise3)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(freq1, np.abs(yf1))
        ax1.set_title("1")
        ax1.set_xlim(0)
        ax1.set_ylim([0,30])
        ax2.plot(freq2, np.abs(yf2))
        ax2.set_title("2")
        ax2.set_xlim(0)
        ax2.set_ylim([0, 30])
        ax3.plot(freq3, np.abs(yf3))
        ax3.set_title("3")
        ax3.set_xlim(0)
        ax3.set_ylim([0, 30])
        plt.show()
        plt.close()
        """yfm, freqm, t = transformation('fft', mecg)
        yff, freqf, t = transformation('fft', fecg)
        yfn, freqn, t = transformation('fft', noise3)
        yfe, freqe, t = transformation('fft', current_signal)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        ax1.plot(freqe,np.abs(yfe))
        ax1.set_title("Ecg")
        ax1.set_xlim(0)
        ax2.plot(freqm,np.abs(yfm))
        ax2.set_title("Mecg")
        ax2.set_xlim(0)
        ax3.plot(freqf,np.abs(yff))
        ax3.set_title("Fecg")
        ax3.set_xlim(0)
        ax4.plot(freqn,np.abs(yfn))
        ax4.set_title("Bw")
        ax4.set_xlim(0)
        plt.show()
        plt.close()"""

    """for filename in os.listdir(SIMULATED_DATASET):
        if "noise2" not in filename:
            #sio.savemat(os.path.join(bwr_path, filename),
            #            {'data': loadmat(os.path.join(SIMULATED_DATASET, filename))['data']})
            continue
        print(filename)
        current_signal = loadmat(os.path.join(SIMULATED_DATASET,filename))['data']
        noise3_path = filename[:filename.find('_mix')]
        index = int(filename[filename.find('x')+1:])
        print(index)
        print(noise3_path)
        noise_signal = loadmat(os.path.join(SIMULATED_DATASET_NOISE,noise3_path))['data']
        baseline,ecg_out = bwr(current_signal)
        #sio.savemat(os.path.join(bwr_path,filename), {'data': ecg_out})

        fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1)
        ax1.plot(current_signal, label="BeforePreprocess")
        ax2.plot(ecg_out, label="AfterPreprocess")
        ax3.plot(baseline, label="Baseline")
        ax4.plot(noise_signal[index * 1024: (index + 1) * 1024] , label="BaselineReal")
        plt.show()
        plt.close()
        

        plt.subplot(2, 1, 1)
        plt.plot(current_signal, 'b-')
        plt.plot(baseline, 'r-')

        plt.subplot(2, 1, 2)
        plt.plot(ecg_out, 'b-')
        plt.show()"""

