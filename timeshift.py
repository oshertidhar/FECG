import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


ECG_SIGNALS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGSignals")


def get_max_amplitude_in_signal(signal):
    max_amplitude = 0
    count = 0
    signal_size = len(signal)
    for i in range(0,len(signal),50):
        for j in range(i,i+50):
            if (j<signal_size and signal[j] > max_amplitude):
                max_amplitude = max_amplitude + signal[j]
    return max_amplitude

def get_values_above_threshold(signal, threshold):
    values = []
    count = 0
    for index, sample in enumerate(signal):
        if (sample > threshold):
            if (count != 0 and values[count - 1] == (index - 1)):
                if signal[index-1] < sample:
                    del values[-1]
                    count = count - 1
                    values.append(index)
                    count = count + 1
            else:
                values.append(index)
                count = count + 1
    return values


def compute_time_shifting(original_signal, output_signal):
    max_amplitude_original = get_max_amplitude_in_signal(original_signal)
    max_amplitude_output = get_max_amplitude_in_signal(output_signal)
    original_threshold = max_amplitude_original / 2
    output_threshold = max_amplitude_output / 2
    print(original_threshold)
    print(output_threshold)
    original_indices_above_threshold = get_values_above_threshold(original_signal, original_threshold)
    output_indices_above_threshold = get_values_above_threshold(output_signal, output_threshold)
    print(original_indices_above_threshold)
    print(output_indices_above_threshold)
    sum_time_differences = 0
    for index in range(len(original_indices_above_threshold)):
        print(original_indices_above_threshold[index])
        print(output_indices_above_threshold[index])
        sum_time_differences = sum_time_differences + abs(
            original_indices_above_threshold[index] - output_indices_above_threshold[index])
    return sum_time_differences / len(original_indices_above_threshold)

def find_range(signal, value):
    min_index = 0
    max_index = len(signal) - 1
    if (value - 5 > 0):
        min_index = value-5
    if (value + 5 < max_index):
        max_index = value + 5
    index = np.argmax(signal[min_index:max_index])
    if (index >= 4):
        return value - (5 - index)
    else:
        return value + (index - 5)

if __name__ == "__main__":
    for filename in os.listdir(ECG_SIGNALS):
        if "fecg" in filename:
            path = os.path.join(ECG_SIGNALS, filename)
            print(path)
            number_file = filename.index("g") + 1
            end_path = filename[number_file:]
            path_label = os.path.join(ECG_SIGNALS, "label_f" + end_path)
            original_signal = np.load(path)
            output_signal = np.load(path_label)
            # plt.plot(original_signal)
            # plt.show()
            # plt.close()
            # plt.plot(output_signal)
            # plt.show()
            # plt.close()
            max_amplitude_original = get_max_amplitude_in_signal(original_signal)
            max_amplitude_output = get_max_amplitude_in_signal(output_signal)

            peaks, _ = find_peaks(original_signal,height=max_amplitude_original/4, distance=50,prominence= max_amplitude_original/2)
            peaks_label, _ = find_peaks(output_signal,height=max_amplitude_output/4, distance=50,prominence= max_amplitude_output/2)

            peak_dir = np.diff(original_signal)
            peak_lab_dir = np.diff(output_signal)
            max_thresh1 = get_max_amplitude_in_signal(peak_dir)
            max_thresh2 = get_max_amplitude_in_signal(peak_lab_dir)

            peaks1, _ = find_peaks(peak_dir, height=max_thresh1 / 5, distance=50)
            peaks2, _ = find_peaks(peak_lab_dir, height=max_thresh2 / 5, distance=50)

            print(peaks1)
            new_peaks = list()
            for i in peaks1:
                new_peaks.append(find_range(original_signal,i))
            print(new_peaks)
            print(peaks2)
            new_peaks2 = list()
            for i in peaks2:
                new_peaks2.append(find_range(output_signal, i))
            print(new_peaks2)

            #print(peaks)
            #print(peaks1)
            #print(peaks_label)
            #print(peaks2)

            peaks_final = [elem for elem in peaks1 if original_signal[elem] >= max_amplitude_original/4 ]
            peaks_label_final = [elem for elem in peaks2 if output_signal[elem] >= max_amplitude_output/4 ]
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
            ax1.plot(original_signal)
            ax1.plot(new_peaks, original_signal[new_peaks], "x")
            ax2.plot(peak_dir)
            ax3.plot(output_signal)
            ax3.plot(new_peaks2, output_signal[new_peaks2], "x")
            ax4.plot(peak_lab_dir)
            plt.show()
            plt.close()
            #avg_time_shifting = compute_time_shifting(original_signal, output_signal)
            #print(avg_time_shifting)