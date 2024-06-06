import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, buttord
import time



def butter_highpass(fc, fs, ft, rp, rs):
    fn = fs/2
    wp = np.array([fc + ft/2]) / fn
    ws = np.array([fc - ft/2]) / fn
    order, Wn = buttord(wp, ws, rp, rs)
    b, a = butter(order, Wn, btype='high', analog=False)
    return b, a

def butter_lowpass(fc, fs, ft, rp, rs):
    fn = fs/2
    wp = np.array([fc - ft/2]) / fn
    ws = np.array([fc + ft/2]) / fn
    order, Wn = buttord(wp, ws, rp, rs)
    b, a = butter(order, Wn, btype='low', analog=False)
    return b, a

def apply_filter(data, fc, fs, ft, rp, rs, filter_type='high'):
    if filter_type == 'high':
        b, a = butter_highpass(fc, fs, ft, rp, rs)
    elif filter_type == 'low':
        b, a = butter_lowpass(fc, fs, ft, rp, rs)
    y = filtfilt(b, a, data)
    return y

def median_filter(data, kernel_size=5):
    return np.array([np.median(data[i:i+kernel_size]) for i in range(len(data)-kernel_size+1)])

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main() -> None:
    ser = serial.Serial('COM3', 9600)
    time.sleep(2)  

    duration = 10  
    sr = 15000 
    ft = 250
    highpass_cutoff = 3500 
    lowpass_cutoff = 5000  
    median_filter_kernel_size = 5 
    highpass_cutoff2 = 2500
    fn = sr / 2
    rp = 0.3
    rs = 35

    wp = np.array([highpass_cutoff + ft/2]) / fn
    ws = np.array([highpass_cutoff - ft/2]) / fn

    data = []

    order = buttord(wp, ws, rp, rs)

    start_time = time.time()
    while time.time() - start_time < duration:
        if ser.in_waiting > 0:
            try:
                value = int(ser.readline().strip())
                data.append(value)
            except ValueError:
                pass

    ser.close()


    data = np.array(data)

    data_highpassed = apply_filter(data, highpass_cutoff, sr, ft, rp, rs, filter_type='high')

    data_lowpassed = apply_filter(data_highpassed, lowpass_cutoff, sr, ft, rp, rs, filter_type='low')

    data_median_filtered = median_filter(data_lowpassed, kernel_size=median_filter_kernel_size)

    # rectified_signal = np.abs(data_median_filtered - np.mean(data_median_filtered))
    rectified_signal = np.abs(data_median_filtered)

    rectified_highpass_signal = apply_filter(rectified_signal, highpass_cutoff2, sr, ft, rp, rs, filter_type='high')

    smoothed_signal = moving_average(rectified_highpass_signal, window_size = 10)

    plt.figure(figsize=(12, 10))

    plt.subplot(6, 1, 1)
    plt.plot(data, label='Raw EMG')
    plt.title('Raw EMG')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(data_highpassed, label='Highpass', color='green')
    plt.title('Highpass')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(data_lowpassed, label='Lowpass', color='blue')
    plt.title('Lowpass')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.plot(rectified_signal, label='Median Filter')
    plt.title('Median Filter')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.plot(np.abs(rectified_highpass_signal), label='Highpass + Median')
    plt.title('Highpass + Median')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(smoothed_signal, label='Moving Average')
    plt.title('Moving Average')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()