import numpy as np
import scipy as sp
from scipy.signal import medfilt, butter, filtfilt
# 提取重力
from scipy.fftpack import fft
from scipy.fftpack import ifft

# 滤波函数
def filter_func(sign):
    array = np.array(sign)
    med_filtered = medfilt(array, kernel_size=3)   # 中值
    b, a = butter(3, 0.6, 'lowpass')
    filtered = filtfilt(b, a, med_filtered)   # 低通
    return filtered

sampling_freq = 17
nyq = sampling_freq/float(2)
freq1 = 0.3
freq2 = 20

# 提取重力
def components_selection_one_signal(r_signal):
    t_signal = np.array(r_signal)
    t_signal_length = len(t_signal)
    f_signal = fft(t_signal)
    freqs = np.array(sp.fftpack.fftfreq(t_signal_length, d=1 / float(sampling_freq)))
    f_DC_signal = []  # 重力
    f_body_signal = []  # 人体
    f_noise_signal = []  # 噪音

    for i in range(len(freqs)):
        freq = freqs[i]
        value = f_signal[i]

        # 重力
        if abs(freq) > 0.3:
            f_DC_signal.append(float(0))
        else:
            f_DC_signal.append(value)

        # 噪音
        if (abs(freq) <= 20):
            f_noise_signal.append(float(0))
        else:
            f_noise_signal.append(value)

        # 人体
        if (abs(freq) <= 0.3 or abs(freq) > 20):
            f_body_signal.append(float(0))
        else:
            f_body_signal.append(value)

    t_DC_component = ifft(np.array(f_DC_signal)).real
    t_body_component = ifft(np.array(f_body_signal)).real
    t_noise = ifft(np.array(f_noise_signal)).real

    total_component = t_signal - t_noise
    return (total_component, t_DC_component, t_body_component, t_noise)