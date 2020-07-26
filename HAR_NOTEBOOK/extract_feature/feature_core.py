import numpy as np
from .feature_time import Feature_time
from .feature_fft import Feature_fft

def get_feature(arr):
    feature_list = list()
    feature_time = Feature_time(arr).time_all()
    feature_list.extend(feature_time)
    feature_fft = Feature_fft(arr).fft_all()
    feature_list.extend(feature_fft)
    return feature_list


def sequence_feature(seq, win_size, step_size):
    if win_size == 0:
        return np.asarray(get_feature(seq))
    window_size = win_size
    step_size = step_size
    r = len(seq)
    feature_mat = list()

    j = 0
    while j < r - step_size:
        window = seq[j:j + window_size]
        win_feature = get_feature(window)
        feature_mat.append(win_feature)
        j += step_size
    return np.asarray(feature_mat)
