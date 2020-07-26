from flask import Flask, json, request
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from scipy.signal import medfilt
import joblib
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import stats
from scipy import signal


app = Flask(__name__)


def process_data(signal, flag):
    array = np.array(signal)
    med_filtred = medfilt(array, kernel_size=3)
    if flag:  # acc列(x,y,z)
        _, grav_acc, body_acc, _ = components_selection_one_signal(med_filtred)
        return (grav_acc, body_acc)
    else:  # gyr列(x,y,z)
        _, _, body_gyro, _ = components_selection_one_signal(med_filtred)
        return body_gyro


def components_selection_one_signal(t_signal):
    t_signal = np.array(t_signal)
    t_signal_length = len(t_signal)
    f_signal = fft(t_signal)
    freqs = np.array(sp.fftpack.fftfreq(t_signal_length, d=1/17))
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


# 时域特征
class Feature_time(object):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def time_mean(self):
        return np.mean(self.data)

    def time_var(self):
        return np.var(self.data)

    def time_std(self):
        return np.std(self.data)

    def time_mode(self):
        return float(scipy.stats.mode(self.data, axis=None)[0])

    def time_max(self):
        return np.max(self.data)

    def time_min(self):
        return np.min(self.data)

    def time_over_zero(self):
        return len(self.data[self.data > 0])

    def time_range(self):
        return self.time_max() - self.time_min()

    def time_all(self):
        feature_all = list()
        feature_all.append(self.time_mean())
        feature_all.append(self.time_var())
        feature_all.append(self.time_std())
        feature_all.append(self.time_mode())
        feature_all.append(self.time_max())
        feature_all.append(self.time_min())
        feature_all.append(self.time_over_zero())
        feature_all.append(self.time_range())
        return feature_all


# 频域特征
class Feature_fft(object):
    def __init__(self, sequence_data):
        self.data = sequence_data
        fft_trans = np.abs(np.fft.fft(sequence_data))
        self.dc = fft_trans[0]
        self.freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
        self._freq_sum_ = np.sum(self.freq_spectrum)

    def fft_dc(self):
        return self.dc[0]

    def fft_mean(self):
        return np.mean(self.freq_spectrum)

    def fft_var(self):
        return np.var(self.freq_spectrum)

    def fft_std(self):
        return np.std(self.freq_spectrum)

    def fft_entropy(self):
        pr_freq = self.freq_spectrum * 1.0 / self._freq_sum_
        entropy = -1 * np.sum([np.log2(p) * p for p in pr_freq])
        return entropy

    def fft_energy(self):
        return np.sum(self.freq_spectrum ** 2) / len(self.freq_spectrum)

    def fft_skew(self):
        fft_mean, fft_std = self.fft_mean(), self.fft_std()

        return np.mean([0 if fft_std == 0 else np.power((x - fft_mean) / fft_std, 3)
                        for x in self.freq_spectrum])

    def fft_kurt(self):
        fft_mean, fft_std = self.fft_mean(), self.fft_std()
        return np.mean([0 if fft_std == 0 else np.power((x - fft_mean) / fft_std, 4) - 3
                        for x in self.freq_spectrum])

    def fft_max(self):
        idx = np.argmax(self.freq_spectrum)
        return idx, self.freq_spectrum[idx]

    def fft_topk_freqs(self, top_k=None):
        idxs = np.argsort(self.freq_spectrum)
        if top_k == None:
            top_k = len(self.freq_spectrum)
        return idxs[:top_k], self.freq_spectrum[idxs[:top_k]]

    def fft_shape_mean(self):
        shape_sum = np.sum([x * self.freq_spectrum[x]
                            for x in range(len(self.freq_spectrum))])
        return 0 if self._freq_sum_ == 0 else shape_sum * 1.0 / self._freq_sum_

    def fft_shape_std(self):
        shape_mean = self.fft_shape_mean()
        var = np.sum([0 if self._freq_sum_ == 0 else np.power((x - shape_mean), 2) * self.freq_spectrum[x]
                      for x in range(len(self.freq_spectrum))]) / self._freq_sum_
        return np.sqrt(var)

    def fft_shape_skew(self):
        shape_mean = self.fft_shape_mean()
        return np.sum([np.power((x - shape_mean), 3) * self.freq_spectrum[x]
                       for x in range(len(self.freq_spectrum))]) / self._freq_sum_

    def fft_shape_kurt(self):
        shape_mean = self.fft_shape_mean()
        return np.sum([np.power((x - shape_mean), 4) * self.freq_spectrum[x] - 3
                for x in range(len(self.freq_spectrum))]) / self._freq_sum_

    def fft_all(self):
        feature_all = list()
        feature_all.append(self.fft_dc())
        feature_all.append(self.fft_shape_mean())
        feature_all.append(self.fft_shape_std() ** 2)
        feature_all.append(self.fft_shape_std())
        feature_all.append(self.fft_shape_skew())
        feature_all.append(self.fft_shape_kurt())
        feature_all.append(self.fft_mean())
        feature_all.append(self.fft_var())
        feature_all.append(self.fft_std())
        feature_all.append(self.fft_skew())
        feature_all.append(self.fft_kurt())
        return feature_all


def get_feature(arr):
    feature_list = list()
    feature_time = Feature_time(arr).time_all()
    feature_list.extend(feature_time)
    feature_fft = Feature_fft(arr).fft_all()
    feature_list.extend(feature_fft)
    return feature_list


# 提取特征核心函数
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


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    data = json.loads(request.values.get("data"))
    row = min(len(data["accXs"]), len(data["gyrXs"]), 50)
    accX = data["accXs"][-row:]
    accY = data["accYs"][-row:]
    accZ = data["accZs"][-row:]
    gyrX = data["gyrXs"][-row:]
    gyrY = data["gyrYs"][-row:]
    gyrZ = data["gyrZs"][-row:]
    feature_arr = np.zeros((row, 9))
    feature_arr[:, 0] = process_data(accX, True)[1]
    feature_arr[:, 1] = process_data(accY, True)[1]
    feature_arr[:, 2] = process_data(accZ, True)[1]
    feature_arr[:, 3] = process_data(accX, True)[0]
    feature_arr[:, 4] = process_data(accY, True)[0]
    feature_arr[:, 5] = process_data(accZ, True)[0]
    feature_arr[:, 6] = process_data(gyrX, False)
    feature_arr[:, 7] = process_data(gyrY, False)
    feature_arr[:, 8] = process_data(gyrZ, False)
    feature_ = pd.DataFrame()
    for i in [0, 1, 2, 6, 7, 8]:
        data = np.array(feature_arr[:, i]).reshape(row,1)
        a = sequence_feature(data, 32, 16)  # 19列特征  shape:(3,19)
        feature = pd.DataFrame(a)
        feature_ = pd.concat([feature_, feature], axis=1)  # 横向连接19个feature
    feature_['gra_x'] = np.full(feature_.shape[0], feature_arr[3][0])
    feature_['gra_y'] = np.full(feature_.shape[0], feature_arr[4][0])
    feature_['gra_z'] = np.full(feature_.shape[0], feature_arr[5][0])
    model = joblib.load('./forest_10.pkl')
    feature_matrix = np.array(feature_)
    result = model.predict(feature_matrix)
    print(result)
    if len(result) < 3:
        vote = result[-1]
    else:
        vote = stats.mode(result)[0][0]
    return str(vote),str(result)


def fliter_signal(sign, filter):  # input: one column
    array = np.array(sign)
    med_filtered = medfilt(array, kernel_size=3)
    b, a = signal.butter(3, filter, 'lowpass')
    med_filtered = signal.filtfilt(b, a, med_filtered)  
    return med_filtered


@app.route('/number', methods=['GET', 'POST'])
def number():
    dic = {'徒手侧平举': 0, '前后交叉跑': 0, '开合跳': 0, '深蹲': 0}  # 总
    sub_dic = {}               # 分
    action = ['徒手侧平举', '前后交叉跑', '开合跳', '深蹲']   # 动作列表
    data_new = {}
    all_result = []            # 存放预测值
    sub_result = {}
    data = json.loads(request.values.get("data"))
    # 重新计算，得到10次的预测值
    for i in range(1, 11):
        data_ = data[str(i)]
        row = min(len(data_["accXs"]), len(data_["gyrXs"]), 50)
        accX = data_["accXs"][-row:]
        accY = data_["accYs"][-row:]
        accZ = data_["accZs"][-row:]
        gyrX = data_["gyrXs"][-row:]
        gyrY = data_["gyrYs"][-row:]
        gyrZ = data_["gyrZs"][-row:]
        feature_arr = np.zeros((row, 9))
        feature_arr[:, 0] = process_data(accX, True)[1]
        feature_arr[:, 1] = process_data(accY, True)[1]
        feature_arr[:, 2] = process_data(accZ, True)[1]
        feature_arr[:, 3] = process_data(accX, True)[0]
        feature_arr[:, 4] = process_data(accY, True)[0]
        feature_arr[:, 5] = process_data(accZ, True)[0]
        feature_arr[:, 6] = process_data(gyrX, False)
        feature_arr[:, 7] = process_data(gyrY, False)
        feature_arr[:, 8] = process_data(gyrZ, False)
        feature_ = pd.DataFrame()
        for j in [0, 1, 2, 6, 7, 8]:
            qq = np.array(feature_arr[:, j]).reshape(row, 1)
            a = sequence_feature(qq, 32, 16)  # 19列特征  shape:(3,19)
            feature = pd.DataFrame(a)
            feature_ = pd.concat([feature_, feature], axis=1)  # 横向连接19个feature
        feature_['gra_x'] = np.full(feature_.shape[0], feature_arr[3][0])
        feature_['gra_y'] = np.full(feature_.shape[0], feature_arr[4][0])
        feature_['gra_z'] = np.full(feature_.shape[0], feature_arr[5][0])
        model = joblib.load('./forest_10.pkl')
        feature_matrix = np.array(feature_)
        result = model.predict(feature_matrix)
        sub_result[str(i-1)] = result   # 每次的预测：0-10
        if len(result) < 3:
            vote = result[-1]
        else:
            vote = stats.mode(result)[0][0]
        all_result.append(vote)
    
    for i in range(1, 11):
        data_ = data[str(i)]
        accZ = data_["accZs"][:]
        vector = fliter_signal(accZ, 0.15)
        indexes, _ = signal.find_peaks(vector, height=-10, distance=1)
        current_action = action[int(all_result[i-1])-1]
        dic[current_action] += len(_['peak_heights'])
    return dic



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

