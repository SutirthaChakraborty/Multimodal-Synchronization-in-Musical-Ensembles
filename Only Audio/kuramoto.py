import numpy as np
from simulation import Simulation

memory = 8
win_s = 1024
hop_s = win_s // 2
samplerate = 44100
beat_vol = 0.00005

# class for fifth attempt
class Kuramoto(Simulation):
    def setup(self):
        self.state_record = np.zeros([memory, len(self.state[0])])
        self.state_record += np.copy(self.state[0])

    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_w = np.zeros([1, num_osc])
        group_w = np.average(t_state[0])
        group_phase = np.average(t_state[1])

        std_dev_w = np.std(self.state_record, axis=0)
        std_dev_w = normalize2(std_dev_w)
        avg_std_dev_w = np.average(std_dev_w)
        a = np.zeros([num_osc, num_osc])
        for i in range(num_osc):
            if std_dev_w[i] > avg_std_dev_w:
                for j in range(num_osc):
                    if std_dev_w[i] == 0:
                        a[i][j] = np.inf
                    else:
                        a[i][j] = std_dev_w[j]/std_dev_w[i]
            else:
                for j in range(num_osc):
                    if std_dev_w[j] == 0:
                        a[i][j] = np.inf
                    else:
                        a[i][j] = std_dev_w[i]/std_dev_w[j]

        a = normalize3(a)

        for i in range(num_osc):
            r = 0
            for j in range(num_osc):
                r += e_val(t_state[1][j])
            r += e_val(t_state[1][i])
            r /= (num_osc + 1)

            diff_phase = 0
            std_dev_w = 0
            for j in range(num_osc):
                diff_phase += a[i][j] * \
                    np.math.sin(t_state[0][j] - t_state[0][i])
                std_dev_w += (t_state[0][i] - t_state[0][j])**2
            std_dev_w /= (num_osc-1)
            std_dev_w = np.math.sqrt(std_dev_w)

            del_theta[0][i] = t_state[0][i] + c*diff_phase
            if t_state[0][i] - group_w > 0:
                del_w[0][i] = -(1-r)*(t_state[0][i])
            else:
                del_w[0][i] = (1-r)*(t_state[0][i])

        del_all = np.append(del_w, del_theta, axis=0)
        t_state += del_all*delta

    def post_processing(self, t_state):
        self.state_record = np.append(
            self.state_record[1:], [t_state[0]], axis=0)

def e_val(theta):
    return np.abs(np.math.cos(theta))


def normalize2(v):
    result = np.copy(v)
    m = 0
    for i in v:
        abs_i = np.abs(i)
        if abs_i > m:
            m = abs_i

    if m == 0:
        result = np.ones([len(v)])
    else:
        result /= m
    return result


def normalize3(v):
    result = np.copy(v)
    m = 0
    for i in v:
        for j in i:
            abs_j = np.abs(j)
            if abs_j > m:
                m = abs_j

    if m == 0:
        result = np.ones([len(v), len(v[0])])
    elif m == np.inf:
        for i in range(len(result)):
            for j in range(len(result[i])):
                if result[i][j] == np.inf:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
    else:
        result /= m
    return result


