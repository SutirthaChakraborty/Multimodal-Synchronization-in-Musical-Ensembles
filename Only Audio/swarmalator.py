import numpy as np
from simulation import Simulation

memory = 8
win_s = 1024
hop_s = win_s // 2
samplerate = 44100
beat_vol = 0.00005

# class for fifth attempt
class Swarmalator(Simulation):
    def setup(self):
        self.state_record = np.zeros([memory, len(self.state[0])])
        self.state_record += np.copy(self.state[0])

    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_r = np.zeros([1, num_osc])
        # the index of robot
        i = len(t_state[0]) - 1
        # for i in range(num_osc):
        diff_phase = 0
        diff_r = 0
        J = 1
        for j in range(num_osc):
            if (t_state[0][j] - t_state[0][i] > np.e):
                # second and third term of first equation
                diff_r += (t_state[0][j] - t_state[0][i]) / abs(t_state[0][j] - t_state[0][i]) * \
                          (1 + J * np.math.cos(t_state[1][j] - t_state[1][i])) - (t_state[0][j] - t_state[0][i]) \
                          / abs(t_state[0][j] - t_state[0][i]) ** 2

                # second term of second equation
                diff_phase += np.math.sin(t_state[1][j] - t_state[1][i]) / abs(t_state[0][j] - t_state[0][i])

        diff_r /= num_osc
        diff_phase /= num_osc - 1
        # first equation
        del_r[0][i] = 0.03 + diff_r
        # second equation
        del_theta[0][i] = t_state[0][i] + c * diff_phase
        del_all = np.append(del_r, del_theta, axis=0)
        t_state += del_all * delta

    def post_processing(self, t_state):
        self.state_record = np.append(
            self.state_record[1:], [t_state[0]], axis=0)

