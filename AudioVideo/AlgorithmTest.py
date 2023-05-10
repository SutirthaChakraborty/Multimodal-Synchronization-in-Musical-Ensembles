import numpy as np
"""
Class Name  :   AlgorithmSimulate
input       :   parameters

output      :   new parameters
"""
class AlgorithmSimulate:
    def __init__(self, delta, wn, person_num):
        self.delta=delta
        self.wn = wn
        self.velocities = np.random.rand(2, person_num+1)

    def swarmalator(self, t_state, vn):
        c=1
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
        del_r[0][i] = vn + diff_r
        # second equation
        del_theta[0][i] = self.wn + c * diff_phase
        del_all = np.append(del_r, del_theta, axis=0)
        t_state += del_all * self.delta
        # if t_state[1][i] > np.pi * 2:
        #     t_state[1][i] -= np.pi * 2
        return t_state

    def kuramoto(self, t_state):
        K=5
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        for i in range(num_osc):
            diff_phase = 0
            # Government equation
            for j in range(num_osc):
                diff_phase += np.math.sin(t_state[1][j] - t_state[1][i])
            diff_phase /= (num_osc - 1)
            t_state[0][i] = self.wn + K * diff_phase * self.delta
            t_state[1][i] += (self.wn + K * diff_phase) * self.delta
            # if t_state[1][i] > np.pi * 2:
            #     t_state[1][i] -= np.pi * 2
        return t_state

    # def janus(self, t_state):
    #     num_osc = len(t_state[0])
    #     del_theta = np.zeros([1, num_osc])
    #     beta = 10
    #     sigma = 10
    #     vn = 2

    #     for i in range(0, num_osc):
    #         if i - 1 == 0:
    #             j = num_osc - 1
    #         elif i + 1 == num_osc:
    #             j = 0
    #         else:
    #             j = i
    #         t_state[0][i] += (vn + beta * np.sin(t_state[1][i] - t_state[0][i]) + sigma * np.sin(
    #             t_state[1][j] - t_state[0][i])) * self.delta
    #         t_state[1][i] += (self.wn + beta * np.sin(t_state[0][i] - t_state[1][i]) + sigma * np.sin(
    #             t_state[0][j] - t_state[1][i])) * self.delta
    #         # if t_state[1][i] > np.pi * 2:
    #         #     t_state[1][i] -= np.pi * 2
    #     return t_state

    # def flock(self, t_state, ti):
    #     positions =np.array([t_state[0][:] * np.cos(t_state[1][:]), t_state[0][:] * np.sin(t_state[1][:])])
    #     num_osc = len(positions[0])
    #     move_to_middle_strength = 0.05
    #     middle = np.mean(positions, 1)
    #     direction_to_middle = positions - middle[:, np.newaxis]
    #     self.velocities -= direction_to_middle * move_to_middle_strength

    #     separations = positions[:, np.newaxis, :] - positions[:, :, np.newaxis]
    #     squared_displacements = separations * separations
    #     square_distances = np.sum(squared_displacements, 0)
    #     alert_distance = 20
    #     far_away = square_distances > alert_distance
    #     separations_if_close = np.copy(separations)
    #     separations_if_close[0, :, :][far_away] = 0
    #     separations_if_close[1, :, :][far_away] = 0
    #     self.velocities += np.sum(separations_if_close, 1)

    #     velocity_differences = self.velocities[:, np.newaxis, :] - self.velocities[:, :, np.newaxis]
    #     formation_flying_distance = 30
    #     formation_flying_strength = 0.5
    #     very_far = square_distances > formation_flying_distance
    #     velocity_differences_if_close = np.copy(velocity_differences)
    #     velocity_differences_if_close[0, :, :][very_far] = 0
    #     velocity_differences_if_close[1, :, :][very_far] = 0
    #     self.velocities -= np.mean(velocity_differences_if_close, 1) * formation_flying_strength

    #     positions += self.velocities
    #     positions[0][-1] = 30 * np.cos(self.wn * ti)
    #     positions[1][-1] = 30 * np.sin(self.wn * ti)
    #     radi=np.sqrt(positions[0][:]**2+positions[1][:]**2)
    #     theta=np.arctan(positions[1][:]/positions[0][:])
    #     for i in range(num_osc):
    #         if positions[0][i] < 0:
    #             theta[i] = np.pi + theta[i]

    #     t_state=np.stack([radi, theta])
    #     return t_state