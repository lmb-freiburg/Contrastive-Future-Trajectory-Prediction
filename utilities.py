""" This code is based on the Trajectron++ repository.

    For usage, see the License of Trajectron++ under:
    https://github.com/StanfordASL/Trajectron-plus-plus
"""
import torch
from Trajectron_plus_plus.trajectron.model.dynamics.single_integrator import SingleIntegrator as SingleIntegratorCVAE
from Trajectron_plus_plus.trajectron.model.dynamics.unicycle import Unicycle as UnicycleCVAE


class SingleIntegrator(SingleIntegratorCVAE):
    def integrate_samples(self, v, x=None):
        p_0 = self.initial_conditions['pos'].unsqueeze(1).unsqueeze(2)
        return torch.cumsum(v, dim=2) * self.dt + p_0


class Unicycle(UnicycleCVAE):
    def integrate_samples(self, control_samples, x=None):
        ph = control_samples.shape[-2]
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        v_0 = self.initial_conditions['vel'].unsqueeze(1)
        phi_0 = torch.atan2(v_0[..., 1], v_0[..., 0])
        phi_0 = phi_0 + torch.tanh(self.p0_model(torch.cat((x, phi_0), dim=-1)))
        u = torch.stack([control_samples[..., 0], control_samples[..., 1]], dim=0)
        x = torch.stack([p_0[..., 0], p_0[..., 1], phi_0, torch.norm(v_0, dim=-1)], dim=0)
        mus_list = []
        for t in range(ph):
            x = self.dynamic(x, u[..., t])
            mus_list.append(torch.stack((x[0], x[1]), dim=-1))
        pos_mus = torch.stack(mus_list, dim=2)
        return pos_mus


def estimate_kalman_filter(history, prediction_horizon):
    """
    Predict the future position by running the kalman filter.

    :param history: 2d array of shape (length_of_history, 2)
    :param prediction_horizon: how many steps in the future to predict
    :return: the predicted position (x, y)
    """
    length_history = history.shape[0]
    z_x = history[:, 0]
    z_y = history[:, 1]
    v_x = 0
    v_y = 0
    for index in range(length_history - 1):
        v_x += z_x[index + 1] - z_x[index]
        v_y += z_y[index + 1] - z_y[index]
    v_x = v_x / (length_history - 1)
    v_y = v_y / (length_history - 1)
    x_x = np.zeros(length_history + 1, np.float32)
    x_y = np.zeros(length_history + 1, np.float32)
    P_x = np.zeros(length_history + 1, np.float32)
    P_y = np.zeros(length_history + 1, np.float32)
    P_vx = np.zeros(length_history + 1, np.float32)
    P_vy = np.zeros(length_history + 1, np.float32)

    # we initialize the uncertainty to one (unit gaussian)
    P_x[0] = 1.0
    P_y[0] = 1.0
    P_vx[0] = 1.0
    P_vy[0] = 1.0
    x_x[0] = z_x[0]
    x_y[0] = z_y[0]

    Q = 0.00001
    R = 0.0001
    K_x = np.zeros(length_history + 1, np.float32)
    K_y = np.zeros(length_history + 1, np.float32)
    K_vx = np.zeros(length_history + 1, np.float32)
    K_vy = np.zeros(length_history + 1, np.float32)
    for k in range(length_history - 1):
        x_x[k + 1] = x_x[k] + v_x
        x_y[k + 1] = x_y[k] + v_y
        P_x[k + 1] = P_x[k] + P_vx[k] + Q
        P_y[k + 1] = P_y[k] + P_vy[k] + Q
        P_vx[k + 1] = P_vx[k] + Q
        P_vy[k + 1] = P_vy[k] + Q
        K_x[k + 1] = P_x[k + 1] / (P_x[k + 1] + R)
        K_y[k + 1] = P_y[k + 1] / (P_y[k + 1] + R)
        x_x[k + 1] = x_x[k + 1] + K_x[k + 1] * (z_x[k + 1] - x_x[k + 1])
        x_y[k + 1] = x_y[k + 1] + K_y[k + 1] * (z_y[k + 1] - x_y[k + 1])
        P_x[k + 1] = P_x[k + 1] - K_x[k + 1] * P_x[k + 1]
        P_y[k + 1] = P_y[k + 1] - K_y[k + 1] * P_y[k + 1]
        K_vx[k + 1] = P_vx[k + 1] / (P_vx[k + 1] + R)
        K_vy[k + 1] = P_vy[k + 1] / (P_vy[k + 1] + R)
        P_vx[k + 1] = P_vx[k + 1] - K_vx[k + 1] * P_vx[k + 1]
        P_vy[k + 1] = P_vy[k + 1] - K_vy[k + 1] * P_vy[k + 1]

    k = k + 1
    x_x[k + 1] = x_x[k] + v_x * prediction_horizon
    x_y[k + 1] = x_y[k] + v_y * prediction_horizon
    P_x[k + 1] = P_x[k] + P_vx[k] * prediction_horizon * prediction_horizon + Q
    P_y[k + 1] = P_y[k] + P_vy[k] * prediction_horizon * prediction_horizon + Q
    P_vx[k + 1] = P_vx[k] + Q
    P_vy[k + 1] = P_vy[k] + Q
    return x_x[k + 1], x_y[k + 1]
