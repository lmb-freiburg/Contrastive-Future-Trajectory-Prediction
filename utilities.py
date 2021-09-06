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
