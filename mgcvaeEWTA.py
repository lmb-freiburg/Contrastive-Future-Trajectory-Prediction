""" This code is based on the Trajectron++ repository.

    For usage, see the License of Trajectron++ under:
    https://github.com/StanfordASL/Trajectron-plus-plus
"""
import torch.nn as nn
import torch.nn.functional as F

from Trajectron_plus_plus.trajectron.model.components import *
from Trajectron_plus_plus.trajectron.model.model_utils import *
from Trajectron_plus_plus.trajectron.model.mgcvae import MultimodalGenerativeCVAE
import utilities


def contrastive_three_modes_loss(features, scores, temp=0.1, base_temperature=0.07):
    device = (torch.device('cuda') if features.is_cuda
              else torch.device('cpu'))
    batch_size = features.shape[0]
    scores = scores.contiguous().view(-1, 1)
    mask_positives = (torch.abs(scores.sub(scores.T)) < 0.1).float().to(device)
    mask_negatives = (torch.abs(scores.sub(scores.T)) > 2.0).float().to(device)
    mask_neutral = mask_positives + mask_negatives

    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0) * mask_neutral
    mask_positives = mask_positives * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
    mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)

    loss = - (temp / base_temperature) * mean_log_prob_pos
    loss = loss.view(1, batch_size).mean()
    return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()


class MultimodalGenerativeCVAEEWTA(MultimodalGenerativeCVAE):
    def __init__(self,
                 env,
                 node_type,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 log_writer=None):
        super().__init__(
            env, node_type, model_registrar, hyperparams,
            device, edge_types, log_writer)
        dynamic_class = getattr(utilities, hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type)

    def create_node_models(self):
        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))
        if self.hyperparams['edge_encoding']:
            if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                           hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                           bidirectional=True,
                                                           batch_first=True))
                self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']

            elif self.hyperparams['edge_influence_combine_method'] == 'attention':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=AdditiveAttention(
                                       encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
                                       decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))
                self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']
        if self.hyperparams['use_map_encoding']:
            if self.node_type in self.hyperparams['map_encoder']:
                me_params = self.hyperparams['map_encoder'][self.node_type]
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        x_size = self.hyperparams['enc_rnn_dim_history']
        if self.hyperparams['edge_encoding']:
            x_size += self.eie_output_dims
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            x_size += self.hyperparams['map_encoder'][self.node_type]['output_size']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule(self.node_type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/q_z_xy',
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.node_type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))

        decoder_input_dims = self.pred_state_length * 20 + x_size
        self.add_submodule(self.node_type + '/decoder/state_action',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))
        self.add_submodule(self.node_type + '/decoder/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(x_size, self.hyperparams['dec_rnn_dim']))

        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     20 * self.pred_state_length))
        self.x_size = x_size
        self.z_size = z_size

        self.add_submodule(self.node_type + '/con_head',
                           model_if_absent=nn.Linear(232, 232))

    def obtain_encoded_tensors(self, mode, inputs, inputs_st, labels, labels_st,
                               first_history_indices, neighbors,
                               neighbors_edge_value, robot, map):
        initial_dynamics = dict()
        batch_size = inputs.shape[0]
        node_history = inputs
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]
        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        n_s_t0 = node_present_state_st

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel
        self.dynamic.set_initial_condition(initial_dynamics)

        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        first_history_indices)
        if self.hyperparams['edge_encoding']:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                encoded_edges_type = self.encode_edge(mode,
                                                      node_history,
                                                      node_history_st,
                                                      edge_type,
                                                      neighbors[edge_type],
                                                      neighbors_edge_value[edge_type],
                                                      first_history_indices)
                node_edges_encoded.append(encoded_edges_type)
            total_edge_influence = self.encode_total_edge_influence(mode,
                                                                    node_edges_encoded,
                                                                    node_history_encoded,
                                                                    batch_size)
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
            do = self.hyperparams['map_encoder'][self.node_type]['dropout']
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        x_concat_list = list()
        if self.hyperparams['edge_encoding']:
            x_concat_list.append(total_edge_influence)
        x_concat_list.append(node_history_encoded)
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            x_concat_list.append(encoded_map)
        x = torch.cat(x_concat_list, dim=1)
        return x, n_s_t0

    def project_to_GMM_params(self, tensor):
        mus = self.node_modules[self.node_type + '/decoder/proj_to_GMM_mus'](tensor)
        return mus

    def p_y_xz(self, x, n_s_t0, prediction_horizon):
        ph = prediction_horizon  # 12
        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']
        initial_state = initial_h_model(x)
        mus = []
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)
        state = initial_state
        input_ = torch.cat([x, a_0.repeat(1, 20)], dim=1)
        features = torch.cat([input_, state], dim=1)
        for j in range(ph):
            h_state = cell(input_, state)
            mu_t = self.project_to_GMM_params(h_state)
            mus.append(mu_t.reshape(-1, 20, 2))
            dec_inputs = [x, mu_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
        mus = torch.stack(mus, dim=2)
        y = self.dynamic.integrate_samples(mus, x)
        return y, features

    def decoder(self, x, n_s_t0, prediction_horizon):
        y, features = self.p_y_xz(x, n_s_t0, prediction_horizon)
        return y, features

    def ewta_loss(self, y, labels, mode='epe-all', top_n=1):
        # y has shape (bs, 20, 12 ,2)
        # labels has shape (bs, 12, 2)
        gts = torch.stack([labels for i in range(20)], dim=1)  # (bs, 20, 12, 2)
        diff = (y - gts) ** 2
        channels_sum = torch.sum(diff, dim=3)  # (bs, 20, 12)
        spatial_epes = torch.sqrt(channels_sum + 1e-20)  # (bs, 20, 12)

        sum_spatial_epe = torch.zeros(spatial_epes.shape[0])
        if mode == 'epe':
            spatial_epe, _ = torch.min(spatial_epes, dim=1)  # (bs, 12)
            sum_spatial_epe = torch.sum(spatial_epe, dim=1)
        elif mode == 'epe-top-n' and top_n > 1:
            spatial_epes_min, _ = torch.topk(-1 * spatial_epes, top_n, dim=1)
            spatial_epes_min = -1 * spatial_epes_min  # (bs, top_n, 12)
            sum_spatial_epe = torch.sum(spatial_epes_min, dim=(1, 2))
        elif mode == 'epe-all':
            sum_spatial_epe = torch.sum(spatial_epes, dim=(1, 2))

        return torch.mean(sum_spatial_epe)

    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon,
                   loss_type,
                   score,
                   contrastive=False,
                   factor_con=100,
                   temp=0.1):
        mode = ModeKeys.TRAIN
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=labels,
                                                labels_st=labels_st,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        mode, top_n = loss_type, 1
        if 'top' in loss_type:
            mode = 'epe-top-n'
            top_n = int(loss_type.replace('epe-top-', ''))
        loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)
        if contrastive:
            con_loss, positive, negative = contrastive_three_modes_loss(features, score, temp=temp)
            final_loss = loss + factor_con * con_loss
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_loss'),
                                           con_loss, self.curr_iter)
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'positives'),
                                           positive, self.curr_iter)
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'negatives'),
                                           negative, self.curr_iter)
        else:
            final_loss = loss

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss, self.curr_iter)
        return final_loss

    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon):
        mode = ModeKeys.PREDICT
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features = F.normalize(features, dim=1)
        return y, features
