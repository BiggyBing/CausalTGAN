import os
import pickle
import torch
import numpy as np
from torch import autograd
from CausalTGAN.model.condGAN import ConditionalGAN
from CausalTGAN.helper.utils import print_progress, load_options
from CausalTGAN.model.module.generator import causal_generator
from CausalTGAN.model.module.discriminator import causalGAN_discriminator

def load_model(model_path, device, feature_info, transfomer):
    this_run_folder = model_path
    options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
    train_options, controller_config = load_options(options_file)
    check_point_folder = os.path.join(this_run_folder, 'checkpoints')

    model = CausalTGAN(device, controller_config, feature_info, transfomer)
    if model.causal_controller is not None:
        checkpoint = torch.load(os.path.join(check_point_folder, 'causal-TGAN.pyt'), map_location='cpu')
        model.load_checkpoint(checkpoint)

    if os.path.exists(os.path.join(this_run_folder, 'condGAN-config.pickle')):
        with open(os.path.join(this_run_folder, 'condGAN-config.pickle'), 'rb') as f:
            condGAN_config = pickle.load(f)

        checkpoint_condGAN = torch.load(os.path.join(check_point_folder, 'Cond-GAN.pyt'), map_location='cpu')
        model.load_checkpoint_condGAN(condGAN_config, checkpoint_condGAN)

    return model, train_options.experiment_name


class CausalTGAN(object):
    def __init__(self, device, config, feature_info, transformer):
        self.config = config
        self.device = device
        self.feature_info = feature_info
        self.transformer = transformer

        self._init_model()

    def _init_model(self):
        if len(self.config.causal_graph) == 0:
            self.causal_controller = None
        else:
            self.condGAN = None
            self.causal_controller = causal_generator(self.device, self.config, self.feature_info)
            data_dim = sum(self.feature_info.dim_info.values())
            data_dim = data_dim * self.config.pac_num
            self.discriminator = causalGAN_discriminator(data_dim).to(self.device)
            self.generators_params = self.causal_controller.get_causal_mechanisms_params()

            self.gen_optimizer = torch.optim.Adam(
                self.generators_params, lr=2e-4, betas=(0.5, 0.9),
                weight_decay=1e-6)

            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=2e-4,
                betas=(0.5, 0.9), weight_decay=1e-6)

    def fit(self, train_data, train_options, full_knowledge, condGAN_config=None, verbose=True):
        if full_knowledge:
            self._fit_causalGAN(train_data, train_options, verbose=verbose)
        else:
            train_data_causalGAN, train_data_condGAN = train_data
            if self.causal_controller is None:
                self.condGAN = ConditionalGAN(self.device, condGAN_config, self.transformer)
                self.condGAN.fit(train_data_condGAN, verbose=verbose)

                condGAN_checkpoints = self.condGAN.fetch_checkpoint()
                self._save_checkpoint(condGAN_checkpoints, os.path.join(train_options.runs_folder, 'checkpoints'), 'Cond-GAN.pyt')
            else:
                if condGAN_config is None:
                    raise ValueError("condGAN_config must be given when using parital knowledge")
                train_data_causalGAN, train_data_condGAN = train_data
                self.condGAN = ConditionalGAN(self.device, condGAN_config, self.transformer)
                self._fit_causalGAN(train_data_causalGAN, train_options, verbose=verbose)
                self.condGAN.fit(train_data_condGAN, verbose=verbose)

                condGAN_checkpoints = self.condGAN.fetch_checkpoint()
                self._save_checkpoint(condGAN_checkpoints, os.path.join(train_options.runs_folder, 'checkpoints'), 'Cond-GAN.pyt')

    def _fit_causalGAN(self, train_data, train_options, verbose):
        for i in range(train_options.number_of_epochs):
            losses_accu = self.train_one_epoch(train_data)
            if verbose:
                print('Epoch {}/{}'.format(i, train_options.number_of_epochs))
                print_progress(losses_accu)
                print('-'*40)

        checkpoint = self.fetch_checkpoint()
        self._save_checkpoint(checkpoint, os.path.join(train_options.runs_folder, 'checkpoints'), 'causal-TGAN.pyt')
        print_progress(losses_accu)
        print('-' * 40)

    def _save_checkpoint(self, checkpoint, checkpoint_folder, checkpoint_filename):
        """ Saves a checkpoint at the end of an epoch. """
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
        torch.save(checkpoint, checkpoint_filename)
        print('Saving checkpoint done.')

    def train_one_epoch(self, train_data):
        G_losses = []
        D_losses = []
        for steps, data in enumerate(train_data):
            batch_size = data.size(0)
            if batch_size % self.config.pac_num != 0:
                continue

            real_data = data.to(self.device)

            D_real = self.discriminator(real_data)
            D_real = D_real.mean()

            fake_data = self.causal_controller.sample(batch_size).contiguous()
            D_fake = self.discriminator(fake_data)
            D_fake = D_fake.mean()

            # train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(real_data, fake_data, self.config.pac_num)

            D_cost = D_fake.mean() - D_real.mean()

            self.discriminator.zero_grad()
            gradient_penalty.backward(retain_graph=True)
            D_cost.backward()
            self.disc_optimizer.step()
            D_losses.append(D_cost.data.cpu().numpy())

            if (steps+1) % self.config.D_iter == 0:
                fake_data = self.causal_controller.sample(batch_size).contiguous()
                G = self.discriminator(fake_data)
                G = G.mean()
                G_cost = -G
                self.gen_optimizer.zero_grad()
                G_cost.backward()
                self.gen_optimizer.step()
                G_losses.append(G_cost.data.cpu().numpy())

        losses = {
            'G_cost         ': np.mean(G_losses),
            'D_cost         ': np.mean(D_losses)
        }
        return losses

    def calc_gradient_penalty(self, real_data, fake_data, pac_num=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac_num, 1, 1, device=self.device)
        alpha = alpha.repeat(1, pac_num, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac_num * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def fetch_checkpoint(self):
        if self.causal_controller is not None:
            # fetch the checkpoint of causal mechanisms
            checkpoints = self.causal_controller.fetch_checkpoints()
            checkpoints['discriminator'] = self.discriminator.state_dict()
            checkpoints['gen_optim'] = self.gen_optimizer.state_dict()
            checkpoints['dis_optim'] = self.disc_optimizer.state_dict()

            return checkpoints


    def load_checkpoint(self, checkpoints):
        if self.causal_controller is not None:
            self.gen_optimizer.load_state_dict(checkpoints['gen_optim'])
            self.disc_optimizer.load_state_dict(checkpoints['dis_optim'])
            self.discriminator.load_state_dict(checkpoints['discriminator'])
            self.causal_controller.restore_from_checkpoints(checkpoints)
        else:
            raise NotImplementedError('Causal-TGAN is not set, it may caused by a null causal graph')

    def load_checkpoint_condGAN(self, condGAN_config, checkpoints):
        self.condGAN = ConditionalGAN(self.device, condGAN_config, self.transformer)
        self.condGAN.load_checkpoint(checkpoints)

    def to_stirng(self):
        return '{}\n{}'.format(str(list(self.causal_controller.nodes.values())[0].causal_mechanism), str(self.discriminator))

    def sample(self, batch_size):
        if self.condGAN is None:
            return self.causal_controller.sample(batch_size)
        else:
            condvec = self.causal_controller.sample(batch_size) if self.causal_controller is not None else None
            return self.condGAN.sample(batch_size, condvec)

