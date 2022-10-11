import torch
import numpy as np
from packaging import version
from torch.nn import functional
from CausalTGAN.helper.utils import print_progress
from CausalTGAN.model.module.discriminator import condGAN_discriminator
from CausalTGAN.model.module.generator import condGAN_generator

class ConditionalGAN(object):
    def __init__(self, device, config, transformer, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, discriminator_steps=3, pac=1):

        self.device = device
        self._transformer = transformer
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._discriminator_steps = discriminator_steps

        self._causal_graph = config.causal_graph
        self._col_names = config.col_names
        self._col_dims = config.col_dims

        self.pac = pac

        data_dim, partial_data_dim = self.process_dims()
        self.no_knowledge_flag = data_dim == partial_data_dim

        self.generator = condGAN_generator(
            self._embedding_dim + data_dim - partial_data_dim,
            self._generator_dim,
            partial_data_dim
        ).to(self.device)

        self.discriminator = condGAN_discriminator(
            data_dim,
            self._discriminator_dim,
            pac=self.pac
        ).to(self.device)

        self.optimizerG = torch.optim.Adam(
            self.generator.parameters(), lr=generator_lr, betas=(0.5, 0.9),
            weight_decay=generator_decay
        )

        self.optimizerD = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr,
            betas=(0.5, 0.9), weight_decay=discriminator_decay
        )

    def process_dims(self):
        self._col_idx_dict = dict(zip(self._col_names, [i for i in range(len(self._col_names))]))
        data_dim = np.sum(self._col_dims)
        self._remove_cols = [item[0] for item in self._causal_graph]
        partial_dim = np.sum([self._col_dims[self._col_idx_dict[name]] for name in self._col_names if name not in self._remove_cols])

        return int(data_dim), int(partial_dim)

    def sample_condvec(self, real_data):
        if self.no_knowledge_flag:
            return None

        condvec = []
        real_data_copy = real_data.clone()
        for rm_col in self._remove_cols:
            col_idx = self._col_idx_dict[rm_col]
            col_start_idx = np.sum(self._col_dims[:col_idx]).astype(int)
            col_end_idx = col_start_idx + self._col_dims[col_idx]
            val = real_data_copy[:, col_start_idx: col_end_idx]
            condvec.append(val)

        return torch.cat(condvec, dim=1)

    def joint(self, fake_data_partial, condvec):
        '''
        joining conditional and generated values
        :return: complete generated samples
        '''
        joint = []
        # two pointer to idx two tensor
        keep_col_idx = 0
        rm_col_idx = 0

        for name in self._col_names:
            if name in self._remove_cols:
                end_idx = rm_col_idx + self._col_dims[self._col_idx_dict[name]]
                joint.append(condvec[:, rm_col_idx: end_idx])
                rm_col_idx = end_idx
            else:
                end_idx = keep_col_idx + self._col_dims[self._col_idx_dict[name]]
                joint.append(fake_data_partial[:, keep_col_idx: end_idx])
                keep_col_idx = end_idx

        return torch.cat(joint, dim=1)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for i, column_info in enumerate(self._transformer.output_info_list):
            if self._col_names[i] in self._remove_cols:
                continue
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                elif span_info.activation_fn == 'relu':
                    ed = st + span_info.dim
                    data_t.append(torch.relu(data[:, st:ed]))
                    st = ed
                else:
                    assert 0

        return torch.cat(data_t, dim=1)

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def fit(self, train_data, verbose=False, epochs=400):
        for i in range(epochs):
            losses_accu = self.train_one_epoch(train_data)
            if verbose:
                print('Epoch {}/{}'.format(i, epochs))
                print_progress(losses_accu)
                print('-' * 40)

    def train_one_epoch(self, train_data):
        G_losses = []
        D_losses = []
        for steps, data in enumerate(train_data):
            real_data = data.to(self.device)
            batch_size = real_data.size(0)
            if batch_size % self.pac != 0:
                continue

            mean = torch.zeros(batch_size, self._embedding_dim, device=self.device)
            std = mean + 1

            fakez = torch.normal(mean=mean, std=std)
            condvec = self.sample_condvec(real_data)

            if condvec is not None:
                fakez = torch.cat([fakez, condvec], dim=1)
            fake = self.generator(fakez)
            fake_data_partial = self._apply_activate(fake)  # generator only generate S_{normal}
            if condvec is not None:
                fake_data = self.joint(fake_data_partial, condvec)
            else:
                fake_data = fake_data_partial

            y_fake = self.discriminator(fake_data)
            y_real = self.discriminator(real_data)

            pen = self.discriminator.calc_gradient_penalty(
                real_data, fake_data, self.device)

            loss_d = torch.mean(y_fake) - torch.mean(y_real)

            self.optimizerD.zero_grad()
            pen.backward(retain_graph=True)
            loss_d.backward()
            self.optimizerD.step()
            D_losses.append(loss_d.data.cpu().numpy())

            if (steps+1) % self._discriminator_steps == 0:
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.sample_condvec(real_data)

                if condvec is not None:
                    fakez = torch.cat([fakez, condvec], dim=1)

                fake = self.generator(fakez)
                fake_data_partial = self._apply_activate(fake)  # generator only generate S_{normal}

                if condvec is not None:
                    fake_data = self.joint(fake_data_partial, condvec)
                else:
                    fake_data = fake_data_partial

                y_fake = self.discriminator(fake_data)

                loss_g = -torch.mean(y_fake)

                self.optimizerG.zero_grad()
                loss_g.backward()
                self.optimizerG.step()
                G_losses.append(loss_g.data.cpu().numpy())

        losses = {
            'G_cost         ': np.mean(G_losses),
            'D_cost         ': np.mean(D_losses)
        }
        return losses

    def fetch_checkpoint(self):
        # fetch the checkpoint of causal mechanisms
        checkpoints = {'generator':self.generator.state_dict(),
                       'discriminator': self.discriminator.state_dict(),
                       'gen_optim': self.optimizerG.state_dict(),
                       'dis_optim': self.optimizerD.state_dict()}

        return checkpoints

    def load_checkpoint(self, checkpoints):
        self.generator.load_state_dict(checkpoints['generator'])
        self.discriminator.load_state_dict(checkpoints['discriminator'])
        self.optimizerG.load_state_dict(checkpoints['gen_optim'])
        self.optimizerD.load_state_dict(checkpoints['dis_optim'])

    def sample(self, batch_size, condvec=None):
        mean = torch.zeros(batch_size, self._embedding_dim, device=self.device)
        std = mean + 1

        fakez = torch.normal(mean=mean, std=std)
        if condvec is not None:
            fakez = torch.cat([fakez, condvec], dim=1)

        fake = self.generator(fakez)
        fake_data_partial = self._apply_activate(fake)  # generator only generate S_{normal}

        if condvec is not None:
            fake_data = self.joint(fake_data_partial, condvec)
        else:
            fake_data = fake_data_partial

        return fake_data
