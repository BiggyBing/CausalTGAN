import torch
import torch.nn as nn

class causalGAN_discriminator(nn.Module):
    def __init__(self, data_dim):
        super(causalGAN_discriminator, self).__init__()
        self.input_dim = data_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model(x.view(-1, self.input_dim))
        return x

class condGAN_discriminator(nn.Module):
    def __init__(self, input_dim, discriminator_dim, pac=1):
        super(condGAN_discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
        alpha = torch.rand(real_data.size(0) // self.pac, 1, 1, device=device)
        alpha = alpha.repeat(1, self.pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))
