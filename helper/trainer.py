import torch
from CausalTGAN.dataset import NumpyDataset, DataTransformer


def train_full_knowledge(train_options, transform_data, trainer):
    dataset = NumpyDataset(transform_data)
    train_data = torch.utils.data.DataLoader(dataset, batch_size=train_options.batch_size, shuffle=True)

    trainer.fit(train_data, train_options, full_knowledge=True, verbose=True)

def train_partial_knowledge(train_options, condGAN_config, transform_data, transform_data_causalGAN, trainer):
    dataset_causalGAN = NumpyDataset(transform_data_causalGAN)
    train_data_causalGAN = torch.utils.data.DataLoader(dataset_causalGAN, batch_size=train_options.batch_size, shuffle=True)

    dataset = NumpyDataset(transform_data)
    train_data = torch.utils.data.DataLoader(dataset, batch_size=train_options.batch_size, shuffle=True)

    train_data = [train_data_causalGAN, train_data]
    trainer.fit(train_data, train_options, condGAN_config=condGAN_config, full_knowledge=False, verbose=True)

def train_no_knowledge(train_options, condGAN_config, transform_data, trainer):
    dataset = NumpyDataset(transform_data)
    train_data = torch.utils.data.DataLoader(dataset, batch_size=train_options.batch_size, shuffle=True)

    train_data = [None, train_data]
    trainer.fit(train_data, train_options, condGAN_config=condGAN_config, full_knowledge=False, verbose=True)
