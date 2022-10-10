import os
import csv
import time
import pickle
import networkx as nx
import logging

import torch
import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset
from CausalTGAN.dataset import DataTransformer, GeneralTransformer, PlainTransformer
from CausalTGAN.helper.constant import KINGS_CATEGORY, LOAN_CATEGORY, CABS_CATEGORY, ADULT_CATEGORY, NEWS_CATEGORY, CENSUS_CATEGORY, CREDIT_CATEGORY, DATASETS



def get_transformer(transformer_type):
    if transformer_type == 'general':
        return GeneralTransformer()
    elif transformer_type == 'plain':
        return PlainTransformer()
    elif transformer_type == 'ctgan':
        return DataTransformer()
    else:
        raise ('Transformer type of {} does not exist, should be one of [\'general\', \'tablegan\', \'ctgan\']'.format(transformer_type))

def data_transform(transformer_type, data_name, data, discrete_cols):
    data_folder = path_by_name(data_name)
    transformer_path = os.path.join(data_folder, 'transformer.pickle')
    if os.path.exists(transformer_path):
        with open(transformer_path, 'rb') as f:
            transformer = pickle.load(f)
        transform_data, data_dims = transformer.transform(data)
    else:
        transformer = get_transformer(transformer_type)
        transformer.fit(data, discrete_cols)
        transform_data, data_dims = transformer.transform(data)

    return transform_data, transformer, data_dims


class NumpyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, transform=None):
        """
        Args:
            file_pat (string): Path to the numpy file .
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = np.load(file_path)
        data = torch.from_numpy(data)
        if len(data.shape) == 3:
            data.unsqueeze(0)
        data = data.permute(0,3,1,2)
        self.data = data.numpy()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample


def model_from_checkpoint(wgan_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    wgan_net.generator.load_state_dict(checkpoint['gen-model'])
    wgan_net.gen_optimizer.load_state_dict(checkpoint['gen-optim'])
    wgan_net.discriminator.load_state_dict(checkpoint['disc-model'])
    wgan_net.disc_optimizer.load_state_dict(checkpoint['disc-optim'])


def load_options(options_file_name):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        gan_config = pickle.load(f)

    return train_options, gan_config


def print_progress(losses_accu):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        print(loss_name.ljust(max_len+4) + '{:.4f}'.format(np.mean(loss_value)))


def create_folder_for_run(runs_folder, experiment_name):

    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, experiment_name+time.strftime("%Y.%m.%d--%H-%M-%S"))

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(np.mean(loss_list)) for loss_list in losses_accu.values()] + ['{:.0f}'.format(duration)]
        writer.writerow(row_to_write)


def save_checkpoint(model, experiment_name, epoch, checkpoint_folder):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = experiment_name + '--epoch-{}.pyt'.format(epoch)
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = model.fetch_checkpoint()
    checkpoint['epoch'] = epoch
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')

# Below by Bingyang
def make_folders_batch_train(args):
    if args.setting == 'Partial':
        level1_folder = os.path.join(args.runs_folder, args.data_name)
        level2_folder = os.path.join(level1_folder, args.setting)
        level3_folder = os.path.join(level2_folder, str(args.partial_p))
        if not os.path.exists(level3_folder):
            os.makedirs(level3_folder)
        return level3_folder

    elif args.setting == 'Wrong':
        level1_folder = os.path.join(args.runs_folder, args.data_name)
        level2_folder = os.path.join(level1_folder, args.setting)
        level3_folder = os.path.join(level2_folder, str(args.delete_p))
        level4_folder = os.path.join(level3_folder, str(args.wrong_p))
        if not os.path.exists(level4_folder):
            os.makedirs(level4_folder)
        return level4_folder

def restore_feature_info(folder_path):
    t_path = os.path.join(folder_path, 'transformer.pickle')
    f_path = os.path.join(folder_path, 'featureInfo.pickle')
    graph_path = os.path.join(folder_path, 'causal_graph.pickle')
    with open(t_path, 'rb') as f:
        transformer = pickle.load(f)
    with open(f_path, 'rb') as f:
        feature_info = pickle.load(f)
    with open(graph_path, 'rb') as f:
        causal_graph = pickle.load(f)

    return transformer, feature_info, causal_graph

def check_BN_datatype(data_name):
    if data_name in ['asia', 'alarm', 'child', 'hepar2', 'insurance', 'link', 'munin']:
        return 'discrete'
    elif data_name in ['arth150', 'ecoli70', 'magic-irri']:
        return 'continuous'
    elif data_name in ['healthcare', 'mehra-complete', 'sangiovese']:
        return 'mixed'
    else:
        return None

def path_by_name(data_name):
    dataset_type = check_BN_datatype(data_name)
    if dataset_type is not None:
        folder_path = './data/bayesian_network/{}/{}'.format(dataset_type, data_name)
        return folder_path
    else:
        folder_path = './data/real_world/{}'.format(data_name)
        return folder_path


def get_discrete_cols(data, data_name):
    dataset_type = check_BN_datatype(data_name)
    col_names = data.keys().to_list()
    if dataset_type == 'continuous':
        discrete_cols = []
    elif dataset_type == 'discrete':
        discrete_cols = col_names
    else:
        discrete_cols = data.dtypes[(data.dtypes=='bool') | (data.dtypes=='object')].keys().to_list()

    if data_name == 'adult':
        discrete_cols = ADULT_CATEGORY
    elif data_name == 'census':
        discrete_cols = CENSUS_CATEGORY
    elif data_name == 'news':
        discrete_cols = NEWS_CATEGORY
    elif data_name == 'credit':
        discrete_cols = CREDIT_CATEGORY
    elif data_name == 'cabs':
        discrete_cols = CABS_CATEGORY
    elif data_name == 'loan':
        discrete_cols = LOAN_CATEGORY
    elif data_name == 'kings':
        discrete_cols = KINGS_CATEGORY

    return discrete_cols, col_names

# TO-DO: correct real dataset loading
def load_data_graph(data_name, graph_path=None):
    """
    Load train dataset and its columns information and causal graph.
    :param data_name: dataset name
    :param graph_path: causal graph path
    """
    data_folder = path_by_name(data_name)
    graph_path = os.path.join(data_folder, 'graph.txt') if graph_path is None else graph_path

    with open(graph_path, "rb") as fp:
        causal_graph = pickle.load(fp)

    data_path = os.path.join(data_folder, 'train.csv')
    data = pd.read_csv(data_path).iloc[:, 1:]
    discrete_cols, col_names = get_discrete_cols(data, data_name)

    return data, col_names, discrete_cols, causal_graph

def load_train_test(data_name):
    data_folder = path_by_name(data_name)

    train_path = os.path.join(data_folder, 'train.csv')
    train = pd.read_csv(train_path).iloc[:, 1:]

    test_path = os.path.join(data_folder, 'test.csv')
    test = pd.read_csv(test_path).iloc[:, 1:]

    return train, test

def _adjMatrix2graph(adjMatrix, col_names):
    graph = [[item, []] for item in col_names]
    for idx, c_nodes in enumerate(adjMatrix):
        c_idx = np.where(np.asarray(c_nodes)==1)
        for i in c_idx[0]:
            graph[i][1].append(col_names[idx])

    return graph

def topology_order(amat):
    order = []
    amat = amat.copy()
    num_node = len(amat[0])
    while True:
        tmp = amat.sum(axis=0)
        cur_root = [i for i in range(num_node) if (tmp[i]==0) & (i not in order)]
        for idx in cur_root:
            amat[idx] = [0 for _ in range(num_node)]
        order.extend(cur_root)
        if len(cur_root) == 0:
            break
    return order

def _no_cycle(amat):
    G = nx.from_numpy_matrix(amat, create_using=nx.DiGraph)
    try:
        tmp = next(nx.simple_cycles(G))
        return False
    except:
        return True

def read_names(path):
    col_names = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip()
            col_names.append(tmp.replace("\"" ,""))
    return col_names

def read_amat(path):
    adj_matrix = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(' ')
            tmp = [int(item) for item in tmp]
            adj_matrix.append(tmp)
    return np.asarray(adj_matrix)






