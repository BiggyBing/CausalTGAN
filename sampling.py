import os
import pickle
import torch
import numpy as np

from CausalTGAN.model.causalTGAN import load_model
from CausalTGAN.helper.utils import restore_feature_info

def synthetic(model_path, gen_num, device, save_path=None):
    transformer, feature_info, causal_graph = restore_feature_info(model_path)
    model, exp_name = load_model(model_path, device, feature_info, transformer)

    if model.causal_controller is not None:
        model.causal_controller.set_causal_mechanisms_eval()
    if model.condGAN is not None:
        model.condGAN.generator.eval()

    r = []
    with torch.no_grad():
        for _ in range(int(np.ceil(gen_num/1000))):
            samples = model.sample(1000)
            r.append(samples.cpu())

    samples = torch.cat(r, dim=0)
    sample_df = transformer.inverse_transform(samples.data.numpy())

    save_path = os.path.join(model_path, 'generated_samples.csv') if save_path is None else save_path
    sample_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    gen_num = 34000
    device = torch.device('cuda:2')
    model_path = '/home/bingyang/Causal-TGAN/CausalTGAN/Testing/CausalTGAN_runs_new_3_adult2022.10.10--22-04-12'
    synthetic(model_path, gen_num, device)
