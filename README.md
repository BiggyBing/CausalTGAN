# Causal-TGAN
This is the code for the project [Causal-TGAN: Causally-Aware Tabular Data Generative Adversarial Networks](https://openreview.net/forum?id=BEhxCh4dvW5)

## Citation
If you use this code for your research, please cite:
```
@inproceedings{
wen2022causaltgan,
title={Causal-{TGAN}: Modeling Tabular Data Using Causally-Aware {GAN}},
author={Bingyang Wen and Yupeng Cao and Fan Yang and Koduvayur Subbalakshmi and Rajarathnam Chandramouli},
booktitle={ICLR Workshop on Deep Generative Models for Highly Structured Data},
year={2022},
url={https://openreview.net/forum?id=BEhxCh4dvW5}
}
```

## Prerequisties
The project is built on python3.6 with the pytorch version of 1.9.0.

To run this project, please add a new python environment path to this project. For example if your cloned repository reside in `/home/username/CausalTGAN`, then one way to do this is `export PYTHONPATH="/home/username"` from command line or add it to your `~/.bashrc`. 

## To prepare a dataset for training Causal-TGAN
1. An example dataset is illustrated in `./data/real_world/adult`. <br>
2. Instructions of adding the customized dataset is following:
    * s  Identify your data type (Continuous, Discrete, Mix)

## To train a Causal-TGAN
`python train.py`. The training details such as dataset and epochs are set inside `train.py`

## To sample from Causal-TGAN
`python sampling.py`. The details such as number of samples to generate and causal-TGAN path are set inside `sampling.py`

