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
- An example dataset is illustrated in `./data/real_world/adult`. <br>

- Instructions of adding the customized dataset is following:
	- Step 1: Add your dataset name and name of column information into `helper/constant.py`. For example:
	```python
	DATASETS = ['asia', 'alarm', 'child', 'insurance', 'your dataset name']
	YOUR_DATASET_NAME = ['column_1', 'column_2', 'column_3']

	```
	
	- Step 2: Check your dataset type (Continuous, Discrete, Mix) and add the dataset type information into `check_BN_datatype()` function in `helper/utils.py`. For example, if your dataset is Continuous type dataset, you can modify the code as following:
	
	```python
	def check_BN_datatype(data_name):
	...
	
          if data_name in ['your_dataset_name']:
              return 'continuous'
	      
	...
	```
	
	- Step 3 (Optional): If customized dataset is Continuous type, please declare Discrete Column in `get_discrete_cols()` function in `helper/utils.py`. For example,
	```python
	def get_discrete_cols(data, data_name):
	...
	
	  if data_name == 'your_dataset_name':
          discrete_cols = Discrete_Column_Name
	  
	...
	```
	
	- Step 4: Prepare your causal graph by using **Tetrad**. You can go to https://www.ccd.pitt.edu/tools/ and click [Tetrad Latest Executable] to download **Tetrad**. We have created a tutorial on how to use **Tetrad** to generate causal graphs. See in [Tetrad_Tutorial.pdf](Tetrad_Tutorial.pdf) <br> 
	           After this steps, you should have a 'graph_tetrad.txt' file. 
		   
	- Step 5: Use `helper/graphFromTetrad.py` to convert Tetrad graph to common graph data and save it as 'graph.txt'. 

- Now, you have completed the preparation steps for the dataset. Your dataset folder structure should look like the following:

```
|-data
|  |-customized_dataset
|     |-graph.txt
|     |-train.csv
|     |-test.csv

```
Then, you can use the `load_data_graph(args.data_name)` function in `train.py` to load your customized datasets and train model.


## To train a Causal-TGAN
`python train.py`. The training details such as dataset and epochs are set inside `train.py`

## To sample from Causal-TGAN
`python sampling.py`. The details such as number of samples to generate and causal-TGAN path are set inside `sampling.py`

