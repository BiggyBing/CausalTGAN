class CausalTGANConfig(object):
    def __init__(self, causal_graph,
                 z_dim, pac_num, D_iter):
        self.causal_graph = causal_graph
        self.z_dim = z_dim
        self.pac_num = pac_num
        self.D_iter = D_iter

class CondGANConfig(object):
    def __init__(self, causal_graph, col_names, col_dims,
                 z_dim=128, pac_num=10, D_iter=5):
        self.causal_graph = causal_graph
        self.col_names = col_names
        self.col_dims = col_dims
        self.z_dim = z_dim
        self.pac_num = pac_num
        self.D_iter = D_iter

class TrainingOptions:
    """
    Configuration options for the training
    """
    def __init__(self,
                 batch_size, number_of_epochs,
                 runs_folder,
                 experiment_name):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.runs_folder = runs_folder
        self.experiment_name = experiment_name
