import pickle

def convert_tetrad_graph(graph_path, save_path):
    with open(graph_path, 'r') as f:
        node_flag = False
        for line in f:
            if node_flag:
                vars = line.strip().split(';')
                vals = [[] for _ in range(len(vars))]
                graph_dict = dict(zip(vars, vals))
                node_flag = False
            if line.strip() == 'Graph Nodes:':
                node_flag = True

            line_content = line.strip()
            if ' --> ' in line_content:
                dot_idx = line_content.index('.')
                line_content = line_content[dot_idx + 2:]
                cause_effct = line_content.split(' --> ')
                graph_dict[cause_effct[1]].append(cause_effct[0])

    graph_output = [[k, v] for k, v in graph_dict.items()]
    with open(save_path, 'wb') as f:
        pickle.dump(graph_output, f)

# demo below
# graph_path = '../data/real_world/adult/graph_tetrad.txt'
# save_path = '../data/real_world/adult/graph_demo.txt'
# convert_tetrad_graph(graph_path, save_path)

