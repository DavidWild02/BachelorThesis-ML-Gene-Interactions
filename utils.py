import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch,DataLoader
import pandas as pd
import numpy as np


def load_data(node_feature_path, edge_incidence_path):
    # 加载节点特征（每个细胞的基因表达水平）
    node_features = pd.read_csv(node_feature_path, index_col=0)
    # node_features = node_features.iloc[:100, :100]
    # print('node_features', node_features.head())
    # 加载边关联矩阵（对所有细胞相同）
    edge_incidence = pd.read_csv(edge_incidence_path, index_col=0).T

    # edge_incidence = edge_incidence.iloc[:50, :100]
    # print('edge_incidence', edge_incidence.head())
    edge_sum = sum(edge_incidence.to_numpy())
    # print('edge index matrix',edge_incidence.to_numpy())
    # print('edge_sum', edge_sum.sum())
    # 创建节点索引映射，使用节点特征矩阵的列顺序
    node_index_map = {gene: idx for idx, gene in enumerate(node_features.columns)}
    # print('node_index_map', dict(list(node_index_map.items())[:10]))

    # 创建 edge_index（对所有细胞相同）
    edge_index = []
    for col in edge_incidence.columns:
        # print('col', col)
        if col in node_index_map:
            # print('node index map of col', node_index_map[col])
            target_idx = node_index_map[col]
            # print('target idx', target_idx)
            source_genes = edge_incidence.index[edge_incidence[col] == 1].tolist()
            # print('source genes', source_genes)
            for source in source_genes:
                if source in node_index_map:
                    source_idx = node_index_map[source]
                    # print('source idx', source_idx)
                    edge_index.append([source_idx, target_idx])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print('edge_index', edge_index)
    print('edge_index shape', edge_index.shape)
    # 为每个细胞创建 Data 对象列表
    data_list = []
    for _, row in node_features.iterrows():
        x = torch.tensor(row.values, dtype=torch.float).unsqueeze(1)  # 形状：[num_genes, 1]
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    return data_list



# Function to sample a connected neighborhood subgraph
def sample_neigh(graph, size, anchor_node=None):
    import random
    from scipy import stats
    
    # Use float instead of np.float
    ps = np.array([len(graph)], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(ps)), ps))

    while True:
        if anchor_node is None:
            # If no anchor node is provided, choose a random starting node
            idx = dist.rvs()  # Select graph
            start_node = random.choice(list(graph.nodes))  # Random starting node
        else:
            # If an anchor node is provided, use it as the starting point
            if anchor_node not in graph.nodes:
                raise ValueError(f"Anchor node '{anchor_node}' not found in the graph.")
            start_node = anchor_node
        
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])

        # Perform breadth-first search to sample neighbors
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]

        if len(neigh) == size:
            return graph.subgraph(neigh), neigh


# Function to sample a connected neighborhood subgraph
# def sample_neigh(graph, size):
#     import random
#     from scipy import stats
    
#     # Use float instead of np.float
#     ps = np.array([len(graph)], dtype=float)  
#     ps /= np.sum(ps)
#     dist = stats.rv_discrete(values=(np.arange(len(ps)), ps))

#     while True:
#         idx = dist.rvs()  # Select graph
#         start_node = random.choice(list(graph.nodes))  # Random starting node
#         neigh = [start_node]
#         frontier = list(set(graph.neighbors(start_node)) - set(neigh))
#         visited = set([start_node])

#         # Perform breadth-first search to sample neighbors
#         while len(neigh) < size and frontier:
#             new_node = random.choice(list(frontier))
#             assert new_node not in neigh
#             neigh.append(new_node)
#             visited.add(new_node)
#             frontier += list(graph.neighbors(new_node))
#             frontier = [x for x in frontier if x not in visited]

#         if len(neigh) == size:
#             return graph.subgraph(neigh), neigh

 