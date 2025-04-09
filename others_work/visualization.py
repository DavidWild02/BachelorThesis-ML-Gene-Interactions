

from utils import *
import networkx as nx
import matplotlib.pyplot as plt
from GAE import GAEModel,predict_edge_labels
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np
from scipy import stats
import random

def plot_embedding_2D(embeddings, labels=None, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'TSNE':
        reducer = TSNE(n_components=2)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Unsupported method. Choose 'PCA' or 'TSNE'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    if labels is not None:
        if len(labels) != reduced_embeddings.shape[0]:
            raise ValueError("Number of labels does not match the number of embeddings.")
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    plt.title(f'2D Embedding using {method}')
    plt.colorbar()  # Only needed if labels are provided
    plt.show()
    #save plot
    plt.savefig(f'{method}_embedding.png')





# # Plot function with subgraph sampling
# def plot_subgraph_with_edge_labels(predicted_edge_labels, batch, gene_names, num_nodes=20):
#     import matplotlib.pyplot as plt
#     import networkx as nx
#     import numpy as np

#     # Create edge weights from predicted_edge_labels
#     edge_weights = predicted_edge_labels.detach().cpu().numpy()

#     # Convert edge_index to list of edges with weights
#     edges_with_weights = [
#         (batch.edge_index[0, i].item(), batch.edge_index[1, i].item(), {"weight": edge_weights[i]})
#         for i in range(batch.edge_index.size(1))
#     ]

#     # Create the main graph
#     G = nx.DiGraph()  # Use a directed graph
#     G.add_edges_from(edges_with_weights)

#     # Sample a connected subgraph of num_nodes nodes
#     subgraph, nodes = sample_neigh(G, num_nodes)
    
#     # Filter gene names to include only nodes in the subgraph
#     subgraph_gene_names = {node: gene_names[node] for node in subgraph.nodes if node in gene_names}
    
#     # Create positions for the subgraph
#     plt.figure(figsize=(12, 10))
#     pos = nx.spring_layout(subgraph, seed=42)  # Layout only for subgraph nodes
    
#     # Create edge colors based on predicted labels
#     edge_colors = []
#     for u, v in subgraph.edges():
#         # Find the index of the edge (u, v) in the original edge_index
#         edge_idx = np.where((batch.edge_index[0] == u) & (batch.edge_index[1] == v))[0]
#         if len(edge_idx) > 0:
#             edge_color = 'green' if predicted_edge_labels[edge_idx].item() == 1 else 'orange'
#         else:
#             edge_color = 'black'  # Default color if edge is not found (shouldn't happen)
#         edge_colors.append(edge_color)

#     # Draw the subgraph with node labels and edge colors
#     nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='lightblue', edgecolors='black')
#     nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, arrowstyle='->', arrowsize=20, width=2)
#     nx.draw_networkx_labels(subgraph, pos, labels=subgraph_gene_names, font_size=10, font_family='sans-serif')
    
#     plt.title('Predicted Edge Labels in Subgraph')
#     plt.show()

    #save the plotted subgraph as png


def plot_subgraph_with_labels(predicted_edge_labels, batch, gene_names, num_nodes=20, anchor_gene_name=None, save_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    # Create a graph from edge_index
    edge_weights = predicted_edge_labels

    # Convert edge_index to list of edges with weights
    edges_with_weights = [
        (batch.edge_index[0, i].item(), batch.edge_index[1, i].item(), {"weight": edge_weights[i]})
        for i in range(batch.edge_index.size(1))
    ]

    # Create the main directed graph
    G = nx.DiGraph()  # Directed graph
    G.add_edges_from(edges_with_weights)

    # If anchor_gene_name is provided, find the corresponding node
    anchor_node = None
    if anchor_gene_name is not None:
        anchor_node = None
        # Reverse the gene_names dictionary to get a mapping from gene_name to node
        gene_name_to_node = {v: k for k, v in gene_names.items()}
        if anchor_gene_name in gene_name_to_node:
            anchor_node = gene_name_to_node[anchor_gene_name]
        else:
            raise ValueError(f"Gene name '{anchor_gene_name}' not found in the graph.")

    # Sample a connected subgraph, either around a random node or the anchor node
    subgraph, nodes = sample_neigh(G, num_nodes, anchor_node=anchor_node)

    # Filter gene names to include only nodes in the subgraph
    subgraph_gene_names = {node: gene_names[node] for node in subgraph.nodes if node in gene_names}

    # Create positions for the subgraph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph)  # Layout only for subgraph nodes

    # Create edge color based on predicted labels
    edge_colors = []
    for u, v, _ in subgraph.edges(data=True):
        # Find the index of the edge (u, v) in the original edge_index
        edge_idx = np.where((batch.edge_index[0] == u) & (batch.edge_index[1] == v))[0]
        if len(edge_idx) > 0:
            edge_color = 'red' if predicted_edge_labels[edge_idx].item() == 1 else 'blue' #green = negative, orange= positive
        else:
            edge_color = 'black'  # Default color if edge is not found (shouldn't happen)
        edge_colors.append(edge_color)
    color_map = []
    for node in subgraph.nodes:
        if len(list(subgraph.successors(node))) > 0:
            # Node has outgoing edges (successors)
            color_map.append('green')
        else:
            # Node has only incoming edges (no successors)
            color_map.append('red')
    print(color_map)
    # Draw the subgraph with node labels and edge colors
    nx.draw(subgraph, pos, with_labels=True, labels=subgraph_gene_names, node_size=400,node_color=color_map,
            edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn, width=2, arrows=True)
    plt.title('Predicted Edge Labels')

    # Save the plot if a save_path is provided
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

    # plt.savefig('~/plot/subgraph_with_predicted_edge_labels.png')

def plot_provided_subgraph_with_labels(predicted_edge_labels, batch, gene_names, node_list, save_path=None):
    edge_weights = predicted_edge_labels

    # Convert gene names to node indices
    node_indices = [node for node, name in gene_names.items() if name in sorted(node_list)]

    # Convert edge_index to list of edges with weights
    edges_with_weights = [
        (batch.edge_index[0, i].item(), batch.edge_index[1, i].item(), {"weight": edge_weights[i]})
        for i in range(batch.edge_index.size(1))
        if batch.edge_index[0, i].item() in node_indices and batch.edge_index[1, i].item() in node_indices
    ]
    
    print("Filtered edges:", edges_with_weights)  # Debugging line

    # Create the subgraph
    G = nx.DiGraph()  # Use DiGraph for directed graph
    G.add_edges_from(edges_with_weights)

    # Check if the graph has nodes before plotting
    if len(G.nodes) == 0:
        print("No nodes found in the subgraph.")
        return

    # Filter the gene names to include only the selected nodes
    subgraph_gene_names = {node: gene_names[node] for node in G.nodes if node in gene_names}
    translation_df = pd.read_excel("/Users/work/Desktop/translation_tables/sd_to_mm/sd_to_mm.xlsx")
    translation_dict = dict(zip(translation_df['Sponge gene'], translation_df['Gene name']))


    for i in translation_dict.keys():
            if translation_dict.get(i) is np.nan:
                translation_dict[i] = i

    node_ortholog_dict = {}

    for node, gene_name in subgraph_gene_names.items():
        if gene_name in translation_dict:
            node_ortholog_dict[node] = translation_dict[gene_name]

    #gene_names = {node: translation_dict[node] for node in subgraph_gene_names}

    # Create positions for the subgraph
    plt.figure(figsize=(10, 8))
    pos = nx.shell_layout(G) # spring_layout

    # Extract edge colors based on predicted labels
    edge_colors = ['green' if data['weight'] == 1 else 'orange' for _, _, data in G.edges(data=True)]

    color_map = []
    for node in G.nodes:
        if len(list(G.successors(node))) > 0:
            color_map.append('violet') # Node has outgoing edges (successors); transcription factor
        else:
            color_map.append('lightblue') # Node has only incoming edges (no successors); target gene
    # Draw the subgraph with node labels and edge colors
    nx.draw(G, pos, with_labels=True, labels=node_ortholog_dict, node_size=1000,node_color=color_map,
            edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn, width=1, arrows=True)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
    #plt.show()



def main():
    
    #load model and test data
    dataset = load_data('/Users/work/Desktop/expression_matrix_yitao.csv', '/Users/work/Desktop/suberites_presence_absence_yitao.csv')
    #load model
    train_data = dataset[5:100]
    test_data = dataset[0:5]
    data = test_data[1]
    model = GAEModel(in_channels=1, out_channels=32)
    model.load_state_dict(torch.load('model/model.pth'))
    # Assuming gene_names is a dictionary of node index to gene name
    gene_names = {i: f'SUB2.g{i+1}' for i in range(data.x.size(0))}
    # print('gene_names:', gene_names)
    # predicted_edge_labels = predict_edge_labels(model, dataset[:5],classification='cluster')
    #load predicted edge labels from csv
    predicted_edge_labels = pd.read_csv('data/predicted_edge_labels.csv').to_numpy().flatten()
    predicted_edge_labels = predicted_edge_labels.reshape(5, 82068)
    print('predicted_edge_labels:', predicted_edge_labels)
    print('predicted_edge_labels shape:', predicted_edge_labels.shape)
    
    pel = predicted_edge_labels[1]
    print('pel shape:', pel.shape)
    print('pel[10]:', pel[:10])
    embeddings = []

    for i in range(len(test_data)):
        z = model.encode(train_data[i].x, train_data[i].edge_index)
        embeddings.append(model.get_edge_embeddings(z,train_data[i].edge_index))
        
    embeddings = torch.cat(embeddings, dim=0).cpu().detach().numpy()
    # #save edge_index to csv
    # print('embeddings shape:', embeddings.shape)
    # print('predicted_edge_labels shape:', predicted_edge_labels.shape)
    # edge_index = data.edge_index.cpu().detach().numpy()
    # np.savetxt('edge_index.csv', edge_index, delimiter=',')
#    node_list = ["SUB2.g5379", "SUB2.g836", "SUB2.g1511", "SUB2.g2584", "SUB2.g2624", "SUB2.g2626", "SUB2.g2986", "SUB2.g3283", "SUB2.g3755", "SUB2.g4142", "SUB2.g4347", "SUB2.g4372", "SUB2.g4822", "SUB2.g4981", "SUB2.g5279", "SUB2.g5319", "SUB2.g5363", "SUB2.g5451", "SUB2.g5694", "SUB2.g5915", "SUB2.g6096", "SUB2.g6641", "SUB2.g6663", "SUB2.g7027", "SUB2.g7114", "SUB2.g7342", "SUB2.g8518", "SUB2.g8571", "SUB2.g8842", "SUB2.g9133", "SUB2.g9219", "SUB2.g9989", "SUB2.g10345", "SUB2.g10354", "SUB2.g10442", "SUB2.g10451", "SUB2.g10520", "SUB2.g10596", "SUB2.g10794", "SUB2.g11287", "SUB2.g11343", "SUB2.g11521", "SUB2.g11598", "SUB2.g11627", "SUB2.g11876", "SUB2.g11992", "SUB2.g12540", "SUB2.g12665", "SUB2.g12869"]
    node_list = ["SUB2.g8259","SUB2.g1976"] + list(pd.read_csv("../gata_runx.csv")["gene_ids"])[:20]


        # plot_embedding_2D(embeddings[:82068], labels=predicted_edge_labels.cpu().detach().numpy()[:82068], method='TSNE')
   # plot the subgraph with predicted edge labels
    # plot_subgraph_with_edge_labels(pel,data,gene_names)
    # plot_subgraph_with_labels(pel,data,gene_names, num_nodes=30, anchor_gene_name='SUB2_g5379', save_path='/Users/caiyitao/Documents/观澜阁/TF_Gene_Interaction_Prediction/plot/subgraph_with_predicted_edge_labels.png')
    plot_provided_subgraph_with_labels(pel,data,gene_names, node_list, save_path='plot/provided_subgraph.png')


if __name__ == '__main__':
    main()
 #   dataset = load_data('data/expression_matrix.csv', '/Users/work/Downloads/suberites_presence_absence.csv')
 