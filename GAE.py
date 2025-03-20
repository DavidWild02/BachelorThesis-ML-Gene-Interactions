import torch
from torch.utils.data import random_split
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch, DataLoader
import pandas as pd
import numpy as np
from utils import *
import wandb
import matplotlib.pyplot as plt


class GAEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAEModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        
        # Decoder that infers the edge label based on node similarity
        self.decoder = torch.nn.Bilinear(out_channels, out_channels, 1)  # Bilinear layer for prediction

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        # Use the embeddings of node pairs (i, j) to predict the label of edge (positive/negative)
        # print('z shape',z.shape)
        src = z[edge_index[0]]  # Source node embeddings
        # print('src shape',src.shape)
        dst = z[edge_index[1]]  # Target node embeddings
        # print('dst shape',dst.shape)
        return torch.sigmoid(self.decoder(src, dst))  # Predict edge type using Bilinear layer

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)  # Get node embeddings
        out = self.decode(z, edge_index)
        return out  # Predict edge labels
    
    def get_edge_embeddings(self, z, edge_index):
        src = z[edge_index[0]]  # Source node embeddings
        dst = z[edge_index[1]]  # Target node embeddings
        
        # You can combine the src and dst embeddings in different ways:
        edge_embeddings = torch.cat([src, dst], dim=1)  # Concatenate source and target embeddings
        
        return edge_embeddings



def train(model, dataset, num_epochs=6):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)

    # Split dataset into training and validation sets
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError("The dataset is too small to be split into training and validation sets.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # print('train_loader length:', len(train_loader))
    # print('val_loader length:', len(val_loader))

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            edge_predictions = model(data.x, data.edge_index).squeeze()
            
            # print('edge_predictions',edge_predictions)
            # Use current predictions as soft labels for the next epoch
            soft_labels = edge_predictions.detach().clone()
            # print('soft_labels',soft_labels)
            # Compute the loss using the soft labels
            loss = criterion(edge_predictions, soft_labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({"loss": loss.item()})
        # Evaluate on validation loader
        evaluate(model, criterion, val_loader)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
            

    return model

def evaluate(model, criterion, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            edge_predictions = model(data.x, data.edge_index).squeeze()
            
            # Use current predictions as soft labels for the next epoch
            soft_labels = edge_predictions.detach().clone()
            
            # Compute the loss using the soft labels
            loss = criterion(edge_predictions, soft_labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
    wandb.log({"val_loss": val_loss})



def estimate_threshold_from_distribution(edge_predictions):
    plt.hist(edge_predictions.cpu().detach().numpy(), bins=50, alpha=0.75)
    plt.title('Distribution of Edge Predictions')
    # plt.show()
    
    # Example threshold: Use the median as the threshold
    median_threshold = torch.median(edge_predictions).item()
    return median_threshold




def cluster_embeddings(z):
    # Assuming `z` are the edge embeddings obtained from model.encode()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(z.cpu().detach().numpy())
    return kmeans.labels_  # Returns cluster labels for each edge



def predict_edge_labels(model, test_dataset,classification='threshold'):
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    predicted_edge_labels = []

    for data in loader:
        # data = data[0] 
        # print('data',data)
        # print('data[1]',data[1]) # Extract single Data object from batch if DataLoader is used
        with torch.no_grad():
            if classification== 'threshold':
                edge_predictions = model(data.x, data.edge_index).squeeze()
                # binary_predictions = (edge_predictions > 0.5).float()
                threshold = estimate_threshold_from_distribution(edge_predictions)
                binary_predictions = (edge_predictions > threshold).float()
                print('edge_predictions shape',edge_predictions.shape)
                print('binary_predictions shape',binary_predictions.shape)
                print('binary_predictions',binary_predictions)
                predicted_edge_labels.append(binary_predictions)       
            elif classification == 'cluster':
                            
                z = model.encode(data.x, data.edge_index)

                # Extract edge embeddings
                edge_embeddings = model.get_edge_embeddings(z, data.edge_index)

# Cluster the edge embeddings
                cluster_labels = cluster_embeddings(edge_embeddings)
                print('edge_embedding shape',edge_embeddings.shape)
               
                predicted_edge_labels.append(cluster_labels)
                print('cluster_labels',cluster_labels)
                print('cluster_labels shape',cluster_labels.shape)
    # Stack or concatenate all the binary predictions into a single tensor
    if len(predicted_edge_labels) > 0:
        # Ensure all predictions are tensors and have the same shape
        tensor_list = [torch.tensor(pred) for pred in predicted_edge_labels]
        return torch.cat(tensor_list, dim=0)
    else:
        return torch.empty(0) 





#

  #main
if __name__ == "__main__":
    wandb.init(project="GAE")
    
    dataset = load_data('/Users/work/Desktop/expression_matrix_yitao.csv', '/Users/work/Desktop/suberites_presence_absence_yitao.csv')
    
    # loader = create_pyg_loader(data_list,batch_size=50)

    num_epochs = 30
                                     
    model = GAEModel(in_channels=1, out_channels=32)
    
    model = train(model, dataset[5:100],num_epochs)
    # torch.save(model.state_dict(), 'model/model.pth')

    predicted_edge_labels = predict_edge_labels(model, dataset[:5],classification='cluster')
    print('predicted_edge_labels shape for test dataset',predicted_edge_labels.shape)
    #save predicted_edge_labels as csv
    pd.DataFrame(predicted_edge_labels.cpu().detach().numpy(), columns=['predicted_edge_labels']).to_csv('predicted_edge_labels.csv', index=False)
    print('1 predicted_edge_labels ',predicted_edge_labels[0:10])
    print('predicted_label shape', predicted_edge_labels.shape  )

 








   