import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from absl import app, flags


# Atom Featurisation
## Auxiliary function for one-hot enconding transformation based on list of
##permitted values

FLAGS = flags.FLAGS

flags.DEFINE_string(name="test_csv_for_ids_path", default="../data/test.csv", help="")
flags.DEFINE_string(name="test_csv_path", default="../shrunken_data/test.csv", help="")
flags.DEFINE_string(name="train_file_path", default="../gnn_data/train.pt", help="")
flags.DEFINE_string(name="test_file_path", default="../gnn_data/test.pt", help="")
flags.DEFINE_string(name="results_dir", default="results", help="")



class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGNNLayer, self).__init__(aggr='max')
        self.lin = nn.Linear(in_channels + 6, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return MessagePassing.propagate(self, edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat((x_j, edge_attr), dim=1)
        return combined

    def update(self, aggr_out):
        return self.lin(aggr_out)

#Define GNN Model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate,out_channels=1):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([CustomGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)


        x = global_max_pool(x, data.batch) # Global pooling to get a graph-level representation
        x = self.lin(x)
        return x




def train_model(train_loader, num_epochs, input_dim, hidden_dim, num_layers, dropout_rate,out_channels, lr):
    model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate,out_channels)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            #loss = criterion(out, batch.y.view(-1, 1).float()) # ??
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')
    
    return model

def predict_with_model(model, test_loader):
    model.eval()
    predictions = []
    #molecule_ids = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            output = torch.sigmoid(model(data))
            predictions.extend(output.view(-1).tolist())
            #molecule_ids.extend(data.molecule_id)

    return predictions

def select_and_save_predictions_with_ids(predictions,test_df,path_test_file_for_ids,output_dir = 'results/',output_file_name ='ids_pred_results.csv'):
    
    #Combine predictions with the bools
    bool_cols  = test_df[['is_BRD4','is_HSA','is_sEH']]
    bool_cols = np.array(bool_cols).reshape(-1)
    y_pred_and_bools =  np.vstack((bool_cols,predictions)).T
    y_pred_and_bools_df = pd.DataFrame({'Bool': y_pred_and_bools[:, 0], 'binds': y_pred_and_bools[:, 1]})
    
    # drop predictions of protiens not in the test set and also drop the bool column
    y_pred_and_bools_df = y_pred_and_bools_df[y_pred_and_bools_df.Bool != 0]
    y_pred_df = y_pred_and_bools_df.drop(['Bool'],axis = 1)
    y_pred_df = y_pred_df.reset_index(drop=True)

    #read the test ids
    test = pd.read_csv(path_test_file_for_ids,index_col=False)#[:len(y_pred_df)]
    test_ids = pd.DataFrame(test.id)
    assert len(y_pred_df)==len(test_ids)
    y_pred_and_ids_df = pd.concat([test_ids,y_pred_df],axis=1)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,output_file_name)
    y_pred_and_ids_df.to_csv(output_path,index=False)
    return y_pred_and_ids_df

def main(argv):
    print('loading training samples')    
    train_data = torch.load(FLAGS.train_file_path)
    # Train model
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    input_dim = train_loader.dataset[0].num_node_features
    hidden_dim = 64
    num_epochs = 11
    num_layers = 4 #Should ideally be set so that all nodes can communicate with each other
    dropout_rate = 0.3
    lr = 0.001
    out_channels =3
    #These are just example values, feel free to play around with them.
    model = train_model(train_loader,num_epochs, input_dim, hidden_dim,num_layers, dropout_rate,out_channels, lr)

    print('loading test samples')    
    test_data = torch.load(FLAGS.test_file_path)
    test_df = pd.read_csv(FLAGS.test_csv_path)
    # Predict
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    predictions = predict_with_model(model, test_loader)
    
    # save predictions
    
    _ = select_and_save_predictions_with_ids(predictions,test_df,FLAGS.test_csv_for_ids_path,output_dir = FLAGS.results_dir)

        
if __name__ == "__main__":
    app.run(main)