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
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from absl import app, flags
import time
import datetime

# Atom Featurisation
## Auxiliary function for one-hot enconding transformation based on list of
##permitted values

FLAGS = flags.FLAGS

flags.DEFINE_string(name="test_csv_for_ids_path", default="../data/test.csv", help="")
flags.DEFINE_string(name="test_csv_path", default="../shrunken_data/test.csv", help="")
flags.DEFINE_string(name="train_file_csv_path", default="../shrunken_data/train.csv", help="")
flags.DEFINE_string(name="test_file_path", default="../gnn_data/test.pt", help="")
flags.DEFINE_string(name="valid_file_path", default="../gnn_data/valid.pt", help="")
flags.DEFINE_string(name="results_dir", default="results", help="")
flags.DEFINE_string(name="output_file_name", default="ids_pred_results.csv", help="")
flags.DEFINE_string(name="saving_model_path", default="ids_pred_results.csv", help="")



flags.DEFINE_integer(name="featurizing_batch_size", default=256, help="")
flags.DEFINE_integer(name="shard_size", default=100000, help="")
flags.DEFINE_integer(name="dev_num", default=0, help="")




def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding
    
    
# Main atom feat. func

def get_atom_features(atom, use_chirality=True):
    # Define a simplified list of atom types
    permitted_atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I','Dy', 'Unknown']
    atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
    atom_type_enc = one_hot_encoding(atom_type, permitted_atom_types)
    
    # Consider only the most impactful features: atom degree and whether the atom is in a ring
    atom_degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour'])
    is_in_ring = [int(atom.IsInRing())]
    
    #print(atom_degree)
    #exit()
    # Optionally include chirality
    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_features = atom_type_enc + atom_degree + is_in_ring + chirality_enc
    else:
        atom_features = atom_type_enc + atom_degree + is_in_ring
    
    return np.array(atom_features, dtype=np.float32)

# Bond featurization

def get_bond_features(bond):
    # Simplified list of bond types
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, 'Unknown']
    bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else 'Unknown'
    
    # Features: Bond type, Is in a ring
    features = one_hot_encoding(bond_type, permitted_bond_types) \
               + [int(bond.IsInRing())]
    
    return np.array(features, dtype=np.float32)


def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y=None):
    data_list = []
    
    for index, smiles in enumerate(x_smiles):
        mol = Chem.MolFromSmiles(smiles)
        
        if not mol:  # Skip invalid SMILES strings
            continue
        
        # Node features
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        #atom_features = np.stack( atom_features, axis=0 )
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge features
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [(start, end), (end, start)]  # Undirected graph
            bond_feature = get_bond_features(bond)
            edge_features += [bond_feature, bond_feature]  # Same features in both directions
        
        #edge_index = np.stack( edge_index, axis=0 )
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        #edge_features = np.stack( edge_features, axis=0 )
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Creating the Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        #data.molecule_id = ids[index]
        if y is not None:
            data.y = torch.tensor([y[index]], dtype=torch.float)
        
        data_list.append(data)
    
    return data_list

def featurize_data_in_batches(smiles_list, labels_list, batch_size):
    data_list = []
    # Define tqdm progress bar
    pbar = tqdm(total=len(smiles_list), desc="Featurizing data")
    for i in range(0, len(smiles_list), batch_size):
        smiles_batch = smiles_list[i:i+batch_size]
        if labels_list is not None:
            labels_batch = labels_list[i:i+batch_size]
        else:
            labels_batch = None
        #ids_batch = ids_list[i:i+batch_size]
        batch_data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles_batch, labels_batch)
        data_list.extend(batch_data_list)
        pbar.update(len(smiles_batch))
        
    pbar.close()
    return data_list


class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGNNLayer, self).__init__(aggr='max')
        self.lin = nn.Linear(in_channels + 6, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return MessagePassing.propagate(self, edge_index
                                        , x=x, edge_attr=edge_attr)

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




def train_model(df_train, num_epochs,model, lr,results_dir,valid_loader,
                featurizing_batch_size=64,training_batch_size=32,shard_size=10000,shuffle = True,device = 'cuda:1',es_patience = 3,):
    
    # define model and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()
    
    start_time = time.time()

    
    best_val = None
    patience = es_patience 
    val_ap_list = []
    val_roc_list = []

    # iteration over the training samples
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        if shuffle:
            df_train = df_train.sample(frac=1)
            
        shard_ind = 0
        num_mini_batches = 0
        # iteration over shard_size of the training samples
        for i in range(0, len(df_train), shard_size):
            shard_ind +=1
            print(f"featurizing shard {shard_ind} / {len(df_train)// shard_size}")
            df_train_shard = df_train[i:i+shard_size]
            smiles_list_train = df_train_shard['molecule_smiles'].tolist()
            labels_list = df_train_shard[['binds_BRD4','binds_HSA','binds_sEH']].values
            train_data = featurize_data_in_batches(smiles_list_train, labels_list, featurizing_batch_size)
            
            train_loader = DataLoader(train_data, batch_size=training_batch_size, shuffle=True)

            # iteration over the train_loader mini-batchs
            print(f"Training on shard {shard_ind} / {len(df_train)// shard_size}")
            for batch in train_loader:
                num_mini_batches +=1
                optimizer.zero_grad()
                batch = batch.to(device)
                out = model(batch)
                #loss = criterion(out, batch.y.view(-1, 1).float()) # ??
                loss = criterion(out, batch.y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
 
 
        # validation
        val_predictions, labels= predict_with_model(model, valid_loader,device,is_labeled=True)
        metrics = {
                    "roc_auc_score": roc_auc_score(y_true=labels, y_score=val_predictions),
                    "average_precision_score": average_precision_score(
                        y_true=labels, y_score=val_predictions,average='micro'
                    ),
                }
        
        # During the first iteration (first epoch) best validation is set to None
        val_ap = metrics["average_precision_score"]
        val_roc = metrics["roc_auc_score"]
        val_ap_list.append(val_ap)
        val_roc_list.append(val_roc)
        
        #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / num_mini_batches}')
        print('Epoch {:03}: | Loss: {:.5f} | Val Ap: {:.5f} | Val roc: {:.5f}  | Training time: {}'.format(
            epoch + 1, 
            total_loss/num_mini_batches, 
            val_ap, 
            val_roc,
            str(datetime.timedelta(seconds=time.time() - start_time))[:7]))


        if not best_val:
            best_val = val_ap  # So any validation roc_auc we have is the best one for now
            torch.save(model.state_dict(), os.path.join(results_dir,'best_val.pth'))  # Saving the model
            #torch.save(model.state_dict(),
            #               f"Fold{fold}_Epoch{epoch+1}_ValidAcc_{val_acc:.3f}_ROC_{val_roc:.3f}.pth")
            continue
            
        if val_ap >= best_val:
            best_val = val_ap
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            torch.save(model.state_dict(), os.path.join(results_dir,'best_val.pth'))  # Saving the model 
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val roc_auc: {:.4f}'.format(best_val))
                break  

      
    
    torch.save(val_ap_list, os.path.join(results_dir,'val_ap_list.pt'))  
    torch.save(val_roc_list, os.path.join(results_dir,'val_roc_list.pt'))  
    return model

def predict_with_model(model, test_loader,device,is_labeled = False):
    model.eval()
    predictions = []
    if is_labeled:
        true_labels = []
    #molecule_ids = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            output = torch.sigmoid(model(data.to(device)))
            predictions.extend(output.view(-1).tolist())
            if is_labeled:
                true_labels.extend(data.y.view(-1).tolist())
            #molecule_ids.extend(data.molecule_id)
    if is_labeled:
        return predictions,true_labels
    else:
        return predictions

def select_and_save_predictions_with_ids(predictions,test_df,path_test_file_for_ids,results_dir = 'results/',output_file_name ='ids_pred_results.csv'):
    
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
    
    output_path = os.path.join(results_dir,output_file_name)
    y_pred_and_ids_df.to_csv(output_path,index=False)
    return y_pred_and_ids_df

def main(argv):
    print('loading training samples')    

    df_train = pd.read_csv(FLAGS.train_file_csv_path)
    
    # simple featurization to get input dimension
    df_train_simple = df_train[:128]
    smiles_list_train = df_train_simple['molecule_smiles'].tolist()
    labels_list = df_train_simple[['binds_BRD4','binds_HSA','binds_sEH']].values
    train_data = featurize_data_in_batches(smiles_list_train, labels_list, 64)
    
    # ---hyper-parameters
    input_dim = train_data[0].num_node_features
    hidden_dim = 64
    num_epochs = 20
    num_layers = 4 #Should ideally be set so that all nodes can communicate with each other
    dropout_rate = 0.3
    lr = 0.001
    out_channels =3
    
    device = "cuda:"+str(FLAGS.dev_num)
    # define model
    model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate,out_channels).to(device)

    os.makedirs(FLAGS.results_dir, exist_ok=True)

    print('loading validation samples')    
    #------ validation set loading
    valid_data = torch.load(FLAGS.valid_file_path)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

    # --- Training
    #model = train_model(train_loader,num_epochs, input_dim, hidden_dim,num_layers, dropout_rate,out_channels, lr)
    model = train_model(df_train=df_train, model=model,valid_loader=valid_loader, lr=lr,num_epochs=num_epochs,featurizing_batch_size=FLAGS.featurizing_batch_size
                        ,training_batch_size=32,shard_size=FLAGS.shard_size,shuffle = True,device = device,results_dir = FLAGS.results_dir)
    
    
    
    # ----- Test set predictions ---
    if FLAGS.test_file_path is not None:
        print('Predicting on test samples')    
        print('loading test samples')    
        test_data = torch.load(FLAGS.test_file_path)
        test_df = pd.read_csv(FLAGS.test_csv_path)
        
        # Predict
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        predictions = predict_with_model(model, test_loader,device=device)
        
        # save predictions
        _ = select_and_save_predictions_with_ids(predictions,test_df,FLAGS.test_csv_for_ids_path,results_dir = FLAGS.results_dir
                                                ,output_file_name = FLAGS.output_file_name)

        
if __name__ == "__main__":
    app.run(main)