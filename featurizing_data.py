import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from rdkit import Chem
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from absl import app, flags


# Atom Featurisation
## Auxiliary function for one-hot enconding transformation based on list of
##permitted values

FLAGS = flags.FLAGS

flags.DEFINE_string(name="train_file_path", default="../shrunken_data/train.csv", help="")
flags.DEFINE_string(name="test_file_path", default="../shrunken_data/test.csv", help="")
flags.DEFINE_string(name="output_folder", default="../gnn_data/", help="")
flags.DEFINE_string(name="output_train_file_name", default="train.pt", help="")
flags.DEFINE_string(name="output_test_file_name", default="test.pt", help="")
flags.DEFINE_string(name="data_type", default="both", help="type of data to featurize both, train_only, test_only")

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
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge features
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [(start, end), (end, start)]  # Undirected graph
            bond_feature = get_bond_features(bond)
            edge_features += [bond_feature, bond_feature]  # Same features in both directions
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
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


def main(argv):
    dtypes = {'buildingblock1_smiles': np.int16, 'buildingblock2_smiles': np.int16, 'buildingblock3_smiles': np.int16,
            'binds_BRD4':np.byte, 'binds_HSA':np.byte, 'binds_sEH':np.byte}
      
    
    os.makedirs(FLAGS.output_folder, exist_ok=True)

    if FLAGS.data_type =='both' or FLAGS.data_type =='train_only': 
        print('loading training samples')    
        df_train = pd.read_csv(FLAGS.train_file_path, dtype = dtypes)
        #df_train = train[:3000]
        print('Num training sampls: ',len(df_train))  
        print('----Featurizing training data -----')
        # Define the batch size for featurization
        batch_size = 2**8
        smiles_list_train = df_train['molecule_smiles'].tolist()
        labels_list = df_train[['binds_BRD4','binds_HSA','binds_sEH']].values
        train_data = featurize_data_in_batches(smiles_list_train, labels_list, batch_size)
        torch.save(train_data,os.path.join(FLAGS.output_folder,FLAGS.output_train_file_name))
    if FLAGS.data_type =='both' or FLAGS.data_type =='test_only': 
        print('loading testing samples')    
        df_test = pd.read_csv(FLAGS.test_file_path, dtype = dtypes)
        #df_test= test
        print('Num testnig samples: ',len(df_test))
        print('----Featurizing test data -----')
        # Define the batch size for featurization
        smiles_list_test = df_test['molecule_smiles'].tolist()
        #labels_list = test[['binds_BRD4','binds_HSA','binds_sEH']].values
        test_data = featurize_data_in_batches(smiles_list_test, None, batch_size)
        torch.save(test_data,os.path.join(FLAGS.output_folder,FLAGS.output_test_file_name))

    #print('----saving Featurized data -----')
    
    
        
if __name__ == "__main__":
    app.run(main)