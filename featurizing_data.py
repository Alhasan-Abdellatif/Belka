import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from rdkit import Chem
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from absl import app, flags
from multiprocessing import Pool


# Atom Featurisation
## Auxiliary function for one-hot enconding transformation based on list of
##permitted values

FLAGS = flags.FLAGS

flags.DEFINE_string(name="csv_file_path", default="../shrunken_data/train.csv", help="")
flags.DEFINE_string(name="output_folder", default="../gnn_data/", help="")
flags.DEFINE_string(name="output_file_name", default="train.pt", help="")

flags.DEFINE_boolean(
    name="is_labeled",
    default=False,
    help=""
)

# torch version of np unpackbits
#https://gist.github.com/vadimkantorov/30ea6d278bc492abf6ad328c6965613a

def tensor_dim_slice(tensor, dim, dim_slice):
	return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]

# @torch.jit.script
def packshape(shape, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8, pack=True):
	dim = dim if dim >= 0 else dim + len(shape)
	bits, nibble = (
		8 if dtype is torch.uint8 else 16 if dtype is torch.int16 else 32 if dtype is torch.int32 else 64 if dtype is torch.int64 else 0), (
		1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0)
	# bits = torch.iinfo(dtype).bits # does not JIT compile
	assert nibble <= bits and bits % nibble == 0
	nibbles = bits // nibble
	shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)),) + shape[1 + dim:]) if pack else (
				shape[:dim] + (shape[dim] * nibbles,) + shape[1 + dim:])
	return shape, nibbles, nibble

# @torch.jit.script
def F_unpackbits(tensor, dim: int = -1, mask: int = 0b00000001, shape=None, out=None, dtype=torch.uint8):
	dim = dim if dim >= 0 else dim + tensor.dim()
	shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=tensor.dtype, pack=False)
	shape = shape if shape is not None else shape_
	out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
	assert out.shape == shape

	if shape[dim] % nibbles == 0:
		shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
		shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
		return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

	else:
		for i in range(nibbles):
			shift = nibble * i
			sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
			sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
			torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
	return out

class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)


# mol to graph adopted from
# from https://github.com/LiZhang30/GPCNDTA/blob/main/utils/DrugGraph.py

PACK_NODE_DIM=9
PACK_EDGE_DIM=1
NODE_DIM=PACK_NODE_DIM*8
EDGE_DIM=PACK_EDGE_DIM*8

def one_of_k_encoding(x, allowable_set, allow_unk=False):
	if x not in allowable_set:
		if allow_unk:
			x = allowable_set[-1]
		else:
			raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
	return list(map(lambda s: x == s, allowable_set))


#Get features of an atom (one-hot encoding:)
'''
	1.atom element: 44+1 dimensions    
	2.the atom's hybridization: 5 dimensions
	3.degree of atom: 6 dimensions                        
	4.total number of H bound to atom: 6 dimensions
	5.number of implicit H bound to atom: 6 dimensions    
	6.whether the atom is on ring: 1 dimension
	7.whether the atom is aromatic: 1 dimension           
	Total: 70 dimensions
'''

ATOM_SYMBOL = [
	'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
	'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
	'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
	'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
	'Pt', 'Hg', 'Pb', 'Dy',
	#'Unknown'
]
#print('ATOM_SYMBOL', len(ATOM_SYMBOL))44
HYBRIDIZATION_TYPE = [
	Chem.rdchem.HybridizationType.S,
	Chem.rdchem.HybridizationType.SP,
	Chem.rdchem.HybridizationType.SP2,
	Chem.rdchem.HybridizationType.SP3,
	Chem.rdchem.HybridizationType.SP3D
]

def get_atom_feature(atom):
	feature = (
		 one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL)
	   + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
	   + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
	   + [atom.IsInRing()]
	   + [atom.GetIsAromatic()]
	)
	#feature = np.array(feature, dtype=np.uint8)
	#print(torch.tensor(feature,dtype=int))
	feature = np.packbits(feature)
	#print(feature)
	#print(F_unpackbits(torch.tensor(feature)))
	if torch.any(F_unpackbits(torch.tensor(feature))[-3:] != 0):
		raise Exception('Problem unpacking')
	return feature


#Get features of an edge (one-hot encoding)
'''
	1.single/double/triple/aromatic: 4 dimensions       
	2.the atom's hybridization: 1 dimensions
	3.whether the bond is on ring: 1 dimension          
	Total: 6 dimensions
'''

def get_bond_feature(bond):
	bond_type = bond.GetBondType()
	feature = [
		bond_type == Chem.rdchem.BondType.SINGLE,
		bond_type == Chem.rdchem.BondType.DOUBLE,
		bond_type == Chem.rdchem.BondType.TRIPLE,
		bond_type == Chem.rdchem.BondType.AROMATIC,
		bond.GetIsConjugated(),
		bond.IsInRing()
	]
	#feature = np.array(feature, dtype=np.uint8)
	feature = np.packbits(feature)
	return feature


def smile_to_graph(smiles):
	mol = Chem.MolFromSmiles(smiles)
	N = mol.GetNumAtoms()
	node_feature = []
	edge_feature = []
	edge = []
	for i in range(mol.GetNumAtoms()):
		atom_i = mol.GetAtomWithIdx(i)
		atom_i_features = get_atom_feature(atom_i)
		node_feature.append(atom_i_features)

		for j in range(mol.GetNumAtoms()):
			bond_ij = mol.GetBondBetweenAtoms(i, j)
			if bond_ij is not None:
				edge.append([i, j])
				bond_features_ij = get_bond_feature(bond_ij)
				edge_feature.append(bond_features_ij)
	node_feature=np.stack(node_feature)
	edge_feature=np.stack(edge_feature)
	edge = np.array(edge,dtype=np.uint8)
	return N,edge,node_feature,edge_feature

def to_pyg_format(N,edge,node_feature,edge_feature):
	graph = Data(
		idx=-1,
		edge_index = torch.from_numpy(edge.T).int(),
		x          = torch.from_numpy(node_feature).byte(),
		edge_attr  = torch.from_numpy(edge_feature).byte(),
	)
	return graph

def to_pyg_list(graph,labels=None):
    L = len(graph)
    for i in range(L):
        N, edge, node_feature, edge_feature = graph[i]
        if labels is not None:
            y = np.reshape(labels[i],(1,len(labels[i])))
            y = torch.tensor(y, dtype=torch.float)
        else:
            y = None
        graph[i] = Data(
            idx=i,
            edge_index=torch.from_numpy(edge.T).int(),
            x=torch.from_numpy(node_feature).byte(),
            edge_attr=torch.from_numpy(edge_feature).byte(),
            y = y
        )    
    return graph


def main(argv):
    #dtypes = {'buildingblock1_smiles': np.int16, 'buildingblock2_smiles': np.int16, 'buildingblock3_smiles': np.int16,
    #        'binds_BRD4':np.byte, 'binds_HSA':np.byte, 'binds_sEH':np.byte}
      
    os.makedirs(FLAGS.output_folder, exist_ok=True)

    print('loading samples')    
    df_train = pd.read_csv(FLAGS.csv_file_path)#, dtype = dtypes)
    #df_train = df_train[:3000]
    print('Num training sampls: ',len(df_train))  
    print('----Featurizing training data -----')
    smiles_list = df_train['molecule_smiles'].tolist()
    if FLAGS.is_labeled:
        labels_list = df_train[['binds_BRD4','binds_HSA','binds_sEH']].values
    else:
        labels_list = None
    
    num_train= len(smiles_list)
    with Pool(processes=64) as pool:
        train_graph = list(tqdm(pool.imap(smile_to_graph, smiles_list), total=num_train))
    
    train_graph = to_pyg_list(train_graph,labels=labels_list)    
    if FLAGS.output_file_name is not None:
        print('----saving Featurized data -----')
        torch.save(train_graph,os.path.join(FLAGS.output_folder,FLAGS.output_file_name))

    
if __name__ == "__main__":
    app.run(main)