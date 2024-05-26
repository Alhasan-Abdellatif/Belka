import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from sklearn.metrics import average_precision_score, roc_auc_score
from absl import app, flags
import time
import datetime
from torch_scatter import scatter
from multiprocessing import Pool
from featurizing_data import *

# Atom Featurisation
## Auxiliary function for one-hot enconding transformation based on list of
##permitted values

FLAGS = flags.FLAGS

flags.DEFINE_string(name="test_csv_for_ids_path", default="../data/test.csv", help="")
flags.DEFINE_string(name="test_csv_path", default="../shrunken_data/test.csv", help="")
flags.DEFINE_string(name="test_file_path", default="../gnn_data/test_byte.pt", help="")


flags.DEFINE_string(name="train_file_path", default="../shrunken_data/train.csv", help="csv or pt")
flags.DEFINE_string(name="valid_file_path", default="../gnn_data/valid.pt", help="")

flags.DEFINE_string(name="results_dir", default="results", help="")


flags.DEFINE_integer(name="featurizing_batch_size", default=256, help="")
flags.DEFINE_integer(name="shard_size", default=1000000, help="")
flags.DEFINE_integer(name="dev_num", default=0, help="")


flags.DEFINE_integer('emb_dim', 96, 'Dimension of the embeddings')
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs for training')
flags.DEFINE_integer('num_layers', 4, 'Number of layers in the model')
flags.DEFINE_float('dropout_rate', 0.3, 'Dropout rate for regularization')
flags.DEFINE_float('lr', 0.001, 'Learning rate for training')
flags.DEFINE_integer('out_channels', 3, 'Number of output channels')
flags.DEFINE_integer('batch_size', 32, 'batch_size')

flags.DEFINE_boolean(name="online_featurizing",default=False,help="")

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


#MODEL: simple MPNNModel
#from https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

class MPNNLayer(MessagePassing):
	def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
		super().__init__(aggr=aggr)

		self.emb_dim = emb_dim
		self.edge_dim = edge_dim
		self.mlp_msg = nn.Sequential(
			nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
			nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
		)
		self.mlp_upd = nn.Sequential(
			nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
			nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
		)

	def forward(self, h, edge_index, edge_attr):
		out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
		return out

	def message(self, h_i, h_j, edge_attr):
		msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
		return self.mlp_msg(msg)

	def aggregate(self, inputs, index):
		return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

	def update(self, aggr_out, h):
		upd_out = torch.cat([h, aggr_out], dim=-1)
		return self.mlp_upd(upd_out)

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNNModel(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        super().__init__()

        self.lin_in = nn.Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        self.pool = global_mean_pool

    def forward(self, data): #PyG.Data - batch of PyG graphs
        h = self.lin_in(F_unpackbits(data.x,-1).float()[:,:36])  

        for conv in self.convs:
            h = h + conv(h, data.edge_index.long(), F_unpackbits(data.edge_attr,-1).float()[:,:6])  # (n, d) -> (n, d)

        h_graph = self.pool(h, data.batch)  
        return h_graph



PACK_NODE_DIM=5
PACK_EDGE_DIM=1
NODE_DIM=PACK_NODE_DIM*8-4
EDGE_DIM=PACK_EDGE_DIM*8-2

class GNNModel(nn.Module):
	def __init__(self, in_dim=NODE_DIM, edge_dim=EDGE_DIM, emb_dim=96, num_layers=4,out_channels=3,dropout = 0.1):
		super().__init__()

		self.output_type = ['infer', 'loss']

		self.smile_encoder = MPNNModel(
			 in_dim=in_dim, edge_dim=edge_dim, emb_dim=emb_dim, num_layers=num_layers,
		)
		self.bind = nn.Sequential(
			nn.Linear(emb_dim, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(1024, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(1024, 512),
			#nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(512, out_channels),
		)

	def forward(self, batch):
		graph = batch['graph']
		x = self.smile_encoder(graph) 
		bind = self.bind(x)
		# --------------------------
		output = {}
		if 'loss' in self.output_type:
			target = batch['bind']
			output['bce_loss'] = F.binary_cross_entropy_with_logits(bind.float(), target.float())
		if 'infer' in self.output_type:
			output['bind'] = torch.sigmoid(bind)

		return output

def my_collate(graph, index=None, device='cpu'):
    if index is None:
        index = np.arange(len(graph)).tolist()
    batch = dotdict(
        x=[],
        edge_index=[],
        edge_attr=[],
        batch=[],
        idx=index
    )
    offset = 0
    for b, i in enumerate(index):
        N, edge, node_feature, edge_feature = graph[i]
        batch.x.append(node_feature)
        batch.edge_attr.append(edge_feature)
        batch.edge_index.append(edge.astype(int) + offset)
        batch.batch += N * [b]
        offset += N
    batch.x = torch.from_numpy(np.concatenate(batch.x)).to(device)
    batch.edge_attr = torch.from_numpy(np.concatenate(batch.edge_attr)).to(device)
    batch.edge_index = torch.from_numpy(np.concatenate(batch.edge_index).T).to(device)
    batch.batch = torch.LongTensor(batch.batch).to(device)
    return batch

def train_model(training_set, num_epochs,model, lr,results_dir,batch_size=32,device = 'cuda:1',es_patience = 8,train_frac = 0.9):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_time = time.time()

    
    best_val = None
    patience = es_patience 
    val_ap_list = []
    binds_BRD4_AP_list = []
    binds_HSA_AP_list = []
    binds_sEH_AP_list = []
    
    SAMPLE = 75000000
    training_set = training_set[:SAMPLE]
    smiles_list_train = training_set['molecule_smiles'].tolist()
    labels_list = training_set[['binds_BRD4','binds_HSA','binds_sEH']].values
   
   
    print('----Featurizing training data -----')
    with Pool(processes=64) as pool:
        train_data = list(pool.imap(smile_to_graph, smiles_list_train))
    
    
    # random splitting
    num_train = int(train_frac * len(training_set))
    rng = np.random.RandomState(123)
    index = np.arange(len(training_set))
    rng.shuffle(index)
    train_idx = index[:num_train]
    val_idx = index[num_train:]

    #train_idx = np.array(training_set.index)[:95000]
    #val_idx = np.array(training_set.index)[:5000]


    # iteration over the training samples
    for epoch in range(num_epochs):
        np.random.shuffle(train_idx)

        model.train()
        total_loss = 0
        #shard_ind = 0
        num_mini_batches = 0
  
        for index in np.arange(0,len(train_idx),batch_size):
            index_batch = train_idx[index:index+batch_size]
            num_mini_batches +=1
            batch = dotdict(
                graph = my_collate(train_data,index_batch,device=device),
                bind = torch.from_numpy(labels_list[index_batch]).float().to(device),
            )
            #batch = batch.to(device)
            model.output_type = ['loss', 'infer']
            
            with torch.cuda.amp.autocast(enabled=True):
                output = model(batch)  #data_parallel(net,batch) #
                bce_loss = output['bce_loss']

            optimizer.zero_grad() 
            scaler.scale(bce_loss).backward() 
            scaler.step(optimizer)
            scaler.update()
            
            torch.clear_autocast_cache()
            total_loss += bce_loss.item()
 
 
        # validation
        print('Validation')
        val_predictions= predict_with_model(model, train_data,device,val_idx,1024)
        labels = torch.tensor(labels_list[val_idx]).view(-1).tolist()
        binds_BRD4_AP  = average_precision_score(y_true=labels[0::3], y_score=val_predictions[0::3],average='macro')
        binds_HSA_AP  = average_precision_score(y_true=labels[1::3], y_score=val_predictions[1::3],average='macro')
        binds_sEH_AP = average_precision_score(y_true=labels[2::3], y_score=val_predictions[2::3],average='macro')
        
        # During the first iteration (first epoch) best validation is set to None
        binds_BRD4_AP_list.append(binds_BRD4_AP)
        binds_HSA_AP_list.append(binds_HSA_AP)
        binds_sEH_AP_list.append(binds_sEH_AP)
        
        val_ap = np.mean([binds_BRD4_AP,binds_HSA_AP,binds_sEH_AP])
        val_ap_list.append(val_ap)

        #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / num_mini_batches}')
        print('Epoch {:03}: | Loss: {:.5f} | binds_BRD4_AP: {:.5f} | binds_HSA_AP: {:.5f} | binds_sEH_AP: {:.5f} | mAP: {:.5f}  | Training time: {}'.format(
            epoch + 1, 
            total_loss/num_mini_batches, 
            binds_BRD4_AP, 
            binds_HSA_AP,
            binds_sEH_AP,
            val_ap,
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
                print('Early stopping. Best Val aP: {:.4f}'.format(best_val))
                break  
        
        torch.save(model.state_dict(), os.path.join(results_dir,'Epoch_'+str(epoch+1)+'.pth'))  # Saving the model


      
    df = pd.DataFrame(val_ap_list)
    df.to_csv(os.path.join(results_dir,'val_ap_list.csv'), index=False)
    df = pd.DataFrame(binds_BRD4_AP_list)
    df.to_csv(os.path.join(results_dir,'binds_BRD4_AP_list.csv'), index=False)
    df = pd.DataFrame(binds_HSA_AP_list)
    df.to_csv(os.path.join(results_dir,'binds_HSA_AP_list.csv'), index=False)
    df = pd.DataFrame(binds_sEH_AP_list)
    df.to_csv(os.path.join(results_dir,'binds_sEH_AP_list.csv'), index=False)

    #torch.save(val_ap_list, os.path.join(results_dir,'val_ap_list.pt'))  
    #return model

def predict_with_model(model, data,device,idx,batch_size = 32):
    model.eval()
    model.output_type = ['infer']
    predictions = []

    with torch.no_grad():
        for t, index in enumerate(np.arange(0,len(idx),batch_size)):
            index_batch = idx[index:index+batch_size]
            batch = dotdict(
                graph = my_collate(data,index_batch,device=device),
            )
            output = model(batch) 
            predictions.extend(output['bind'].view(-1).tolist())

    return predictions

def select_and_save_predictions_with_ids(predictions,test_df,path_test_file_for_ids):
    
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

    return y_pred_and_ids_df

def main(argv):
 
    # ---hyper-parameters
    input_dim = NODE_DIM
    edge_dim = EDGE_DIM
    
    device = "cuda:"+str(FLAGS.dev_num)
    # define model
    #model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate,out_channels).to(device)
    model = GNNModel(in_dim=input_dim, edge_dim=edge_dim, emb_dim=FLAGS.emb_dim, num_layers=FLAGS.num_layers,
                     out_channels = FLAGS.out_channels,dropout=FLAGS.dropout_rate).to(device)

    os.makedirs(FLAGS.results_dir, exist_ok=True)

    #print('loading validation samples')    
    #------ validation set loading
    #valid_data = torch.load(FLAGS.valid_file_path)
    #valid_loader = DataLoader(valid_data, batch_size=FLAGS.batch_size, shuffle=False)

    print('loading training samples')    
    #if FLAGS.online_featurizing:
    training_set = pd.read_csv(FLAGS.train_file_path)
    #else:
    #training_set = torch.load(FLAGS.train_file_path)


    # --- Training
    #model = train_model(train_loader,num_epochs, input_dim, hidden_dim,num_layers, dropout_rate,out_channels, lr)
    train_model(training_set=training_set, model=model, lr=FLAGS.lr,num_epochs=FLAGS.num_epochs
                        ,batch_size=FLAGS.batch_size,device = device,results_dir = FLAGS.results_dir)
    
    model.load_state_dict(torch.load(os.path.join(FLAGS.results_dir,'best_val.pth')))  # Loading best model of this fold
    
    # ----- Test set predictions ---
    if FLAGS.test_file_path is not None:
        print('Predicting on test samples')    
        print('loading test samples')    
        test_df = pd.read_csv(FLAGS.test_csv_path)

        smiles_list_test = test_df['molecule_smiles'].tolist()
        #print(smiles_list_train)
        print('----Featurizing testing data -----')
        with Pool(processes=64) as pool:
            test_data = list(pool.imap(smile_to_graph, smiles_list_test))
        
        # Predict
        #test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        test_idx = np.array(test_df.index)
        predictions = predict_with_model(model, test_data,device,test_idx)
        
        # save predictions
        y_pred_and_ids_df = select_and_save_predictions_with_ids(predictions,test_df,FLAGS.test_csv_for_ids_path)

        output_path = os.path.join(FLAGS.results_dir,'ids_pred_results.csv')
        y_pred_and_ids_df.to_csv(output_path,index=False)
        
if __name__ == "__main__":
    app.run(main)