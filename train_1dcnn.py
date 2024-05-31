import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
#from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset, Dataset, DataLoader

from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from sklearn.metrics import average_precision_score, roc_auc_score
from absl import app, flags
import time
import datetime
#from torch_scatter import scatter
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

MAX_LEN = 142
MAX_LEN = 130
FEATURES = [f'enc{i}' for i in range(MAX_LEN)]
TARGETS = ['bind1', 'bind2', 'bind3']
#TARGETS = ['binds_BRD4','binds_HSA','binds_sEH']


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



class oneDCNN(nn.Module):
    def __init__(self,num_embeddings=37, num_filters=32, hidden_dim=128):
        super(oneDCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=num_filters, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=3, padding=0, stride=1)
        self.conv3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*3, kernel_size=3, padding=0, stride=1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters*3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #print(x.shape)
        #print(torch.max(x))
        #print(self.embedding(x).shape)
        x = self.embedding(x).transpose(1, 2)  # Transpose for Conv1d
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        #x = torch.sigmoid(self.output(x))
        x = self.output(x)
        return x


def train_model(training_set, num_epochs,model, lr,results_dir,batch_size=32,device = 'cuda:1',es_patience = 8,train_frac = 0.9):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4,verbose=True)

    start_time = time.time()

    
    best_val = None
    patience = es_patience 
    val_ap_list = []
    binds_BRD4_AP_list = []
    binds_HSA_AP_list = []
    binds_sEH_AP_list = []
    
    SAMPLE = len(training_set)
    training_set = training_set[:SAMPLE]
    #smiles_list_train = training_set['molecule_smiles'].tolist()
    #labels_list = training_set[['binds_BRD4','binds_HSA','binds_sEH']].values
    

   
   
    #print('----Featurizing training data -----')
    #with Pool(processes=64) as pool:
    #    train_data = list(pool.imap(smile_to_graph, smiles_list_train))
    
    
    # random splitting
    num_train = int(train_frac * len(training_set))
    rng = np.random.RandomState(123)
    index = np.arange(len(training_set))
    rng.shuffle(index)
    train_idx = index[:num_train]
    val_idx = index[num_train:]

    #train_idx = np.array(training_set.index)[:95000]
    #val_idx = np.array(training_set.index)[:5000]
    
     # Convert pandas dataframes to PyTorch tensors
    

    X_train = torch.tensor(training_set.loc[train_idx, FEATURES].values, dtype=torch.int)
    y_train = torch.tensor(training_set.loc[train_idx, TARGETS].values, dtype=torch.float16)
    X_val = torch.tensor(training_set.loc[val_idx, FEATURES].values, dtype=torch.int)
    y_val = torch.tensor(training_set.loc[val_idx, TARGETS].values, dtype=torch.float16)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_val, y_val)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        
    
    

    print('----Starting training -----')

    # iteration over the training samples
    for epoch in range(num_epochs):
        np.random.shuffle(train_idx)

        model.train()
        total_loss = 0
        #shard_ind = 0
        num_mini_batches = 0
  
        #for index in np.arange(0,len(train_idx),batch_size):
        for inputs, targets in train_loader: 
            #index_batch = train_idx[index:index+batch_size]
            num_mini_batches +=1
            #batch = dotdict(
            #    graph = my_collate(train_data,index_batch,device=device),
            #    bind = torch.from_numpy(labels_list[index_batch]).float().to(device),
            #)
            #batch = batch.to(device)
            #model.output_type = ['loss', 'infer']
            optimizer.zero_grad() 

            with torch.cuda.amp.autocast(enabled=True):
                #print(inputs)
                output = model(inputs.to(device))  #data_parallel(net,batch) #
                bce_loss = F.binary_cross_entropy_with_logits(output, targets.to(device))

            scaler.scale(bce_loss).backward() 
            scaler.step(optimizer)
            scaler.update()
            
            torch.clear_autocast_cache()
            total_loss += bce_loss.item()
 
 
        # validation
        print('Validation')
        val_predictions= predict_with_model(model, valid_loader,device)
        val_predictions = np.array(val_predictions).reshape(-1).tolist()
        labels = y_val.reshape(-1).tolist()
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
        print('Epoch {:05}: | Loss: {:.7f} | binds_BRD4_AP: {:.5f} | binds_HSA_AP: {:.5f} | binds_sEH_AP: {:.5f} | mAP: {:.5f}  | Training time: {}'.format(
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
        scheduler.step(val_ap)


      
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

def predict_with_model(model, val_loader,device,):
    model.eval()
    model.output_type = ['infer']
    predictions = []

    with torch.no_grad():
        #for t, index in enumerate(np.arange(0,len(idx),batch_size)):
        for inputs in val_loader:  # Assuming you have a DataLoader named val_loader
            output = model(inputs[0].to(device)) 
            predictions.extend(torch.sigmoid(output).tolist())

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
    #input_dim = NODE_DIM
    #edge_dim = EDGE_DIM
    
    device = "cuda:"+str(FLAGS.dev_num)
    # define model
    #model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate,out_channels).to(device)
    #model = GNNModel(in_dim=input_dim, edge_dim=edge_dim, emb_dim=FLAGS.emb_dim, num_layers=FLAGS.num_layers,
    #                 out_channels = FLAGS.out_channels,dropout=FLAGS.dropout_rate).to(device)
    model = oneDCNN(FLAGS.emb_dim).to(device)

    os.makedirs(FLAGS.results_dir, exist_ok=True)

    #print('loading validation samples')    
    #------ validation set loading
    #valid_data = torch.load(FLAGS.valid_file_path)
    #valid_loader = DataLoader(valid_data, batch_size=FLAGS.batch_size, shuffle=False)

    print('loading training samples')    
    #if FLAGS.online_featurizing:
    #training_set = pd.read_csv(FLAGS.train_file_path)
    training_set = pd.read_parquet(FLAGS.train_file_path)
    
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
        #test_df = pd.read_csv(FLAGS.test_csv_path)
         
        test_data = pd.read_parquet(FLAGS.test_file_path)
        test_idx = np.array(test_data.index)
        X_test = torch.tensor(test_data.loc[test_idx, FEATURES].values, dtype=torch.int)
        # Create TensorDatasets
        test_dataset = TensorDataset(X_test)
        tst = pd.read_csv(FLAGS.test_csv_for_ids_path,index_col=False)#[:len(y_pred_df)]
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        predictions = predict_with_model(model, test_loader,device)
        predictions = np.array(predictions)
        tst['binds'] = 0
        tst.loc[tst['protein_name']=='BRD4', 'binds'] = predictions[(tst['protein_name']=='BRD4').values, 0]
        tst.loc[tst['protein_name']=='HSA', 'binds'] = predictions[(tst['protein_name']=='HSA').values, 1]
        tst.loc[tst['protein_name']=='sEH', 'binds'] = predictions[(tst['protein_name']=='sEH').values, 2]
        output_path = os.path.join(FLAGS.results_dir,'ids_pred_results.csv')
        tst[['id', 'binds']].to_csv(output_path, index = False)

       
        
if __name__ == "__main__":
    app.run(main)