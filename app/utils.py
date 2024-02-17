
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn.init as init
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader


import csv
import os
import errno



def get_datasets():
	return {
	 	'imdb': load_dataset('imdb'), 
		'sst2': load_dataset('sst2'),
		'y_p': load_dataset('yelp_polarity')
	}
 
 
   
def accuracy_score(output, label):
	output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
	return (output_class == label).sum().item()/len(output)



def collate_fn(batch):
	input_ids = pad_sequence([item[0]['input_ids'] for item in batch], batch_first=True, padding_value=0)
	attention_mask = pad_sequence([item[0]['attention_mask'] for item in batch], batch_first=True, padding_value=0)
	labels = torch.tensor([item[1] for item in batch])

	return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels



def write_csv(ts_dir, head, values, categoty_type):
	if (not os.path.exists(f'app/results/{ts_dir}/{categoty_type}_results.csv')):
		
		with open(f'app/results/{ts_dir}/{categoty_type}_results.csv', 'w', encoding='UTF8') as f:
			writer = csv.writer(f)
			writer.writerow(head)
			f.close()
	
	with open(f'app/results/{ts_dir}/{categoty_type}_results.csv', 'a', encoding='UTF8') as f:
		writer = csv.writer(f)
		writer.writerow(values)
		f.close()
		
		
  
def create_ts_dir_res(timestamp):
	
	mydir = os.path.join('app/results', timestamp) #results
	try:
		os.makedirs(mydir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise  # This was not a "directory exist" error..
		


def init_params(m):
	if isinstance(m, nn.Linear):
		init.normal_(m.weight, std=1e-3)
		if m.bias is not None: init.constant_(m.bias, 0)        



def read_embbedings_pt(dataset_name, choosen_model_embedding, bool_numpy = False, bool_validation = False):

	path = f'app/embeddings/{choosen_model_embedding}/{dataset_name}'
	if not bool_validation:
		x_train = torch.cat((torch.load(f'{path}/train_embeddings.pt'), torch.load(f'{path}/val_embeddings.pt')), dim=0, dtype=torch.float32)
		x_test = torch.load(f'{path}/test_embeddings.pt')
			
		y_train = torch.cat((torch.load(f'{path}/labels/train_labels.pt'), torch.load(f'{path}/labels/val_labels.pt')), dim=0)
		y_test = torch.load(f'{path}/labels/test_labels.pt')
	
		if not bool_numpy: return x_train, x_test, y_train, y_test
		else: return x_train.numpy(), x_test.numpy(), y_train.numpy(), y_test.numpy()
	else:
		x_train = torch.load(f'{path}/train_embeddings.pt')
		x_val = torch.load(f'{path}/val_embeddings.pt')
		x_test = torch.load(f'{path}/test_embeddings.pt')
			
		y_train = torch.load(f'{path}/labels/train_labels.pt')
		y_val = torch.load(f'{path}/labels/val_labels.pt')
		y_test = torch.load(f'{path}/labels/test_labels.pt')
		
		return x_train, x_val, x_test, y_train, y_val, y_test



def get_text_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test):
	train_dl = DataLoader(TensorDataset(x_train, y_train))
	val_dl = DataLoader(TensorDataset(x_val, y_val))
	test_dl = DataLoader(TensorDataset(x_test, y_test))
	return train_dl, val_dl, test_dl



def accuracy_result(model_results, ground_truth):
	result_list = 0
	for i in range(ground_truth.shape[0]):
		if model_results[i] == ground_truth[i]:
			result_list += 1
	return result_list/ground_truth.shape[0]



def get_competitors_embeddings_dls(ds_name, choosen_model_embedding):
	x_train, x_val, x_test, y_train, y_val, y_test = read_embbedings_pt(ds_name, choosen_model_embedding, bool_validation=True)
   
	x_train = x_train[-1][:, 0, :]
	x_val = x_val[-1][:, 0, :]
	x_test = x_test[-1][:, 0, :]
   
	#train_dl, val_dl, test_dl = get_competitors_embeddings_dls(ds_name)
	return get_text_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test)
 