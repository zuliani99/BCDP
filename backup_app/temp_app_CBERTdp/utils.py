
from torch.nn.utils.rnn import pad_sequence

import torch
from datasets import load_dataset

import numpy as np

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
        
        
        
def read_embbedings(dataset_name, methods_name):

	path = f'app/embeddings/{dataset_name}'
	
	x_train = np.concatenate([np.load(f'{path}/{methods_name}/train_embeddings.npy'), np.load(f'{path}/{methods_name}/val_embeddings.npy')], 0, dtype=np.float32)
	x_test = np.load(f'{path}/{methods_name}/test_embeddings.npy')
        
	y_train = np.concatenate([np.load(f'{path}/train_labels.npy'), np.load(f'{path}/val_labels.npy')], 0)
	y_test = np.load(f'{path}/test_labels.npy')
        
	return x_train, x_test, y_train, y_test



def accuracy_result(model_results, ground_truth):
    result_list = 0
    for i in range(ground_truth.shape[0]):
        if model_results[i] == ground_truth[i]:
            result_list += 1
    return result_list/ground_truth.shape[0]