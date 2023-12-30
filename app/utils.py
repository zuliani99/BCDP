
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import torch
import numpy as np
from datasets import load_dataset
import os

def get_datasets():
	return {
     	'imdb': load_dataset('imdb'), 
        'sst2': load_dataset('sst2'),
        'y_p': load_dataset('yelp_polarity')
    }
 
def to_npy(datasets_dict, splits_perc, strategy_name):
  
	for ds_name in list(datasets_dict.keys()):
		for ds_type in ['train', 'test']:
			to_array = np.empty((0,768))
			for idx in range(int(splits_perc * 100)):
				to_array = np.vstack((to_array,
									torch.load(f'app/embeddings/{ds_name}/{strategy_name}/{ds_type}_{idx}.pt').cpu().detach().numpy()))
				
    			# delete .tensor file
				os.remove(f'app/embeddings/{ds_name}/{strategy_name}/{ds_type}_{idx}.pt') 

			np.save(f'app/embeddings/{ds_name}/{strategy_name}/{ds_type}.npy', to_array)
   
def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)



def collate_fn(batch):
    input_ids = pad_sequence([item[0]['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item[0]['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item[1] for item in batch])

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels



def write_csv():
    pass