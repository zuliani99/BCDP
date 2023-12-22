
import torch
import numpy as np
from datasets import load_dataset
import os

def get_datasets():
	return {
     	'imdb': load_dataset('imdb'), 
        'sst2': load_dataset('sst2'),
        'r_t': load_dataset('rotten_tomatoes')
    }
 
def to_npy(datasets_dict, splits_perc, strategy_name):
  
	for ds_name in list(datasets_dict.keys()):
		for ds_type in ['train', 'test']:
			to_array = np.empty((0,768))
			for idx in range(int(splits_perc * 100)):
				to_array = np.vstack((to_array,
									torch.load(f'embeddings/{ds_name}/{strategy_name}/{ds_type}_{idx}.pt').cpu().detach().numpy()))
				
    			# delete .tensor file
				os.remove(f'embeddings/{ds_name}/{strategy_name}/{ds_type}_{idx}.pt') 

			np.save(f'embeddings/{ds_name}/{strategy_name}/{ds_type}.npy', to_array)