
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
									torch.load(f'embeddings/{ds_name}/{strategy_name}/{ds_type}_{idx}.pt').cpu().detach().numpy()))
				
    			# delete .tensor file
				os.remove(f'embeddings/{ds_name}/{strategy_name}/{ds_type}_{idx}.pt') 

			np.save(f'embeddings/{ds_name}/{strategy_name}/{ds_type}.npy', to_array)
   
def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)



def get_embeddings(method):
	
	for ds_name, dataset in method.datasets_dict.items():

		for ds_type in ['train', 'test']:
       
			path = f'app/embeddings/{ds_name}/{method.__class__.__name__}/{ds_type}'

			print(f'------------ Obtaning the embeddings for {ds_name} - {ds_type} ------------')

			len_ds = len(dataset[ds_type])
			dim_split = int(len_ds * method.embedding_split_perc)

			range_splits = [(idx, len_ds) if idx + dim_split > len_ds else (idx, idx + dim_split) for idx in range(0, len_ds, dim_split)]

			for idx, (strat_range, end_range) in enumerate(range_splits):

				embeddings_tensor = torch.empty((0, method.embedding_dim)).to(method.device)

				torch.save(embeddings_tensor, f'{path}_{idx}.pt')

				ds_dict = dict(dataset[ds_type][strat_range:end_range].items())

				if 'text' in ds_dict: ds_dict = ds_dict['text']
				elif 'sentence' in ds_dict: ds_dict = ds_dict['sentence']
				else: raise Exception('Invalid key for datasets')


				for text in tqdm(ds_dict, total = len(ds_dict), leave=False, desc=f'Working on split {idx}'):

					embeddings_tensor = torch.load(f'{path}_{idx}.pt')
					encoded_text = method.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(method.device)
					
					if method.__class__.__name__ == 'LayerAggregation':
						_, embeds = method.model(encoded_text)
					else:
						embeds = method.model(encoded_text)
         
					embeddings_tensor = torch.cat((embeddings_tensor, embeds), dim=0)

					torch.save(embeddings_tensor, f'{path}_{idx}.pt')


		to_npy(method.datasets_dict, method.embedding_split_perc, method.__class__.__name__)



def collate_fn(batch):
    #print(batch)
    input_ids = pad_sequence([item[0]['input_ids'] for item in batch], batch_first=True, padding_value=0)
    token_type_ids = pad_sequence([item[0]['token_type_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item[0]['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item[1] for item in batch])

    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}, labels
