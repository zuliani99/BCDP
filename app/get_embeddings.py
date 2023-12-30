from utils import to_npy
import torch
from tqdm import tqdm
import numpy as np
import os

class GetEmbeddings():
	def __init__(self, name, embedding_split_perc, device, tokenizer, model, embedding_dim = None):
		self.name = name
		self.embedding_split_perc = embedding_split_perc
		self.device = device
		self.tokenizer = tokenizer
		self.model = model
		self.embedding_dim = embedding_dim
	
	
	def get_embeddings(self, ds_name, dataset):
		if self.embedding_dim is None: return

		for ds_type in ['train', 'test']:
      			
			path = f'app/embeddings/{ds_name}/{self.name}'#/{ds_type}'
         
			if not os.path.exists(f'app/embeddings/{ds_name}/{ds_type}_labels.npy'):
				np.save(f'app/embeddings/{ds_name}/{ds_type}_labels.npy', np.array([-1 if x == 0 else x for x in dataset[ds_type]['label']]))
			

			print(f'------------ Obtaning the embeddings for {ds_name} - {ds_type} ------------')

			len_ds = len(dataset[ds_type])
			dim_split = int(len_ds * self.embedding_split_perc)

			range_splits = [(idx, len_ds) if idx + dim_split > len_ds else (idx, idx + dim_split) for idx in range(0, len_ds, dim_split)]

			for idx, (strat_range, end_range) in enumerate(range_splits):
				
				embeddings_tensor = torch.empty((0, self.embedding_dim)).to(self.device)

				torch.save(embeddings_tensor, f'{path}/{ds_type}_{idx}.pt')

				ds_dict = dict(dataset[ds_type][strat_range:end_range].items())

				if 'text' in ds_dict: ds_dict = ds_dict['text']
				elif 'sentence' in ds_dict: ds_dict = ds_dict['sentence']
				else: raise Exception('Invalid key for datasets')
    

				for text in tqdm(ds_dict, total = len(ds_dict), leave=False, desc=f'Working on split {idx}'):

					embeddings_tensor = torch.load(f'{path}/{ds_type}_{idx}.pt')
						
					encoded_text = self.tokenizer(text, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt', padding=True).to(self.device)

					if self.name == 'LayerAggregation':
						_, embeds = self.model(encoded_text)
					else:
						embeds = self.model(encoded_text)
			
					embeddings_tensor = torch.cat((embeddings_tensor, embeds), dim=0)

					torch.save(embeddings_tensor,  f'{path}/{ds_type}_{idx}.pt')


		to_npy(self.datasets_dict, self.embedding_split_perc, self.name)
