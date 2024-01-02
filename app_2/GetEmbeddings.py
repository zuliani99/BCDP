
#from utils import to_npy
import torch
from tqdm import tqdm
import numpy as np
#import os

class GetEmbeddings():
	def __init__(self, name, embedding_split_perc, device, tokenizer, model, batch_size, embeddings_dim = None):
		self.name = name
		self.embedding_split_perc = embedding_split_perc
		self.device = device
		self.tokenizer = tokenizer
		self.model = model
		self.batch_size = batch_size
		self.embeddings_dim = embeddings_dim
	
	
	'''def get_embeddings(self, ds_name, dataset):
		if self.embeddings_dim is None: return

		for ds_type in ['train', 'test']:
      			
			path = f'app/embeddings/{ds_name}/{self.name}'#/{ds_type}'
         
			if not os.path.exists(f'app/embeddings/{ds_name}/{ds_type}_labels.npy'):
				np.save(f'app/embeddings/{ds_name}/{ds_type}_labels.npy', np.array([-1 if x == 0 else x for x in dataset[ds_type]['label']]))
			

			print(f'------------ Obtaning the embeddings for {ds_name} - {ds_type} ------------')

			len_ds = len(dataset[ds_type])
			dim_split = int(len_ds * self.embedding_split_perc)

			range_splits = [(idx, len_ds) if idx + dim_split > len_ds else (idx, idx + dim_split) for idx in range(0, len_ds, dim_split)]

			for idx, (strat_range, end_range) in enumerate(range_splits):
				
				embeddings_tensor = torch.empty((0, self.embeddings_dim)).to(self.device)

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


		to_npy(ds_name, self.embedding_split_perc, self.name)'''
  
  

	def get_embeddings(self, ds_name, dataloaders):
		if self.embeddings_dim is None: return

		for dl_name, dataloader in dataloaders.items():

			pbar = tqdm(dataloader, total = len(dataloader), leave=False, desc=f'Obtaining embedding for {ds_name} - {dl_name}')


			labels_npy = np.empty(0)
			embeddings_tensor = torch.empty((0, self.embeddings_dim)).to(self.device)

   
			with torch.inference_mode(): # Allow inference mode
				for dictionary, labels in pbar:
        
					for key in list(dictionary.keys()):
						dictionary[key] = dictionary[key].to(self.device)

					labels_npy = np.vstack((labels_npy, np.array([
         						-1 if x == 0 else x for x in torch.squeeze(labels).numpy()])))
        						
					if self.name == 'LayerAggregation':
						_, embeds = self.model(dictionary)
					else:
						embeds = self.model(dictionary)

					embeddings_tensor = torch.cat((embeddings_tensor, embeds), dim=0)
     
				np.save(f'app/embeddings/{ds_name}/{dl_name}_labels.npy', labels_npy)
				np.save(f'app/embeddings/{ds_name}/{self.name}/embeddings_{dl_name}.npy',
					embeddings_tensor.cpu().detach().numpy()
            	)
