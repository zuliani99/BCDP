
from utils import to_npy
import torch

from tqdm import tqdm


class MainApproch():
	def __init__(self, device, datasets_dict, model, tokenizer, embedding_split_perc):
		self.device = device
		self.datasets_dict = datasets_dict
		self.model = model
		self.tokenizer = tokenizer
		self.embedding_split_perc = embedding_split_perc



	def get_embeddings_main_approach(self):
	
		for ds_name, dataset in self.datasets_dict.items():

			for ds_type in ['train', 'test']:
       
				path = f'embeddings/{ds_name}/{self.__class__.__name__}/{ds_type}'

				print(f'------------ Obtaning the embeddings for {ds_name} - {ds_type} ------------')

				len_ds = len(dataset[ds_type])
				dim_split = int(len_ds * self.embedding_split_perc)

				range_splits = [(idx, len_ds) if idx + dim_split > len_ds else (idx, idx + dim_split) for idx in range(0, len_ds, dim_split)]

				for idx, (strat_range, end_range) in enumerate(range_splits):

					embeddings_tensor = torch.empty((0,768)).to(self.device)
					outputs = torch.empty((0,768)).to(self.device)

					torch.save(embeddings_tensor, f'{path}_{idx}.pt')

					ds_dict = dict(dataset[ds_type][strat_range:end_range].items())

					if 'text' in ds_dict: ds_dict = ds_dict['text']
					elif 'sentence' in ds_dict: ds_dict = ds_dict['sentence']
					else: raise Exception('Invalid key for datasets')


					for text in tqdm(ds_dict, total = len(ds_dict), leave=False, desc=f'Working on split {idx}'):

						embeddings_tensor = torch.load(f'{path}_{idx}.pt')
						encoded_text = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
						outputs = self.model(**encoded_text, output_hidden_states=True).last_hidden_state[:, 0, :]
						embeddings_tensor = torch.cat((embeddings_tensor, outputs), dim=0)

						torch.save(embeddings_tensor, f'{path}_{idx}.pt')


		to_npy(self.datasets_dict, self.embedding_split_perc, self.__class__.__name__)
  
  
	def run(self):
		self.get_embeddings_main_approach()