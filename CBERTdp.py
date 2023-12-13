# -*- coding: utf-8 -*-

from transformers import BertTokenizer, BertModel
import torch
from datasets import load_dataset
from tqdm.notebook import tqdm
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_datasets():
	#imdb = load_dataset('imdb')
	#sst2 = load_dataset('sst2')
	yelp_review_full = load_dataset('yelp_review_full')
	#return {'imdb': imdb, 'sst2': sst2, 'yelp': yelp_review_full}
	return {'yelp': yelp_review_full}



def to_npy(datasets_dict):
	for ds_name in list(datasets_dict.keys()):
		to_array = np.empty((0,768))
		for idx in range(10):
			to_array = np.vstack((to_array,
								torch.load(f'{ds_name}/embeddings_{idx}.pt').numpy()))
		np.save(f'{ds_name}/embeddings.npy', to_array)


def main():
	datasets_dict = get_datasets()

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained("bert-base-uncased")

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained("bert-base-uncased")
	model = model.to(device)

	split = 0.1

 
	for ds_name, dataset in datasets_dict.items():

		print(f'------------ {ds_name} ------------')


		len_train = len(dataset['train'])
		dim_split = int(len_train * split)

		range_splits = [(idx, len_train) if idx + dim_split > len_train else (idx, idx + dim_split) for idx in range(0, len_train, dim_split)]

		for idx, (strat_range, end_range) in enumerate(range_splits):

			embeddings_tensor = torch.empty((0,768)).to(device)
			outputs = torch.empty((0,768)).to(device)

			torch.save(embeddings_tensor, f'{ds_name}/embeddings_{idx}.pt')


			print(f'---------- Split {idx} ----------')

			ds_dict = dict(dataset['train'][strat_range:end_range].items())

			if 'text' in ds_dict: ds_dict = ds_dict['text']
			elif 'sentence' in ds_dict: ds_dict = ds_dict['sentence']
			else: raise Exception('Invalid key for datasets')


			for text in tqdm(ds_dict, total = len(ds_dict), leave=False):

				embeddings_tensor = torch.load(f'{ds_name}/embeddings_{idx}.pt')

				encoded_text = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)

				outputs = model(**encoded_text, output_hidden_states=True).last_hidden_state[:, 0, :]

				embeddings_tensor = torch.cat((embeddings_tensor, outputs), dim=0)

				torch.save(embeddings_tensor, f'{ds_name}/embeddings_{idx}.pt')

	to_npy(datasets_dict)

if __name__ == "__main__":
  main()