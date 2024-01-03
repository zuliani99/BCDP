
from dataset import CustomTextDataset

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

import torch
from datasets import load_dataset

def get_datasets():
	return {
	 	'imdb': load_dataset('imdb'), 
		#'sst2': load_dataset('sst2'),
		#'y_p': load_dataset('yelp_polarity')
	}
 
 
   
   
def accuracy_score(output, label):
	output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
	return (output_class == label).sum().item()/len(output)



def collate_fn(batch):
	input_ids = pad_sequence([item[0]['input_ids'] for item in batch], batch_first=True, padding_value=0)
	attention_mask = pad_sequence([item[0]['attention_mask'] for item in batch], batch_first=True, padding_value=0)
	labels = torch.tensor([item[1] for item in batch])

	return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels


def get_dataloaders(datasets_dict, tokenizer, batch_size):
	
	datalaoders = {}
	
	for ds_name, dataset in datasets_dict.items():

		datalaoders[ds_name] = {}
			
		train_ds = CustomTextDataset(dataset['train'], tokenizer)
		test_ds = CustomTextDataset(dataset['test'], tokenizer)
   
		train_size = len(train_ds)

		val_size = int(train_size * 0.2)
		train_size -= val_size

		train_data, val_data = random_split(train_ds, [int(train_size), int(val_size)])
	
		datalaoders[ds_name]['train_dl'] = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
		datalaoders[ds_name]['val_dl'] = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
   
		datalaoders[ds_name]['test_dl'] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
		
	
	return datalaoders

def write_csv():
	pass