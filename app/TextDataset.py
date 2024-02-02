
from torch.utils.data import Dataset, Sampler, DataLoader, random_split
import torch

from utils import collate_fn
import numpy as np
import os

class CustomTextDataset(Dataset):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.indices = np.arange(self.vocab.num_rows)

        
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, idx): 
        """ retrieve a specific item of the dataset

        @param idx: int, the index of the item to be retrieved

        @Return:
            - a dictionary containing the input-ids and the attention masks
            - the label associated with the item
        """
        
        encoding = self.tokenizer.encode_plus(
            self.vocab[idx]['text'] if 'text' in self.vocab[idx] else self.vocab[idx]['sentence'],

            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt', padding=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask}, self.vocab[idx]['label']
    
    
class UniqueShuffle(Sampler):
    def __init__(self, dataset, ds_name, dl_type):
        self.indices = dataset.indices #if isinstance(dataset, Subset) else list(dataset.vocab.keys())
        self.dl_order_path = f'app/embeddings/{ds_name}/dataloaders_order/{dl_type}.npy'
        self.shuffle_indices()

    def shuffle_indices(self):
        if not os.path.exists(self.dl_order_path):
            print('Creating and Saving Dataloader Order')
            self.indices = list(torch.randperm(len(self.indices)))
            np.save(self.dl_order_path, np.array(self.indices, dtype=np.int8))
        else:
            print('Loading Dataloader Order') 
            self.indices = np.load(self.dl_order_path).tolist()
        
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
    
def get_dataloaders(datasets_dict, tokenizer, batch_size):
	
	datalaoders = {}
	
	for ds_name, dataset in datasets_dict.items():

		datalaoders[ds_name] = {}
			
		train_ds = CustomTextDataset(dataset['train'], tokenizer)
		if ds_name == 'sst2': test_ds = CustomTextDataset(dataset['validation'], tokenizer)
		else: test_ds = CustomTextDataset(dataset['test'], tokenizer)
   
		train_size = len(train_ds)

		val_size = int(train_size * 0.2)
		train_size -= val_size

		train_data, val_data = random_split(train_ds, [int(train_size), int(val_size)])
	
		datalaoders[ds_name]['train'] = DataLoader(train_data,
                                            batch_size=batch_size,
                                            collate_fn=collate_fn,
                                            sampler=UniqueShuffle(train_data, ds_name, 'train'),
                                            pin_memory=True)
  
		datalaoders[ds_name]['val'] = DataLoader(val_data,
                							batch_size=batch_size,
                             				collate_fn=collate_fn,
                                			sampler=UniqueShuffle(val_data, ds_name, 'val'),
                                 			pin_memory=True)
   
		datalaoders[ds_name]['test'] = DataLoader(test_ds,
                                            batch_size=batch_size,
                                            collate_fn=collate_fn,
                                            sampler=UniqueShuffle(test_ds, ds_name, 'test'),
                                            pin_memory=True)
		
	return datalaoders