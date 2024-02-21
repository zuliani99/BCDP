
from torch.utils.data import Dataset, DataLoader, random_split

from utils import collate_fn
import numpy as np

# Custom text dataset
class CustomTextDataset(Dataset):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.indices = np.arange(self.vocab.num_rows)

        
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, idx): 
        
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
    
    
    
# Get the dictionary containing the datloaders from the custom text datssets
def get_dsname_dataloaders(datasets_dict, tokenizer, batch_size):
	
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
                                            shuffle=True,
                                            pin_memory=True)
  
		datalaoders[ds_name]['val'] = DataLoader(val_data,
                							batch_size=batch_size,
                             				collate_fn=collate_fn,
                                            shuffle=False,
                                 			pin_memory=True)
   
		datalaoders[ds_name]['test'] = DataLoader(test_ds,
                                            batch_size=batch_size,
                                            collate_fn=collate_fn,
                                            shuffle=False,
                                            pin_memory=True)
		
	return datalaoders