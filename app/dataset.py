import torch
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, idx): 
        encoding = self.tokenizer(
            self.vocab[idx]['text'], return_tensors='pt', truncation=True, padding=True
            )
        #input_ids = encoding['input_ids'].squeeze()
        #attention_mask = encoding['attention_mask'].squeeze()

        #return #{'input_ids': input_ids, 'attention_mask': attention_mask, 'label': self.vocab[idx]['label']}
        return encoding, self.vocab[idx]['label']