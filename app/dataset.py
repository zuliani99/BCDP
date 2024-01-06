
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        
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