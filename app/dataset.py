from torch.data.utils import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):   
        return self.tokenizer(
            self.vocab[idx][0], return_tensors='pt', truncation=True, padding=True
            ), self.vocab[idx][1]