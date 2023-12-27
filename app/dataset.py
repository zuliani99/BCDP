from torch.data.utils import Dataset

class WikiTextDataset(Dataset):
    def __init__(self, file_path, vocab, tokenizer):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
    
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = self.tokenizer(f.read())
        return tokens
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        pass