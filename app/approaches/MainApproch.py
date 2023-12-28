
from utils import get_embeddings
import torch.nn as nn

class Bert(nn.Module):
	def __init__(self, bert):
		super(Bert, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		return self.pre_trained_bert(**x, output_hidden_states=True).last_hidden_state[:, 0, :]
		
	

class MainApproch():
	def __init__(self, device, datasets_dict, model, tokenizer, embedding_split_perc):
		self.device = device
		self.datasets_dict = datasets_dict
		self.model = Bert(model).to(device)
		self.tokenizer = tokenizer
		self.embedding_split_perc = embedding_split_perc
		self.embedding_dim = 768

  
	def run(self):
		get_embeddings(self)
		# run clusering