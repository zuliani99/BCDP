
from app.get_embeddings import GetEmbeddings
import torch.nn as nn

class BertLastLayer(nn.Module):
	def __init__(self, bert):
		super(BertLastLayer, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		return self.pre_trained_bert(**x, output_hidden_states=True).last_hidden_state[:, 0, :]
		
	

class MainApproch(GetEmbeddings):
	def __init__(self, device, datasets_dict, model, tokenizer, embedding_split_perc):
		GetEmbeddings.__init__(self.__class__.__name__, embedding_split_perc,
                         device, tokenizer, BertLastLayer(model).to(device), embedding_dim = 768)
		self.datasets_dict = datasets_dict

  
	def run(self):
		for ds_name, dataset in self.datasets_dict.items():
			self.get_embeddings(ds_name, dataset)
			# run clusering