
from GetEmbeddings import GetEmbeddings
import torch.nn as nn

class BertLastLayer(nn.Module):
	def __init__(self, bert):
		super(BertLastLayer, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		return self.pre_trained_bert(**x, output_hidden_states=True).last_hidden_state[:, 0, :]
		
	

class MainApproch(GetEmbeddings):
	def __init__(self, device, dataloaders, model, tokenizer, embedding_split_perc, batch_size):
		GetEmbeddings.__init__(self, 'MainApproch', embedding_split_perc,
                         device, tokenizer, BertLastLayer(model).to(device), batch_size,
                         embeddings_dim = 768)
		
		self.dataloaders = dataloaders

  
	def run(self):
		for ds_name, dls in self.dataloaders.items():
		#for ds_name, dataset in self.datasets_dict.items():
			#self.get_embeddings(ds_name, dataset)
			self.get_embeddings(ds_name, dls)
   
			# run clusering