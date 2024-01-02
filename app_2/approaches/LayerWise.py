
from GetEmbeddings import GetEmbeddings
import torch.nn as nn
import torch

class BertLayersWise(nn.Module):
	def __init__(self, bert):
		super(BertLayersWise, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		hidden_states_batches = outputs[2]
		# (1,728*12)
		return torch.cat([h_state[:,0,:] for h_state in hidden_states_batches[1:]], dim=1)


class LayerWise(GetEmbeddings):
	def __init__(self, device, dataloaders, model, tokenizer, embedding_split_perc, batch_size):
		GetEmbeddings.__init__(self, 'LayerWise', embedding_split_perc,
                         device, tokenizer, BertLayersWise(model).to(device), batch_size,
                         embeddings_dim = 12 * 768)
		self.dataloaders = dataloaders

  
	def run(self):
		for ds_name, dls in self.dataloaders.items():
		#for ds_name, dataset in self.datasets_dict.items():
			#self.get_embeddings(ds_name, dataset)
			self.get_embeddings(ds_name, dls)
   
			# run clusering