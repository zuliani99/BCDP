
from ClusteringEmbeddings import ClusteringEmbeddings
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


class LayerWise(ClusteringEmbeddings):
	def __init__(self, device, dataloaders, model, tokenizer, embedding_split_perc, timestamp, spherical_kmenas):
		
		super().__init__(self.__class__.__name__, embedding_split_perc,
                         device, tokenizer, BertLayersWise(model).to(device),
                         embeddings_dim = 12 * 768)
  
		self.dataloaders = dataloaders
		self.timestamp = timestamp
		self.spherical_kmenas = spherical_kmenas

  
	def run(self):
     
		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')
     
		for ds_name, dls in self.dataloaders.items():
      
			print(f'--------------- {ds_name} ---------------')

			self.get_embeddings(ds_name, dls)
   
			# run clusering
			self.faiss_clusering.run_faiss_kmeans(ds_name, self.__class__.__name__, self.timestamp, self.spherical_kmenas)
	
		print(f'\n---------------------------------- END {self.__class__.__name__} ----------------------------------\n\n')
   
