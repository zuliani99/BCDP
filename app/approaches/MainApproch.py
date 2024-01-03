
from ClusteringEmbeddings import ClusteringEmbeddings
import torch.nn as nn

class BertLastLayer(nn.Module):
	def __init__(self, bert):
		super(BertLastLayer, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		return self.pre_trained_bert(**x, output_hidden_states=True).last_hidden_state[:, 0, :]
		
	

class MainApproch(ClusteringEmbeddings):
	def __init__(self, device, dataloaders, model, tokenizer, embedding_split_perc):
		ClusteringEmbeddings.__init__(self, self.__class__.__name__, embedding_split_perc,
                         device, tokenizer, BertLastLayer(model).to(device),
                         embeddings_dim = 768)
		
		self.dataloaders = dataloaders

  
	def run(self):
		for ds_name, dls in self.dataloaders.items():
		#for ds_name, dataset in self.datasets_dict.items():
			#self.get_embeddings(ds_name, dataset)
			self.get_embeddings(ds_name, dls)
   
			# run clusering
			#self.faiss_clusering.run_faiss_kmeans(ds_name, self.__calss__.__name)