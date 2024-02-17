
from utils import read_embbedings_pt
#from ClusteringEmbeddings import ClusteringEmbeddings
#import torch.nn as nn
import torch


'''class BertLayersWise_last4(nn.Module):
	def __init__(self, bert):
		super(BertLayersWise_last4, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		hidden_states_batches = outputs[-1]
		# (1,728*4)
		return torch.cat([h_state[:,0,:] for h_state in hidden_states_batches[:-4]], dim=1)


class BertLayersWise_mean_cls(nn.Module):
	def __init__(self, bert):
		super(BertLayersWise_mean_cls, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		hidden_states_batches = outputs[-1]
		# (1,728)
		return torch.squeeze(
      		torch.mean(torch.cat([h_state[:,0,:] for h_state in hidden_states_batches[1:]], dim=1), dim=1)
		)'''


class LayerWise(object):#(ClusteringEmbeddings):
	def __init__(self, datasets_name, timestamp, embeddings_dim):
		
		
		'''super().__init__(self.__class__.__name__, embedding_split_perc,
                         device,
                         BertLayersWise_last4(model).to(device) if embeddings_dim == 4 * 768 else BertLayersWise_mean_cls(model).to(device),
                         embeddings_dim = embeddings_dim)'''
  
		self.datasets_name = datasets_name
		self.timestamp = timestamp
		self.embeddings_dim = embeddings_dim
		self.name = f'{self.__class__.__name__}_{embeddings_dim}'

  
	def run(self):
     
		print(f'---------------------------------- START {self.name} ----------------------------------')
     
		for ds_name in self.datasets_name:
      
			print(f'--------------- {ds_name} ---------------')

			#self.get_embeddings(ds_name, dls)
			x_train, x_test, y_train, y_test = read_embbedings_pt(ds_name)
			if self.embeddings_dim == 4 * 768:
				x_train = torch.cat([h_state[:,0,:] for h_state in x_train[:-4]], dim=1)
				x_test = torch.cat([h_state[:,0,:] for h_state in x_test[:-4]], dim=1)
            
			else:
				x_train = torch.mean(torch.cat([h_state[:,0,:] for h_state in x_train[1:]], dim=1), dim=1)
				x_test = torch.mean(torch.cat([h_state[:,0,:] for h_state in x_test[1:]], dim=1), dim=1)

			# run clusering
			#self.faiss_clusering.run_faiss_kmeans(ds_name, self.name, self.timestamp)
	
		print(f'\n---------------------------------- END {self.name} ----------------------------------\n\n')
   