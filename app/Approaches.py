

import torch
import torch.nn as nn

from utils import get_text_dataloaders, read_embbedings_pt
from FaissClustering import Faiss_KMEANS
from TrainEvaluate import Train_Evaluate

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class Approaches(object):
	def __init__(self, timestamp, embeddings_dim = None, bool_ablations = False):
		self.timestamp = timestamp
		self.embeddings_dim = embeddings_dim
		self.bool_ablations = bool_ablations
		self.faiss_kmeans = Faiss_KMEANS()
  
  
	def run_clustering(self, ds_name, method_name, data):
    
		x_train, x_test, y_train, y_test = data
    
		ablations_dict = {}
   
		if self.bool_ablations:
			print(f'--------------- {ds_name} ---------------')
				
			pca = PCA(n_components=2)
			tsne = TSNE()
	
			ablations_dict['pca'] = {'x_train': pca.fit_transform(x_train), 'x_test': pca.fit_transform(x_train)}
			ablations_dict['tsne'] = {'x_train': tsne.fit_transform(x_train), 'x_test': tsne.fit_transform(x_train)}
				
			for ab_name, x_reduced in ablations_dict.items():
					
				self.faiss_kmeans.run_faiss_kmeans(ds_name, f'{method_name}_{ab_name}', self.timestamp, 'our_ablations'
									  (x_reduced['x_train'], x_reduced['x_test'], y_train, y_test))
			else:
				print(f'--------------- {ds_name} ---------------')
	 
				# run clusering
				self.faiss_kmeans.run_faiss_kmeans(ds_name, method_name, self.timestamp, 'our_approaches'
										(x_train, x_test, y_train, y_test))



class MainApproch(Approaches):
	def __init__(self, common_parmas, bool_ablations):
     
		super().__init__(common_parmas['timestamp'], bool_ablations = bool_ablations)
		self.datasets_name = common_parmas['datasets_name']
		self.choosen_model_embedding = common_parmas['choosen_model_embedding']



	def run(self):
	
		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
		  
		for ds_name in self.datasets_name:

			x_train, x_test, y_train, y_test = read_embbedings_pt(ds_name, self.choosen_model_embedding, bool_numpy=True)
			x_train = x_train[-1][:, 0, :]
			x_test = x_test[-1][:, 0, :]
   
			self.run_clustering(ds_name, self.__class__.__name__, (x_train, x_test, y_train, y_test))
			

		print(f'\n---------------------------------- END {self.__class__.__name__ } ----------------------------------\n\n')
  
  
  


class LayerWise(Approaches):
	def __init__(self, common_parmas, embeddings_dim, bool_ablations):
     
		super().__init__(common_parmas['timestamp'], embeddings_dim = embeddings_dim, bool_ablations = bool_ablations)
  
		self.datasets_name = common_parmas['datasets_name']
		self.choosen_model_embedding = common_parmas['choosen_model_embedding']
		self.name = f'{self.__class__.__name__}_{embeddings_dim}'

  
	def run(self):
	 
		print(f'---------------------------------- START {self.name} ----------------------------------')
	 
		for ds_name in self.datasets_name:
			
			x_train, x_test, y_train, y_test = read_embbedings_pt(ds_name, self.choosen_model_embedding, bool_numpy = True)
   
			if self.embeddings_dim == 768:		
				x_train, x_test = np.mean(x_train, axis=1), np.mean(x_test, axis=1)
	
			self.run_clustering(ds_name, self.__class__.__name__, (x_train, x_test, y_train, y_test))

		
		print(f'\n---------------------------------- END {self.name} ----------------------------------\n\n')
   




class SelfAttentionLayer(nn.Module):
	def __init__(self, input_size, output_size, attention_heads):
		super(SelfAttentionLayer, self).__init__()

		self.multihead_att = nn.MultiheadAttention(input_size, attention_heads)

		# Linear transformation for the output of attention heads
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(input_size, input_size // 2),
   			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(input_size // 2, input_size // 4),
   			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(input_size // 4, output_size)
		)
  
	def forward(self, x):
	 
		# Linear transformations for Query, Key, and Value
		attn_output, attn_output_weights = self.multihead_att(x, x, x)
  
		outputs = self.classifier(attn_output)

		return outputs, attn_output_weights




class LayerAggregation(Approaches):
	
	def __init__(self, params, common_parmas, embeddings_dim, bool_ablations):
		
		super().__init__(common_parmas['timestamp'], embeddings_dim = embeddings_dim, bool_ablations = bool_ablations)
		self.datasets_name = common_parmas['datasets_name']
		self.choosen_model_embedding = common_parmas['choosen_model_embedding']
  
		self.tran_evaluate = Train_Evaluate(self.__class__.__name__, params, SelfAttentionLayer(self.embeddings_dim, output_size=2, attention_heads=8))
  
  
	def get_LayerAggregation_Embeddigns(self, dataloader):
  
		LA_embeds = torch.empty((0, self.embeddings_dim), dtype=torch.float32, device=self.device)		
		LA_labels = torch.empty((0), device=self.device)		
  
		self.tran_evaluate.model.eval()

		with torch.inference_mode(): # Allow inference mode
			for bert_embeds, labels in dataloader:

				bert_embeds = bert_embeds.to(self.device)
				labels = labels.to(self.device)
			
				_, embeds = self.tran_evaluate.model(bert_embeds)

				LA_embeds = torch.cat((LA_embeds, embeds), dim=1)
				LA_labels = torch.cat((LA_embeds, labels))

		return LA_embeds.cpu().numpy(), LA_labels.cpu().numpy()


					 
	def run(self):

		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	
  
		for ds_name in self.datasets_name:

			x_train, x_val, x_test, y_train, y_val, y_test = read_embbedings_pt(ds_name, self.choosen_model_embedding, bool_validation=True)
   			
			# create tensor dataloaders
			train_dl, val_dl, test_dl = get_text_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test)

			self.tran_evaluate.fit(ds_name, train_dl, val_dl)
   
			# we can for eaxample save these metrics to compare with the additional embedding
			_, _ = self.tran_evaluate.test(test_dl)
   
			x_train, y_train = self.get_LayerAggregation_Embeddigns(train_dl)
			x_val, y_val = self.get_LayerAggregation_Embeddigns(train_dl)
			x_test, y_test = self.get_LayerAggregation_Embeddigns(train_dl)
   
			x_train = np.vstack((x_train, x_val))
			y_train = np.append(y_train, y_val)
   
			self.run_clustering(ds_name, self.__class__.__name__, (x_train, x_test, y_train, y_test))
   
		print(f'\n---------------------------------- END {self.__class__.__name__} ----------------------------------\n\n')

   

