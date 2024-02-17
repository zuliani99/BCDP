
from TrainEvaluate import Train_Evaluate

import torch
import torch.nn as nn

from utils import get_text_dataloaders, init_params, read_embbedings_pt
from FaissClustering import Faiss_KMEANS

import numpy as np


class MainApproch(object):
	def __init__(self, datasets_name, timestamp):
		
		self.datasets_name = datasets_name
		self.timestamp = timestamp
		self.faiss_kmeans = Faiss_KMEANS()

  
	def run(self):
	
		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
		  
		for ds_name in self.datasets_name:

			print(f'--------------- {ds_name} ---------------')

			x_train, x_test, y_train, y_test = read_embbedings_pt(ds_name, bool_numpy=True)
			x_train = x_train[-1][:, 0, :]
			x_test = x_test[-1][:, 0, :]
   
			# run clusering
			self.faiss_kmeans.run_faiss_kmeans(ds_name, self.__class__.__name__, self.timestamp
									  (x_train, x_test, y_train, y_test))
			

		print(f'\n---------------------------------- END {self.__class__.__name__ } ----------------------------------\n\n')
  
  
  
class LayerWise(object):
	def __init__(self, datasets_name, timestamp, embeddings_dim):
  
		self.datasets_name = datasets_name
		self.timestamp = timestamp
		self.embeddings_dim = embeddings_dim
		self.name = f'{self.__class__.__name__}_{embeddings_dim}'
		self.faiss_kmeans = Faiss_KMEANS()

  
	def run(self):
	 
		print(f'---------------------------------- START {self.name} ----------------------------------')
	 
		for ds_name in self.datasets_name:
	  
			print(f'--------------- {ds_name} ---------------')

			x_train, x_test, y_train, y_test = read_embbedings_pt(ds_name, bool_numpy = True)
   
			if self.embeddings_dim == 768:		
				x_train, x_test = np.mean(x_train, axis=1), np.mean(x_test, axis=1)

			# run clusering
			self.faiss_kmeans.run_faiss_kmeans(ds_name, self.__class__.__name__, self.timestamp
									  (x_train.numpy(), x_test.numpy(), y_train.numpy(), y_test.numpy()))
	
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




class LayerAggregation(Train_Evaluate):
	def __init__(self, params, datasets_name, timestamp, embeddings_dim):
		
		self.datasets_name = datasets_name
		self.timestamp = timestamp
		self.embeddings_dim = embeddings_dim
		self.faiss_kmeans = Faiss_KMEANS()

  
		params['model'] = SelfAttentionLayer(self.embeddings_dim, output_size=2, attention_heads=8)
		params['model'].apply(init_params)

		super().__init__(self.__class__.__name__, params)
  
  
	def get_LayerAggregation_Embeddigns(self, dataloader):
  
		LA_embeds = torch.empty((0, self.embeddings_dim), dtype=torch.float32, device=self.device)		
		LA_labels = torch.empty((0), device=self.device)		
  
		self.model.eval()

		with torch.inference_mode(): # Allow inference mode
			for bert_embeds, labels in dataloader:

				bert_embeds = bert_embeds.to(self.device)
				labels = labels.to(self.device)
			
				_, embeds = self.model(bert_embeds)

				LA_embeds = torch.cat((LA_embeds, embeds), dim=1)
				LA_labels = torch.cat((LA_embeds, labels))

		return LA_embeds.cpu().numpy(), LA_labels.cpu().numpy()

					 
	def run(self):

		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	
  
		for ds_name in self.datasets_name:
	  
			print(f'--------------- {ds_name} ---------------')
			
			x_train, x_val, x_test, y_train, y_val, y_test = read_embbedings_pt(ds_name, bool_validation=True)
   			
			# create tensor dataloaders
			train_dl, val_dl, test_dl = get_text_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test)

			self.fit(ds_name, train_dl, val_dl)
   
			# we can for eaxample save these metrics to compare with the additional embedding
			_, _ = self.test(test_dl)
   
			x_train, y_train = self.get_LayerAggregation_Embeddigns(train_dl)
			x_val, y_val = self.get_LayerAggregation_Embeddigns(train_dl)
			x_test, y_test = self.get_LayerAggregation_Embeddigns(train_dl)
   
			x_train = np.vstack((x_train, x_val))
			y_train = np.append(y_train, y_val)#np.vstack((y_train, y_val))
   
			# run clusering
			self.faiss_kmeans.run_faiss_kmeans(ds_name, self.__class__.__name__, self.timestamp
									  (x_train, x_test, y_train, y_test))
   
   
		print(f'\n---------------------------------- END {self.__class__.__name__} ----------------------------------\n\n')

   

