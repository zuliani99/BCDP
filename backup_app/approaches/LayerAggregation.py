
from TrainEvaluate import Train_Evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils import init_params, read_embbedings_pt



class SelfAttentionLayer(nn.Module):
	def __init__(self, input_size, output_size, attention_heads):
		super(SelfAttentionLayer, self).__init__()

		self.input_size = input_size
		self.attention_heads = attention_heads

		# Linear transformations for Query, Key, and Value
		self.W_q = nn.Linear(input_size, input_size)
		self.W_k = nn.Linear(input_size, input_size)
		self.W_v = nn.Linear(input_size, input_size)

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
		Q = self.W_q(x)
		K = self.W_k(x)
		V = self.W_v(x)

		# Scaled dot-product attention
		attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_size, dtype=torch.float32))
		attention_weights = F.softmax(attention_scores, dim=-1)
		attended_values = torch.matmul(attention_weights, V)

		outputs = self.classifier(attended_values)

		return outputs, attended_values
	
	



class LayerAggregation(Train_Evaluate):
	def __init__(self, params, datasets_name, timestamp):
		
		self.datasets_name = datasets_name
		self.timestamp = timestamp
  
		#params['model'] = Bert_Layer_aggregation(params['model'], params['batch_size'])
		params['model'] = SelfAttentionLayer(12 * 768, output_size=2, attention_heads=8)
		params['model'].apply(init_params)
		params['embeddings_dim'] = 768 * 12

		super().__init__(self.__class__.__name__, params)
  
	def create_tensor_dataset(self):
		pass
  

	def run(self):

		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	
  
		for ds_name in self.datasets_name:
      
			print(f'--------------- {ds_name} ---------------')
			
			x_train, x_val, x_test, y_train, y_val, y_test = read_embbedings_pt(ds_name, bool_validation=True)

			self.fit(ds_name, DataLoader(TensorDataset(x_train, y_train)), DataLoader(TensorDataset(x_val, y_val)))
   
			# we can for eaxample save these metrics to compare with the additional embedding
			_, _ = self.test(DataLoader(TensorDataset(x_test, y_test)))
   
			#self.get_embeddings(ds_name, dls['dataset'], self.embeddings_dim)
			#self.get_embeddings(ds_name, dls)
			
			# run clusering
			#self.faiss_clusering.run_faiss_kmeans(ds_name, self.__class__.__name, self.timestamp)
   
		print(f'\n---------------------------------- END {self.__class__.__name__} ----------------------------------\n\n')

   


