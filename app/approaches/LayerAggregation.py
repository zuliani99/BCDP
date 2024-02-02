
from TrainEvaluate import Train_Evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



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
	
	


class Bert_Layer_aggregation(nn.Module):
	def __init__(self, bert, batch_size):
		super(Bert_Layer_aggregation, self).__init__()
		self.batch_size = batch_size
		self.pre_trained_bert = bert
		self.self_attention_layer = SelfAttentionLayer(12 * 768, output_size=2, attention_heads=8)
		self.freeze_layers()
		self.self_attention_layer.apply(init_params)
		
		
	def forward(self, x):
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		hidden_states_batches = outputs[2]
		aggregated_tensor = torch.cat([h_state[:,0,:] for h_state in hidden_states_batches[1:]], dim=1)
		outputs, attentions = self.self_attention_layer(aggregated_tensor)
		return outputs, attentions 
		

	def freeze_layers(self):
		for param in self.pre_trained_bert.parameters():
			param.requires_grad = False
		



class LayerAggregation(Train_Evaluate):
	def __init__(self, params, dataloaders, timestamp):
		
		self.dataloaders = dataloaders
		self.timestamp = timestamp
  
		params['model'] = Bert_Layer_aggregation(params['model'], params['batch_size'])
		params['embeddings_dim'] = 768 * 12

		super().__init__(self.__class__.__name__, params)
  

	def run(self):

		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	
  
		for ds_name, dls in self.dataloaders.items():
      
			print(f'--------------- {ds_name} ---------------')

			self.fit(ds_name, self.__class__.__name__, dls['train'], dls['val'])
   
			# we can for eaxample save these metrics to compare with the additional embedding
			_, _ = self.test(dls['test'])
   
			#self.get_embeddings(ds_name, dls['dataset'], self.embeddings_dim)
			self.get_embeddings(ds_name, dls)
			
			# run clusering
			self.faiss_clusering.run_faiss_kmeans(ds_name, self.__class__.__name, self.timestamp)
   
		print(f'\n---------------------------------- END {self.__class__.__name__} ----------------------------------\n\n')

   


def init_params(m):
	if isinstance(m, nn.Linear):
		init.normal_(m.weight, std=1e-3)
		if m.bias is not None: init.constant_(m.bias, 0)
	elif isinstance(m, nn.Sequential):
		for c in list(m.children()): init_params(c)