
from TrainEvaluate import Train_Evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F



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
		self.W_o = nn.Linear(input_size, output_size)


	def forward(self, x):
		# Linear transformations for Query, Key, and Value
		Q = self.W_q(x)
		K = self.W_k(x)
		V = self.W_v(x)

		# Split into multiple attention heads
		Q = self.split_heads(Q)
		K = self.split_heads(K)
		V = self.split_heads(V)

		# Scaled dot-product attention
		attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_size, dtype=torch.float32))
		attention_weights = F.softmax(attention_scores, dim=-1)
		attended_values = torch.matmul(attention_weights, V)

		# Concatenate attention heads
		attended_values = self.concat_heads(attended_values)

		# Linear transformation for the output
		outputs = self.W_o(attended_values)

		return outputs, attended_values


	def split_heads(self, x):
		batch_size, features = x.size()
		head_size = features // self.attention_heads

		x = x.view(batch_size, self.attention_heads, head_size)
		x = x.permute(0, 2, 1).contiguous()
		x = x.view(batch_size * self.attention_heads, head_size)

		return x


	def concat_heads(self, x):
		batch_size_heads, head_size = x.size()
		batch_size = batch_size_heads // self.attention_heads

		x = x.view(batch_size, self.attention_heads, head_size)
		x = x.permute(0, 2, 1).contiguous()
		x = x.view(batch_size, self.attention_heads * head_size)

		return x
	
	


class Bert_Layer_aggregation(nn.Module):
	def __init__(self, bert, batch_size):
		super(Bert_Layer_aggregation, self).__init__()
		self.batch_size = batch_size
		self.pre_trained_bert = bert
		self.self_attention_layer = SelfAttentionLayer(12 * 768, output_size=2, attention_heads=8)
		self.freeze_layers()
		
		
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
	def __init__(self, params, dataloaders):
		
		self.dataloaders = dataloaders
		params['model'] = Bert_Layer_aggregation(params['model'], params['batch_size'])
		params['embeddings_dim'] = 768 * 12
		
		super().__init__(self.__class__.__name__, params)
  

	def run(self):
	 
		for ds_name, dls in self.dataloaders.items():

			self.fit(ds_name, self.__class__.__name__, dls['train_dl'], dls['val_dl'])
   
			# we can for eaxample save these metrics to compare with the additional embedding
			test_accuracy, test_loss = self.test(dls['test_dl'])
   
			#self.get_embeddings(ds_name, dls['dataset'], self.embeddings_dim)
			self.get_embeddings(ds_name, dls)
			
			# run clusering
			#self.faiss_clusering.run_faiss_kmeans(ds_name, self.__calss__.__name)
