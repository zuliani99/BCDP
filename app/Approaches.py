

import torch
import torch.nn as nn

from utils import get_text_dataloaders, read_embbedings
from FaissClustering import Faiss_KMEANS
from TrainEvaluate import Train_Evaluate

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import os


class Approaches(object):
	def __init__(self, timestamp, base_embeds_model, embeddings_dim = None, bool_ablations = False):
		self.timestamp = timestamp
		self.embeddings_dim = embeddings_dim
		self.bool_ablations = bool_ablations
		self.base_embeds_model = base_embeds_model
		self.faiss_kmeans = Faiss_KMEANS()
  
  
	def run_clustering(self, ds_name, method_name, data):
    
		x_train, x_test, y_train, y_test = data
       
		if self.bool_ablations:
			print(f'--------------- {ds_name} - PCA & TSNE ablations ---------------')
   
			pca = PCA(n_components=2)
			pca_x = {}
       
			print(f'\nRunning PCA ablations\n')
       
			path_reduced_embeds = f'app/embeddings/{self.base_embeds_model}/{ds_name}/ablations/PCA_{method_name}'
			for str_x, x in [('x_train', x_train), ('x_test', x_test)]:
				if os.path.exists(f'{path_reduced_embeds}_{str_x}.npy'):
					print(f' => Loading reduced embeddings {str_x}')
					pca_x[str_x] = np.load(f'{path_reduced_embeds}_{str_x}.npy')
					print(' DONE\n')
				else:
					print(f' => Running PCA on the embedding of {str_x} - {ds_name} from {method_name}')
					pca_x[str_x] = pca.fit_transform(x)
					print(' => Saving reduced embeddings:')
					np.save(f'{path_reduced_embeds}_{str_x}.npy', pca_x[str_x])
					print(' DONE\n')


			self.faiss_kmeans.run_faiss_kmeans(
        		ds_name,
           		f'{method_name}_PCA',
              	self.timestamp,
                'our_ablations',
				(np.copy(pca_x['x_train']), np.copy(pca_x['x_test']), y_train, y_test)
     		)

			print('----------------------------------------------------------------\n')
			
		else:
			print(f'--------------- {ds_name} ---------------')
	 
			# run clusering
			self.faiss_kmeans.run_faiss_kmeans(
				ds_name,
				method_name,
				self.timestamp,
    			'our_approaches',
				data
			)

			print('------------------------------------------\n')



class MainApproch(Approaches):
	def __init__(self, common_parmas, bool_ablations):
     
		super().__init__(common_parmas['timestamp'], common_parmas['base_embeds_model'], bool_ablations = bool_ablations)
		self.datasets_name = common_parmas['datasets_name']


	def run(self):
	
		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
		  
		for ds_name in self.datasets_name:

			x_train, x_test, y_train, y_test = read_embbedings(ds_name, self.base_embeds_model)
			# [#sentence, #layers, 768] -> [#sentence, 768] 

			x_train = np.squeeze(np.copy(x_train[:,-1,:]))
			x_test = np.squeeze(np.copy(x_test[:,-1,:]))
      
      
			self.run_clustering(ds_name, self.__class__.__name__, (x_train, x_test, y_train, y_test))
			

		print(f'\n---------------------------------- END {self.__class__.__name__ } ----------------------------------\n\n')
  
  
  


class LayerWise(Approaches):
	def __init__(self, common_parmas, embeddings_dim, bool_ablations):
     
		super().__init__(common_parmas['timestamp'], common_parmas['base_embeds_model'], embeddings_dim = embeddings_dim, bool_ablations = bool_ablations)
  
		self.datasets_name = common_parmas['datasets_name']
		self.name = f'{self.__class__.__name__}_{embeddings_dim}'

  
	def run(self):
	 
		print(f'---------------------------------- START {self.name} ----------------------------------')
	 
		for ds_name in self.datasets_name:
			
			x_train, x_test, y_train, y_test = read_embbedings(ds_name, self.base_embeds_model)
   
			if self.embeddings_dim == 768:
				# [#sentence, #layers, 768] -> [#sentence, 768] 
       
				# mean of oll CLS tokens
				x_train, x_test = np.squeeze(np.mean(x_train, axis=1)), np.squeeze(np.mean(x_test, axis=1))
			else:
				# [#sentence, #layers, 768] -> [#sentence, #layers x 768] 
				
				# reshape the embeddings
				x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
				x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

			self.run_clustering(ds_name, self.name, (x_train, x_test, y_train, y_test))

		
		print(f'\n---------------------------------- END {self.name} ----------------------------------\n\n')
   




class SelfAttentionLayer(nn.Module):
	def __init__(self, input_size, n_layers, output_size, attention_heads):
		super(SelfAttentionLayer, self).__init__()

		self.multihead_att = nn.MultiheadAttention(input_size, attention_heads)
		self.new_input_size = input_size * n_layers

		# Linear transformation for the output of attention heads
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(self.new_input_size, self.new_input_size // 2),
   			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(self.new_input_size // 2, self.new_input_size // 4),
   			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(self.new_input_size // 4, output_size)
		)
  
	def forward(self, x):
		mhsa = self.multihead_att(x, x, x, need_weights = False)[0]
		attn_output = torch.reshape(mhsa, (mhsa.shape[0], mhsa.shape[1] * mhsa.shape[2]))
		outputs = self.classifier(attn_output)
		return outputs, attn_output




class LayerAggregation(Approaches):
	
	def __init__(self, params, common_parmas, n_layers, embeddings_dim, bool_ablations):
		
		super().__init__(common_parmas['timestamp'], common_parmas['base_embeds_model'], embeddings_dim = embeddings_dim, bool_ablations = bool_ablations)
		self.datasets_name = common_parmas['datasets_name']
		self.n_layers = n_layers
  
		self.tran_evaluate = Train_Evaluate(self.__class__.__name__, params, self.base_embeds_model, SelfAttentionLayer(self.embeddings_dim, n_layers, output_size=2, attention_heads=8))
  
  
	def get_LayerAggregation_Embeddigns(self, dataloader):
  
		LA_embeds = torch.empty((0, self.embeddings_dim * self.n_layers), dtype=torch.float32, device=self.tran_evaluate.device)		
		#LA_embeds = np.empty((0, self.embeddings_dim * self.n_layers), dtype=np.float32)		
		
		LA_labels = torch.empty((0), device=self.tran_evaluate.device)		
		#LA_labels = np.empty((0), dtype=np.int8)		

  
		self.tran_evaluate.model.eval()

		with torch.inference_mode(): # Allow inference mode
			for bert_embeds, labels in dataloader:

				bert_embeds = bert_embeds.to(self.tran_evaluate.device)
				labels = labels.to(self.tran_evaluate.device)
			
				_, embeds = self.tran_evaluate.model(bert_embeds)

				LA_embeds = torch.cat((LA_embeds, embeds), dim=0)
				#LA_embeds = np.vstack((LA_embeds, embeds.detach().cpu().numpy()))
				
				LA_labels = torch.cat((LA_labels, torch.flatten(labels)))
				#LA_labels = np.append(LA_labels, labels.numpy().flatten())

		#return LA_embeds, LA_labels
		return LA_embeds.cpu().numpy(), LA_labels.cpu().numpy()


					 
	def run(self):

		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	
  
		for ds_name in self.datasets_name:
      
			self.tran_evaluate.load_initial_checkpoint()

			x_train, x_val, x_test, y_train, y_val, y_test = read_embbedings(ds_name, self.base_embeds_model, bool_validation=True)

			# create tensor dataloaders
			train_dl, val_dl, test_dl = get_text_dataloaders(x_train, x_val, x_test, y_train, y_val, y_test, self.tran_evaluate.batch_size)

			self.tran_evaluate.fit(ds_name, train_dl, val_dl)
   
			# we can for eaxample save these metrics to compare with the additional embedding
			_, _ = self.tran_evaluate.test(test_dl)
   
			print(' => Obtaining the Layer Aggregation embeddings:')
			x_train, y_train = self.get_LayerAggregation_Embeddigns(train_dl)
			x_val, y_val = self.get_LayerAggregation_Embeddigns(val_dl)
			x_test, y_test = self.get_LayerAggregation_Embeddigns(test_dl)
			print(' DONE\n')
   
			x_train = np.vstack((x_train, x_val))#.astype(np.float32)
			y_train = np.append(y_train, y_val)#.astype(np.int8)
   
			y_train[y_train == 0] = -1
			y_test[y_test == 0] = -1
      
			torch.cuda.empty_cache()
   
			self.run_clustering(ds_name, self.__class__.__name__, (x_train, x_test, y_train.astype(np.int8), y_test.astype(np.int8)))
   
		print(f'\n---------------------------------- END {self.__class__.__name__} ----------------------------------\n\n')

   

