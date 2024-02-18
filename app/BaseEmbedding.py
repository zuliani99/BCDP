
import torch
import torch.nn as nn

import numpy as np
import os

class BaseEmebddingModel(nn.Module):
	def __init__(self, bert):
		super(BaseEmebddingModel, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		hidden_states_batches = outputs.hidden_states
  
		return torch.cat([torch.unsqueeze(h_state[:,0,:], dim=1) for h_state in hidden_states_batches[1:]], dim=1)


class BaseEmbedding(object):
	def __init__(self, model, device, dataloaders, n_layers):
		self.device = device
		self.model = BaseEmebddingModel(model).to(self.device)
		self.embeddings_dim = (n_layers, 768)
		self.dataloaders = dataloaders
  
  
	def save_base_embeddings(self, model_name):
		
		for ds_name, dls in self.dataloaders.items():
			print(f' for {ds_name}')		
   
			for dl_name, dataloader in dls.items():
       
				path_embeddings = f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.npy'
				path_labels = f'app/embeddings/{model_name}/{ds_name}/labels/{dl_name}_labels.npy'
    
				save_labels_npy, save_embeddings_npy = False, False
					
				if not os.path.exists(f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.npy'): save_embeddings_npy = True
				if not os.path.exists(f'app/embeddings/{model_name}/{ds_name}/labels/{dl_name}_labels.npy'): save_labels_npy = True
    
				if not save_labels_npy and not save_embeddings_npy: continue

				labels_npy = np.empty((0), dtype=np.int8)
				embeddings_npy = np.empty((0, self.embeddings_dim[0], self.embeddings_dim[1]))
	
				with torch.inference_mode(): # Allow inference mode
					for idx, (dictionary, labels) in enumerate(dataloader):
						
						if save_labels_npy:
							labels_npy = np.append(labels_npy, np.array([-1 if x == 0 else x for x in torch.squeeze(labels).numpy()]))

						if save_embeddings_npy:
							for key in list(dictionary.keys()): dictionary[key] = dictionary[key].to(self.device)
			
							embeds = self.model(dictionary).cpu().detach().numpy()
       		
							embeddings_npy = np.vstack((embeddings_npy, embeds))

							if(idx % 100 == 0):
								if not os.path.exists(path_embeddings): np.save(path_embeddings, embeddings_npy)
								else: np.save(path_embeddings, np.vstack((np.load(path_embeddings), embeddings_npy)))

								embeddings_npy = np.empty((0, self.embeddings_dim[0], self.embeddings_dim[1]))
							
		
					if save_labels_npy: np.save(path_labels, labels_npy)

					if save_embeddings_npy and (idx % 100 != 0):
						np.save(path_embeddings, np.vstack((np.load(path_embeddings), embeddings_npy)))

			print('	-> DONE')
