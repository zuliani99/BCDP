
import torch
from tqdm import tqdm
import numpy as np
import os

from FaissClustering import FaissClustering


class ClusteringEmbeddings(object):
	def __init__(self, name, embedding_split_perc, device, tokenizer, model, embeddings_dim = None):
		self.name = name
		self.embedding_split_perc = embedding_split_perc
		self.device = device
		self.tokenizer = tokenizer
		self.model = model.to(device)
		self.embeddings_dim = embeddings_dim
		self.faiss_clusering = FaissClustering()
	
	 

	def get_embeddings(self, ds_name, dataloaders):
		
		print('OBTAINING THE EMBEDDINGS:')
     
		if self.embeddings_dim is None: return

		for dl_name, dataloader in dataloaders.items():

			save_labels_npy, save_embeddings_npy = False, False
			
			if not os.path.exists(f'app/embeddings/{ds_name}/{dl_name}_labels.npy'): save_labels_npy = True
			if not os.path.exists(f'app/embeddings/{ds_name}/{self.name}/{dl_name}_embeddings.npy'): save_embeddings_npy = True

			if not save_labels_npy and not save_embeddings_npy: continue


			pbar = tqdm(enumerate(dataloader), total = len(dataloader), leave=False, desc=f'Obtaining embedding for {ds_name} - {dl_name}')
   
   
			labels_npy = np.empty((0))
			embeddings_tensor = torch.empty((0, self.embeddings_dim)).to(self.device)
   
			with torch.inference_mode(): # Allow inference mode
				for idx, (dictionary, labels) in pbar:
        
					for key in list(dictionary.keys()):
						dictionary[key] = dictionary[key].to(self.device)
					
					if save_labels_npy:
						labels_npy = np.append(labels_npy, np.array([-1 if x == 0 else x for x in torch.squeeze(labels).numpy()]))

					if save_embeddings_npy:
						if self.name == 'LayerAggregation':
							_, embeds = self.model(dictionary)
						else:
							embeds = self.model(dictionary)
       
					if save_embeddings_npy:
						embeddings_tensor = torch.cat((embeddings_tensor, embeds), dim=0)

						if(idx % 100 == 0):
							if not os.path.exists(f'app/embeddings/{ds_name}/{self.name}/{dl_name}_embeddings.npy'):
								np.save(f'app/embeddings/{ds_name}/{self.name}/{dl_name}_embeddings.npy',
														embeddings_tensor.cpu().detach().numpy()
													)

							else:
								prev_embeddings = np.load(f'app/embeddings/{ds_name}/{self.name}/{dl_name}_embeddings.npy')
								np.save(f'app/embeddings/{ds_name}/{self.name}/{dl_name}_embeddings.npy', np.vstack((prev_embeddings, embeddings_tensor.cpu().detach().numpy())))

							embeddings_tensor = torch.empty((0, self.embeddings_dim)).to(self.device)
						
      
				if save_labels_npy: np.save(f'app/embeddings/{ds_name}/{dl_name}_labels.npy', labels_npy)
				if save_embeddings_npy and (idx % 100 != 0):
					prev_embeddings = np.load(f'app/embeddings/{ds_name}/{self.name}/{dl_name}_embeddings.npy')
					np.save(f'app/embeddings/{ds_name}/{self.name}/{dl_name}_embeddings.npy', np.vstack((prev_embeddings, embeddings_tensor.cpu().detach().numpy())))

		print('	-> DONE')
