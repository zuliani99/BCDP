
import torch
import torch.nn as nn
import os

class BaseEmebddingModel(nn.Module):
	def __init__(self, bert):
		super(BaseEmebddingModel, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		hidden_states_batches = outputs[-1]
  
		# (1,768*12) for bert
		# (1,768*6) for distilbert

		return torch.cat([h_state[:,0,:] for h_state in hidden_states_batches[1:]], dim=1)

class BaseEmbedding(object):
	def __init__(self, model, device, dataloaders, n_layers):
		self.device = device
		self.model = BaseEmebddingModel(model).to(self.device)
		self.embeddings_dim = n_layers * 768
		self.dataloaders = dataloaders
  
  
	def save_base_embeddings(self, model_name):
		
		print('OBTAINING THE EMBEDDINGS')

		for ds_name, dls in self.dataloaders.items():
			print(f' for {ds_name}')		
   
			for dl_name, dataloader in dls.items():

				save_labels_pt, save_embeddings_pt = False, False

				if not os.path.exists(f'app/embeddings/{model_name}/{ds_name}/labels/{dl_name}_labels.pt'): save_labels_pt = True
				if not os.path.exists(f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.pt'): save_embeddings_pt = True
    
				if not save_labels_pt and not save_embeddings_pt: continue
	
				labels_pt = torch.empty(0, dtype=torch.int8, device=self.device)
				embeddings_tensor = torch.empty((0, self.embeddings_dim), device=self.device)
	
				with torch.inference_mode(): # Allow inference mode
					for idx, (dictionary, labels) in enumerate(dataloader):
      
						if save_labels_pt:
							labels.apply_(lambda x: -1 if x == 0 else x)
							labels = labels.to(self.device)					
       
							labels_pt = torch.cat((labels_pt, labels), dim = 0)


						if save_embeddings_pt:
							for key in list(dictionary.keys()):
								dictionary[key] = dictionary[key].to(self.device)
			
							embeds = self.model(dictionary)

							embeddings_tensor = torch.cat((embeddings_tensor, embeds), dim=0)

							if(idx % 100 == 0):
								if not os.path.exists(f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.pt'):
									torch.save(embeddings_tensor.cpu().detach(), f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.pt')

								else:
									prev_embeddings = torch.load(f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.pt')
									torch.save(torch.cat((prev_embeddings, embeddings_tensor.cpu().detach())), f'app/embeddings/{ds_name}/{dl_name}_embeddings.pt')

								embeddings_tensor = torch.empty((0, self.embeddings_dim)).to(self.device)
							
		
					if save_labels_pt: 
						torch.save(labels_pt, f'app/embeddings/{model_name}/{ds_name}/labels/{dl_name}_labels.pt')
      	
					if save_embeddings_pt and (idx % 100 != 0):
						prev_embeddings = torch.load(f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.pt')
						torch.save(torch.cat((prev_embeddings, embeddings_tensor.cpu().detach())),
                            f'app/embeddings/{model_name}/{ds_name}/{dl_name}_embeddings.pt'
						)

			print('	-> DONE')
