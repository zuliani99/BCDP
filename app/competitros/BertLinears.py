
from utils import write_csv
from TrainEvaluate import Train_Evaluate
import torch.nn as nn


class BertLinearLayer(nn.Module):

	def __init__(self, pre_trained_bert, n_classes):
		super(BertLinearLayer, self).__init__()
		self.pre_trained_bert = pre_trained_bert
		self.drop = nn.Dropout(p=0.5)
		self.out = nn.Linear(self.pre_trained_bert.config.hidden_size, n_classes)
		
		#print('hidden_size', self.pre_trained_bert.config.hidden_size)
  
	def forward(self, x):
		output = self.pre_trained_bert(**x, output_hidden_states=True)
  
		#print('output[2][-1][:,0,:] batch x 768', output[2][-1][:,0,:].shape)
		output = output[2][-1][:,0,:]

		output = self.drop(output)
		
		return self.out(output)


class BertLinears(Train_Evaluate):
	def __init__(self, params, dataloaders, timestamp):
		
		self.dataloaders = dataloaders
		self.timestamp = timestamp
  
		params['model'] = BertLinearLayer(params['model'], n_classes=2)
		params['embeddings_dim'] = None
  		
		super().__init__(self.__class__.__name__, params)
		

	def run(self):

		for ds_name, dls in self.dataloaders.items():
      
			print(f'--------------- {ds_name} ---------------')
			
			self.fit(ds_name, self.__class__.__name__, dls['train_dl'], dls['val_dl'])

   
			# we can for eaxample save these metrics to compare with the additional embedding
			#test_accuracy, test_loss = self.test(self.test_dl)
			test_accuracy, test_loss = self.test(['test_dl'])
   
			# write results
			# write_csv(self.__class__.__name__, ds_name, test_accuracy, test_loss)
			write_csv(
                ts_dir=self.timestamp,
                head = ['method', 'dataset', 'test_accuracy', 'test_loss'],
                values = [self.__class__.__name__, ds_name, test_accuracy, test_loss],
                categoty_type='competitors'
            )