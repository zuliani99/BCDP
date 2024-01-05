#https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb

import torch.nn as nn
from utils import write_csv

from TrainEvaluate import Train_Evaluate


class BertGRUModel(nn.Module):
	def __init__(self, pre_trained_bert, gru_hidden_size, num_classes):
		super(BertGRUModel, self).__init__()

		self.pre_trained_bert = pre_trained_bert

		# TODO: FIX THIS
		# NO ATTRIBUTE HIDDEN_SIZE
		self.gru = nn.GRU(input_size=self.pre_trained_bert.config.hidden_size,
						  hidden_size=gru_hidden_size,
						  batch_first=True)
  
		self.dropout = nn.Dropout(0.5)
  
		self.fc = nn.Linear(gru_hidden_size, num_classes)

	def forward(self, x):
		
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)#.last_hidden_state[:, 0, :]
		#bert_output = outputs.last_hidden_state
  
		print(outputs[2][-1][:,0,:].shape)

		# cls embedding of each last batch last layer
		_, gru_hidden = self.gru(outputs[2][-1][:,0,:])
		#hidden = [batch size, hid dim]
  
		gru_hidden = self.dropout(gru_hidden)

		#output = [batch size, out dim]
		gru_hidden = gru_hidden[-1,: , :]

		return self.fc(gru_hidden)
		
	

class BertGRU(Train_Evaluate):
	def __init__(self, params, dataloaders, timestamp):
		
		self.dataloaders = dataloaders
		self.timestamp = timestamp
  
		params['model'] = BertGRUModel(params['model'], gru_hidden_size=384, num_classes=2)
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