
from utils import get_competitors_embeddings_dls, write_csv
from TrainEvaluate import Train_Evaluate
import time

import torch
import torch.nn as nn


class BertLinearLayer(nn.Module):

	def __init__(self, pt_hidden_size, n_classes):
		super(BertLinearLayer, self).__init__()
		self.drop = nn.Dropout(p=0.5)
		self.out = nn.Linear(pt_hidden_size, n_classes)
  
	def forward(self, x):
		return self.out(self.drop(x))



class BertLinears(Train_Evaluate):
	def __init__(self, params, common_parmas, pt_hidden_size):
     
		super().__init__(self.__class__.__name__, params, BertLinearLayer(pt_hidden_size, n_classes=2))
		
		self.datasets_name = common_parmas['datasets_name']
		self.timestamp = common_parmas['timestamp']
		self.choosen_model_embedding = common_parmas['choosen_model_embedding']
		

	def run(self):
	
		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    

		for ds_name in self.datasets_name:
	 
			print(f'--------------- {ds_name} ---------------')
   
			self.load_initial_checkpoint()
   
			train_dl, val_dl, test_dl = get_competitors_embeddings_dls(ds_name, self.choosen_model_embedding, self.batch_size)
   
			self.fit(ds_name, train_dl, val_dl)

			start = time.time()
			test_accuracy, test_loss = self.test(test_dl)
			end = time.time()
   
			# write results
			write_csv(
				ts_dir=self.timestamp,
				head = ['method', 'dataset', 'test_accuracy', 'test_loss', 'elapsed'],
				values = [self.__class__.__name__, ds_name, test_accuracy, test_loss, end-start],
				categoty_type = 'competitors'
			)
   
			print('------------------------------------------\n')
   
		print(f'\n---------------------------------- END {self.__class__.__name__ } ----------------------------------\n\n')
   
   





class BertLSTMGRUModel(nn.Module):
	def __init__(self, hidden_size, num_classes, bidirectional, lstm_gru):
		super(BertLSTMGRUModel, self).__init__()
		
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional
		self.lstm_gru = lstm_gru
		self.dropout = nn.Dropout(0.5)
		self.fc1 = nn.Linear(
      					self.hidden_size * 2 if self.bidirectional else self.hidden_size, 
                       	self.hidden_size if self.bidirectional else num_classes
                    )
		self.fc2 = nn.Linear(self.hidden_size, num_classes)
		self.sigmoid = nn.Sigmoid()
 
 
	
	def forward(self, x):
		batch_size = x.shape[0]
		hidden = self.init_hidden(batch_size)
  
		#print(x.shape, hidden.shape)

		output, _ = self.lstm_gru(x, hidden)

		if self.bidirectional:
			out_fwd = output[:, -1, :(self.hidden_size)]
			out_bwd = output[:, 0, (self.hidden_size):]
			output = torch.cat((out_fwd, out_bwd), 1)
			output = self.dropout(output)
			output = self.fc1(output)
			output = self.fc2(output)
		else:
			output = self.dropout(output)
			output = self.fc1(output[:,-1,:])

		output = self.sigmoid(output)
		return output
        
        
	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.lstm_gru.__class__.__name__ == 'LSTM':
			hidden = (weight.new(2 if self.bidirectional else 1, batch_size, self.hidden_size).zero_(),
					weight.new(2 if self.bidirectional else 1, batch_size, self.hidden_size).zero_())
		else:
			hidden = weight.new(2 if self.bidirectional else 1, batch_size, self.hidden_size).zero_()
		return hidden


class BertLSTMGRU(Train_Evaluate):
	def __init__(self, params, common_parmas, pt_hidden_size, lstm_gru, bidirectional=False):
          
		super().__init__(f'Bert_bi{lstm_gru}' if bidirectional is True else f'Bert_{lstm_gru}', params, BertLSTMGRUModel(
      						pt_hidden_size, num_classes=2, bidirectional=bidirectional,
                            lstm_gru = 
                            	nn.LSTM(
									input_size=pt_hidden_size, hidden_size=pt_hidden_size,
									batch_first=True, bidirectional=bidirectional
								) if lstm_gru == 'LSTM' else
                           	 	nn.GRU(
                                  	input_size=pt_hidden_size, hidden_size=pt_hidden_size,
									batch_first=True, bidirectional=bidirectional
								)  
                        	)
                   		)
  
		self.custom_name = f'Bert_bi{lstm_gru}' if bidirectional is True else f'Bert_{lstm_gru}'
		self.datasets_name = common_parmas['datasets_name']
		self.timestamp = common_parmas['timestamp']
		self.choosen_model_embedding = common_parmas['choosen_model_embedding']
	

		
	def run(self):
	
		print(f'---------------------------------- START {self.custom_name} ----------------------------------')	    
	
		for ds_name in self.datasets_name:
	 
			print(f'--------------- {ds_name} ---------------')
   
			self.load_initial_checkpoint()
			
			train_dl, val_dl, test_dl = get_competitors_embeddings_dls(ds_name, self.choosen_model_embedding, self.batch_size)

			self.fit(ds_name, train_dl, val_dl)
   
			start = time.time()
			test_accuracy, test_loss = self.test(test_dl)
			end = time.time()
   
			# write results
			write_csv(
				ts_dir=self.timestamp,
				head = ['method', 'dataset', 'test_accuracy', 'test_loss', 'elapsed'],
				values = [self.custom_name, ds_name, test_accuracy, test_loss, end-start],
				categoty_type = 'competitors'
			)

			print('------------------------------------------\n')

   
		print(f'\n---------------------------------- END {self.custom_name} ----------------------------------\n\n')
   