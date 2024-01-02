#https://www.kaggle.com/code/hanjoonchoe/nlp-lstm-bert-pytorch

import torch.nn as nn

from TrainEvaluate import Train_Evaluate


class BertLSTMModel(nn.Module):
	def __init__(self, pre_trained_bert, lstm_hidden_size, num_classes, bidirectional):
		super(BertLSTMModel, self).__init__()

		self.pre_trained_bert = pre_trained_bert

		# TODO: FIX THIS
		# NO ATTRIBUTE HIDDEN_SIZE
		self.lstm = nn.LSTM(input_size=self.pre_trained_bert.config.hidden_size,
							hidden_size=lstm_hidden_size,
							batch_first=True,
							bidirectional=bidirectional)
  
		self.dropout = nn.Dropout(0.5)

		self.fc = nn.Linear(lstm_hidden_size, num_classes)

	def forward(self, x):
		
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
  
		print(outputs[2][-1][:,0,:].shape)

		_, (lstm_hidden, _) = self.lstm(outputs[2][-1][:,0,:])
  
		lstm_hidden = self.dropout(lstm_hidden)

		# only take the output from the last time step
		lstm_hidden = lstm_hidden[-1, :, :]

		return self.fc(lstm_hidden)

#output, (hn, cn) = rnn(input, (h0, c0))
		
	

class BertLSTM(Train_Evaluate):
	def __init__(self, device, dataloaders, model, tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, embeddings_dim, bidirectional):
		
		self.dataloaders = dataloaders
  
		Train_Evaluate.__init__(self, 'BertLinears_bi' if bidirectional is True else 'BertLinears', device,
						BertLSTMModel(model, lstm_hidden_size=384, num_classes=2, bidirectional=bidirectional),
						tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, embeddings_dim)
		

	def run(self):

		for ds_name, dls in self.dataloaders.items():
			print(f'--------------- {ds_name} ---------------')
			
			self.fit(ds_name, self.__class__.__name__, dls['train_dl'], dls['val_dl'])

   
			# we can for eaxample save these metrics to compare with the additional embedding
			#test_accuracy, test_loss = self.test(self.test_dl)
			test_accuracy, test_loss = self.test(dls['test_dl'])
   
			# write results
			# write_csv(self.__class__.__name__, ds_name, test_accuracy, test_loss)