#https://www.kaggle.com/code/hanjoonchoe/nlp-lstm-bert-pytorch

import torch.nn as nn

from train_evaluate import Train_Evaluate


class BertLSTMModel(nn.Module):
	def __init__(self, pre_trained_bert, lstm_input_size, lstm_hidden_size, num_classes, bi_directional):
		super(BertLSTMModel, self).__init__()

		self.pre_trained_bert = pre_trained_bert

		# TODO: FIX THIS
		# NO ATTRIBUTE HIDDEN_SIZE
		self.lstm = nn.LSTM(input_size=lstm_input_size,
							hidden_size=lstm_hidden_size,
							batch_first=True,
							bi_directional=bi_directional)
  
		self.dropout = nn.Dropout(0.5)

		self.fc = nn.Linear(lstm_hidden_size, num_classes)

	def forward(self, x):
		
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
  
		print(outputs[2][-1].shape)

		_, (lstm_hidden, _) = self.lstm(outputs[2][-1])
  
		lstm_hidden = self.dropout(lstm_hidden)

		# only take the output from the last time step
		lstm_hidden = lstm_hidden[-1, :, :]

		return self.fc(lstm_hidden)

#output, (hn, cn) = rnn(input, (h0, c0))
		
	

class BertLSTM(Train_Evaluate):
	def __init__(self, device, dataloaders, model, tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, dim_embedding, bi_directional):
		
		self.dataloaders = dataloaders
  
		Train_Evaluate.__init__(self, 'BertLinears_bi' if bi_directional is True else 'BertLinears', device,
						BertLSTMModel(model, lstm_input_size=768,lstm_hidden_size=384, num_classes=2, bi_directional=bi_directional),
						tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, dim_embedding)
		

	def run(self):

		for ds_name, dls in self.dataloaders.items():
			print(f'--------------- {ds_name} ---------------')
			
			self.fit(ds_name, self.__class__.__name__, dls['train_dl'], dls['val_dl'])

   
			# we can for eaxample save these metrics to compare with the additional embedding
			#test_accuracy, test_loss = self.test(self.test_dl)
			test_accuracy, test_loss = self.test(dls['test_dl'])
   
			# write results
			# write_csv(self.__class__.__name__, ds_name, test_accuracy, test_loss)