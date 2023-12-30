#https://www.kaggle.com/code/hanjoonchoe/nlp-lstm-bert-pytorch

import torch.nn as nn

from train_evaluate import Train_Evaluate


class BertLSTMModel(nn.Module):
	def __init__(self, pre_trained_bert, lstm_hidden_size, num_classes, bi_directional):
		super(BertLSTMModel, self).__init__()

		self.pre_trained_bert = pre_trained_bert

		self.lstm = nn.LSTM(input_size=pre_trained_bert.hidden_size,
							hidden_size=lstm_hidden_size,
							batch_first=True,
							bi_directional=bi_directional)

		self.fc = nn.Linear(lstm_hidden_size, num_classes)

	def forward(self, x):
		
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		bert_output = outputs.last_hidden_state

		lstm_output, _ = self.lstm(bert_output)

		# only take the output from the last time step
		lstm_output = lstm_output[:, -1, :]

		return self.fc(lstm_output)
		
	

class BertLSTM(Train_Evaluate):
	def __init__(self, device, dataloaders, model, tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, dim_embedding, bi_directional):
		
		self.dataloaders = dataloaders
  
		Train_Evaluate.__init__(self, 'BertLinears_bi' if bi_directional is True else 'BertLinears', device,
						BertLSTMModel(model, 512, num_classes=2, bi_directional=bi_directional),
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