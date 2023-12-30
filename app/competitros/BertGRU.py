#https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb

import torch.nn as nn

from train_evaluate import Train_Evaluate


class BertGRUModel(nn.Module):
    def __init__(self, pre_trained_bert, gru_hidden_size, num_classes):
        super(BertGRUModel, self).__init__()

        self.pre_trained_bert = pre_trained_bert

        self.gru = nn.GRU(input_size=pre_trained_bert.hidden_size,
                          hidden_size=gru_hidden_size,
                          batch_first=True)

        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x):
        
        outputs = self.pre_trained_bert(**x, output_hidden_states=True)
        bert_output = outputs.last_hidden_state

        lstm_output, _ = self.lstm(bert_output)

        # only take the output from the last time step
        lstm_output = lstm_output[:, -1, :]

        return self.fc(lstm_output)
        
    

class BertGRU(Train_Evaluate):
	def __init__(self, device, datasets_dict, model, tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, dim_embedding, bi_directional):
		Train_Evaluate.__init__(self, 'BertLinears', device, datasets_dict,
						BertGRUModel(model, 512, num_classes=2),
						tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, dim_embedding)
		

	def run(self):

		for ds_name, dls in self.dataloaders.items():
			print(f'--------------- {ds_name} ---------------')
			
			self.fit(ds_name, self.__class__.__name__, dls['train_dl'], dls['val_dl'])

   
			# we can for eaxample save these metrics to compare with the additional embedding
			#test_accuracy, test_loss = self.test(self.test_dl)
			test_accuracy, test_loss = self.test(['test_dl'])
   
			# write results
			# write_csv(self.__class__.__name__, ds_name, test_accuracy, test_loss)