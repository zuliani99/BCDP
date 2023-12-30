from utils import collate_fn
from dataset import CustomTextDataset
from train_evaluate import Train_Evaluate
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


class BertLinearLayer(nn.Module):

	def __init__(self, pretrained_model, n_classes):
		super(BertLinearLayer, self).__init__()
		self.bert = pretrained_model
		self.drop = nn.Dropout(p=0.5)
		self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
	def forward(self, input_ids, attention_mask):
		_, pooled_output = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask
		)

		output = self.drop(pooled_output)
		return self.out(output)


class BertLinears(Train_Evaluate):
	def __init__(self, device, datasets_dict, model, tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, dim_embedding):
		Train_Evaluate.__init__(device, datasets_dict,
						BertLinearLayer(model),
						tokenizer, embedding_split_perc, loss_fn, score_fn,
						patience, epochs, batch_size, dim_embedding)
		

	def run(self):

		for ds_name, dataset in self.datasets_dict.items():
			print(f'--------------- {ds_name} ---------------')
			
			train_ds = CustomTextDataset(dataset['train'], self.tokenizer)
			test_ds = CustomTextDataset(dataset['test'], self.tokenizer)
   
			train_size = len(train_ds)

			val_size = int(train_size * 0.2)
			train_size -= val_size

			train_data, val_data = random_split(train_ds, [int(train_size), int(val_size)])
	
			train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
			val_dl = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
   
			test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
   
			self.fit(ds_name, self.__class__.__name__, train_dl, val_dl)
   
			# we can for eaxample save these metrics to compare with the additional embedding
			#test_accuracy, test_loss = self.test(self.test_dl)
			test_accuracy, test_loss = self.test(test_dl)
   
			# write results
			# write_csv(self.__class__.__name__, ds_name, test_accuracy, test_loss)