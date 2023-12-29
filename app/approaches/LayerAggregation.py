
from tqdm import tqdm
from dataset import CustomTextDataset

from utils import collate_fn, get_embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split



class SelfAttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, attention_heads):
        super(SelfAttentionLayer, self).__init__()

        self.input_size = input_size
        self.attention_heads = attention_heads

        # Linear transformations for Query, Key, and Value
        self.W_q = nn.Linear(input_size, input_size)
        self.W_k = nn.Linear(input_size, input_size)
        self.W_v = nn.Linear(input_size, input_size)

        # Linear transformation for the output of attention heads
        self.W_o = nn.Linear(input_size, output_size)


    def forward(self, x):
        # Linear transformations for Query, Key, and Value
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple attention heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_size, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)

        # Concatenate attention heads
        attended_values = self.concat_heads(attended_values)

        # Linear transformation for the output
        outputs = self.W_o(attended_values)

        return outputs, attended_values


    def split_heads(self, x):
        batch_size, features = x.size()
        head_size = features // self.attention_heads

        x = x.view(batch_size, self.attention_heads, head_size)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size * self.attention_heads, head_size)

        return x


    def concat_heads(self, x):
        batch_size_heads, head_size = x.size()
        batch_size = batch_size_heads // self.attention_heads

        x = x.view(batch_size, self.attention_heads, head_size)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size, self.attention_heads * head_size)

        return x
	
	


class Bert_Layer_aggregation(nn.Module):
	def __init__(self, bert, batch_size):
		super(Bert_Layer_aggregation, self).__init__()
		self.batch_size = batch_size
		self.pre_trained_bert = bert
		self.self_attention_layer = SelfAttentionLayer(12 * 768, output_size=2, attention_heads=8)
		self.freeze_layers()
		
		
	def forward(self, x):
		outputs = self.pre_trained_bert(**x, output_hidden_states=True)
		hidden_states_batches = outputs[2]
		aggregated_tensor = torch.cat([h_state[:,0,:] for h_state in hidden_states_batches[1:]], dim=1)
		outputs, attentions = self.self_attention_layer(aggregated_tensor)
		return outputs, attentions 
		

	def freeze_layers(self):
		for param in self.pre_trained_bert.parameters():
			param.requires_grad = False
		



class LayerAggregation():
	def __init__(self, device, datasets_dict, model, tokenizer, embedding_split_perc, loss_fn, score_fn, patience, epochs, batch_size):
		self.device = device
		self.batch_size = batch_size
		self.datasets_dict = datasets_dict
		self.model = Bert_Layer_aggregation(model, self.batch_size).to(device)
		self.tokenizer = tokenizer
		self.embedding_split_perc = embedding_split_perc
		self.loss_fn = loss_fn
		self.score_fn = score_fn
		self.optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)
		self.patience = patience
		self.epochs = epochs
		self.embedding_dim = 768 * 12
  
		self.best_check_filename = 'app/chekpoint/layer_aggregation.pth.tar'
	
		

	def __save_checkpoint(self, filename):

		checkpoint = { 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict() }
		torch.save(checkpoint, filename)



	def __load_checkpoint(self, filename):

		checkpoint = torch.load(filename, map_location=self.device)
		self.model.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])



	def evaluate(self, val_dl, epoch = 0, epochs = 0):
		val_accuracy, val_loss = .0, .0

		self.model.eval()

		pbar = tqdm(val_dl, total = len(val_dl), leave=False)

		with torch.inference_mode(): # Allow inference mode
			for texts, labels in pbar:
				texts, label = texts.to(self.device), labels.to(self.device)
    
				texts = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to(self.device)
    
				outputs, _ = self.model(texts)
				
				accuracy = self.score_fn(outputs, label)
				loss = self.loss_fn(outputs, labels)

				val_accuracy += accuracy
				val_loss += loss

				if epoch > 0: pbar.set_description(f'EVALUATION Epoch [{epoch} / {epochs}]')
				else: pbar.set_description('TESTING')
				pbar.set_postfix(accuracy = accuracy)

			val_accuracy /= len(val_dl)
			val_loss /= len(val_dl)
   
		return val_accuracy, val_loss


	def fit(self):
		self.model.train()
		
		best_val_loss = float('inf')
		actual_patience = 0

		for epoch in range(self.epochs):  # loop over the dataset multiple times			

			train_accuracy, train_loss = 0.0, 0.0

			pbar = tqdm(self.train_dl, total = len(self.train_dl), leave=False)

			for dictionary, labels in pbar:
								
				# zero the parameter gradients
				self.optimizer.zero_grad()
    				
				outputs, _ = self.model(dictionary)

				loss = self.loss_fn(outputs, labels)
				
				loss.backward()
				self.optimizer.step()

				accuracy = self.score_fn(outputs, labels)

				train_accuracy += accuracy
				train_loss += loss

				# Update the progress bar
				pbar.set_description(f'TRAIN Epoch [{epoch + 1} / {self.epochs}]')
				pbar.set_postfix(accuracy = accuracy, loss = loss.item())
	

			train_accuracy /= len(self.train_dl)
			train_loss /= len(self.train_dl)
   

			# Validation step
			val_accuracy, val_loss = self.evaluate(self.val_dl, epoch + 1, self.epochs)

			print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
				epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss))


			if(val_loss < best_val_loss):
				best_val_loss = val_accuracy
				actual_patience = 0
				self.__save_checkpoint(self.best_check_filename)
			else:
				actual_patience += 1
				if actual_patience >= self.patience:
					print(f'Early stopping, validation accuracy do not decreased for {self.patience} epochs')
					pbar.close() # Closing the progress bar before exiting from the train loop
					break
								

		self.__load_checkpoint(self.best_check_filename)

		print('Finished Training\n')


	def run(self):
		for ds_name, dataset in self.datasets_dict.items():
			print(f'--------------- {ds_name} ---------------')
			'''max_word = 0
			for text in dataset['train']['text']:
				s = text.split(' ')
				if len(s) > max_word: 
					print(max_word)
					max_word = len(s)

			print(max_word)'''
			
			train_ds = CustomTextDataset(dataset['train'], self.tokenizer)
			test_ds = CustomTextDataset(dataset['test'], self.tokenizer)
   
			train_size = len(train_ds)

			val_size = int(train_size * 0.2)
			train_size -= val_size

			train_data, val_data = random_split(train_ds, [int(train_size), int(val_size)])
    
			self.train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
			self.val_dl = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
   
			self.test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
   
			self.fit()
			self.evaluate(self.test_dl)
			get_embeddings(self)
   
			
			# run clusering
  