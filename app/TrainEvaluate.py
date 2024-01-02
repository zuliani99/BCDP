
from tqdm import tqdm
from app.ClusteringEmbeddings import ClusteringEmbeddings
import torch
import os

class Train_Evaluate(ClusteringEmbeddings):
	def __init__(self, name, device, model, tokenizer, embedding_split_perc, loss_fn, score_fn, patience, epochs, batch_size, embeddings_dim):
		ClusteringEmbeddings.__init__(self, name, embedding_split_perc,
                         device, tokenizer, model.to(device), embeddings_dim)
  
		self.batch_size = batch_size
		self.loss_fn = loss_fn
		self.score_fn = score_fn
		self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
		self.patience = patience
		self.epochs = epochs
  
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=3, verbose=True)
  
		self.best_check_filename = 'app/checkpoints'
		self.init_check_filename = 'app/checkpoints/init'#_LA.pth.tar'

		
	def __save_init_checkpoint(self, filename):

		checkpoint = { 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict() }
		torch.save(checkpoint, filename)


	def __save_best_checkpoint(self, filename, actual_patience, epoch, best_val_loss):

		checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                      'actual_patience': actual_patience, 'epoch': epoch, 'best_val_loss': best_val_loss}
		torch.save(checkpoint, filename)



	def __load_checkpoint(self, filename):

		checkpoint = torch.load(filename, map_location=self.device)
		self.model.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.scheduler.load_state_dict(checkpoint['scheduler'])

		return checkpoint['actual_patience'], checkpoint['epoch'], checkpoint['best_val_loss']
  



	def evaluate(self, val_dl, epoch = 0, epochs = 0):
		val_accuracy, val_loss = .0, .0

		self.model.eval()

		pbar = tqdm(val_dl, total = len(val_dl), leave=False)

		with torch.inference_mode(): # Allow inference mode
			for dictionary, labels in pbar:

				for key in list(dictionary.keys()):
					dictionary[key] = dictionary[key].to(self.device)
				labels = labels.to(self.device)
    		
				if self.name == 'LayerAggregation':
					outputs, _ = self.model(dictionary)
				else: 
					outputs = self.model(dictionary)
     				
				accuracy = self.score_fn(outputs, labels)
				loss = self.loss_fn(outputs, labels)

				val_accuracy += accuracy
				val_loss += loss

				if epoch > 0: pbar.set_description(f'EVALUATION Epoch [{epoch} / {epochs}]')
				else: pbar.set_description('TESTING')
				pbar.set_postfix(accuracy = accuracy)

			val_accuracy /= len(val_dl)
			val_loss /= len(val_dl)
   
		return val_accuracy, val_loss
	
	

	def test(self, test_dl):
		test_accuracy, test_loss = self.evaluate(test_dl)

		print('\nTESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f} \n'.format(test_accuracy, test_loss))

		return test_accuracy, test_loss



	def fit(self, ds_name, self_name, train_dl, val_dl):
		
		check_best_path = f'{self.best_check_filename}/{ds_name}_{self_name}.pth.tar'
		
		actual_epoch = 0
		best_val_loss = float('inf')
		actual_patience = 0

		if os.path.exists(check_best_path):
			actual_patience, actual_epoch, best_val_loss = self.__load_checkpoint(check_best_path)
		
		if not os.path.exists(f'{self.init_check_filename}_{self_name}.pth.tar'):
			self.__save_init_checkpoint(f'{self.init_check_filename}_{self_name}.pth.tar')

	
		self.model.train()
		

		for epoch in range(actual_epoch, self.epochs):  # loop over the dataset multiple times			

			train_accuracy, train_loss = 0.0, 0.0

			pbar = tqdm(train_dl, total = len(train_dl), leave=False)

			for dictionary, labels in pbar:
       
				#print(dictionary)
       
				for key in list(dictionary.keys()):
					dictionary[key] = dictionary[key].to(self.device)
				labels = labels.to(self.device)
								
				# zero the parameter gradients
				self.optimizer.zero_grad()
				
				if self.name == 'LayerAggregation':
					outputs, _ = self.model(dictionary)
				else: 
					outputs = self.model(dictionary)
     
				#print(outputs.shape)
				#print(labels.shape)

     
				loss = self.loss_fn(outputs, labels)
				
				loss.backward()
				self.optimizer.step()

				accuracy = self.score_fn(outputs, labels)

				train_accuracy += accuracy
				train_loss += loss

				# Update the progress bar
				pbar.set_description(f'TRAIN Epoch [{epoch + 1} / {self.epochs}]')
				pbar.set_postfix(accuracy = accuracy, loss = loss.item())
	

			train_accuracy /= len(train_dl)
			train_loss /= len(train_dl)
   
			# scheduler
			self.scheduler.step(train_loss)
   

			# Validation step
			val_accuracy, val_loss = self.evaluate(val_dl, epoch + 1, self.epochs)

			print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
				epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss))


			if(val_loss < best_val_loss):
				best_val_loss = val_accuracy
				actual_patience = 0
				self.__save_best_checkpoint(check_best_path, actual_patience, epoch, best_val_loss)
			else:
				actual_patience += 1
				if actual_patience >= self.patience:
					print(f'Early stopping, validation accuracy do not decreased for {self.patience} epochs')
					pbar.close() # Closing the progress bar before exiting from the train loop
					break
								

		self.__load_checkpoint(check_best_path)

		print('Finished Training\n')
  