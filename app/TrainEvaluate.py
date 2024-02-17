
import torch
import os

from utils import init_params

class Train_Evaluate(object):
    def __init__(self, name, params, model):
  
        self.batch_size = params['batch_size']
        self.loss_fn = params['loss_fn']
        self.score_fn = params['score_fn']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.patience = params['patience']
        self.epochs = params['epochs']
        self.device = params['device']
        self.model = model.to(self.device)
        self.name = name
  
        self.best_check_filename = 'app/checkpoints'
        self.init_check_filename = 'app/checkpoints/init'
        
        self.model.apply(init_params)



    def __save_best_checkpoint(self, filename, actual_patience, epoch, best_val_loss):
        """ Save the best model checkpoint along with relevant training information.

		@param filename: str, the filename to which the checkpoint should be saved
		@param actual_patience: int, the current patience value during training
		@param epoch: int, the current epoch number
		@param best_val_loss: float, best validation loss achieved during training

		@Return: None
		"""

        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), #'scheduler': self.scheduler.state_dict(),
                      'actual_patience': actual_patience, 'epoch': epoch, 'best_val_loss': best_val_loss}
        torch.save(checkpoint, filename)



    def __load_checkpoint(self, filename):
        """ Load a model checkpoint from a specified file 

		@param filename: str, the filename from which to load the model checkpoint

		@Return: tuple [int, int, float] containing the values for actual_patience, epoch and best_val_loss from the loaded checkpoint

		"""

        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['actual_patience'], checkpoint['epoch'], checkpoint['best_val_loss']
  



    def evaluate(self, val_dl):
        """Evaluate the model's performance on a validation dataset.
		@param val_dl: the data loader for the validation dataset
		@param epoch: int, the current epoch number, default is 0
		@param epochs: int, the total number of training epochs, default is 0

		@Return: a tuple containing the computed validation accuracy and loss

		"""
  
        val_accuracy, val_loss = .0, .0

        self.model.eval()

        with torch.inference_mode(): # Allow inference mode
            for bert_ebmbeds, labels in val_dl:

                bert_ebmbeds, labels = bert_ebmbeds.to(self.device), labels.to(self.device)
            
                if self.name == 'LayerAggregation': outputs, _ = self.model(bert_ebmbeds)
                else: outputs = self.model(bert_ebmbeds)
                     
                accuracy = self.score_fn(outputs, labels)
                loss = self.loss_fn(outputs, labels)

                val_accuracy += accuracy
                val_loss += loss

            val_accuracy /= len(val_dl)
            val_loss /= len(val_dl)
   
        return val_accuracy, val_loss
    
    

    def test(self, test_dl):
        """ Evaluate the model's performance on a test dataset and print the results.
		@param test_dl: the data loader for the test dataset

		@Return: a tuple containing the computed test accuracy and loss
		
		"""
  
        test_accuracy, test_loss = self.evaluate(test_dl)

        print('\nTESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f} \n'.format(test_accuracy, test_loss))

        return test_accuracy, test_loss



    def fit(self, model_name, train_dl, val_dl):
        
        check_best_path = f'{self.best_check_filename}/{model_name}_{self.name}.pth.tar'
        
        actual_epoch = 0
        best_val_loss = float('inf')
        actual_patience = 0

        if os.path.exists(check_best_path):
            actual_patience, actual_epoch, best_val_loss = self.__load_checkpoint(check_best_path)
        
        #if not os.path.exists(f'{self.init_check_filename}_{self_name}.pth.tar'):
        #	self.__save_init_checkpoint(f'{self.init_check_filename}_{self_name}.pth.tar')

        
        if actual_epoch + 1 == self.epochs: return 
  
        for epoch in range(actual_epoch, self.epochs):  # loop over the dataset multiple times	
      
            self.model.train()		

            train_accuracy, train_loss = 0.0, 0.0


            for bert_ebmbeds, labels in train_dl:

                bert_ebmbeds, labels = bert_ebmbeds.to(self.device), labels.to(self.device)
                                
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                if self.name == 'LayerAggregation': outputs, _ = self.model(bert_ebmbeds)
                else: outputs = self.model(bert_ebmbeds)

                loss = self.loss_fn(outputs, labels)
                
                loss.backward()
                self.optimizer.step()

                accuracy = self.score_fn(outputs, labels)

                train_accuracy += accuracy
                train_loss += loss

    

            train_accuracy /= len(train_dl)
            train_loss /= len(train_dl)
   

            # Validation step
            val_accuracy, val_loss = self.evaluate(val_dl)

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
                    break
                                

        self.__load_checkpoint(check_best_path)

        print('Finished Training\n')
  