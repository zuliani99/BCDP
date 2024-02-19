
import torch
import os

from utils import init_params

class Train_Evaluate(object):
    def __init__(self, name, params, base_embeds_model, model):
  
        self.batch_size = params['batch_size']
        self.loss_fn = params['loss_fn']
        self.score_fn = params['score_fn']
        self.patience = params['patience']
        self.epochs = params['epochs']
        self.device = params['device']
        self.model = model.to(self.device)
        self.name = name
        self.base_embeds_model = base_embeds_model
  
        self.check_filename = f'app/checkpoints/{base_embeds_model}'
        self.init_check_filename = f'{self.check_filename}/init/init_{self.name}.pth.tar'
        
        self.model.apply(init_params)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        self.__save_init_checkpoint()
        
    def __save_init_checkpoint(self):
        if not os.path.exists(self.init_check_filename):
            print(f' => Saving initial {self.name} checkpoint')
            checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            torch.save(checkpoint, self.init_check_filename)
            print(' DONE\n')
        else:
            print(f' => Initial {self.name} checkpoint already present')

    
    def load_initial_checkpoint(self):
        print(f' => Loading initial {self.name} checkpoint')
        checkpoint = torch.load(self.init_check_filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(' DONE\n')


    def __save_best_checkpoint(self, filename, actual_patience, epoch, best_val_loss):
        """ Save the best model checkpoint along with relevant training information.

		@param filename: str, the filename to which the checkpoint should be saved
		@param actual_patience: int, the current patience value during training
		@param epoch: int, the current epoch number
		@param best_val_loss: float, best validation loss achieved during training

		@Return: None
		"""
        print(f' => Saving best {self.name} checkpoint')
        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                      'actual_patience': actual_patience, 'epoch': epoch, 'best_val_loss': best_val_loss}
        torch.save(checkpoint, filename)
        print(' DONE\n')



    def __load_best_checkpoint(self, filename):
        """ Load the best model checkpoint from a specified file 

		@param filename: str, the filename from which to load the model checkpoint

		@Return: tuple [int, int, float] containing the values for actual_patience, epoch and best_val_loss from the loaded checkpoint

		"""

        print(f' => Loading {self.name} best checkpoint')
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(' DONE\n')
        
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
                else: outputs = torch.squeeze(self.model(bert_ebmbeds))
                     
                accuracy = self.score_fn(outputs, labels)
                loss = self.loss_fn(outputs, labels)

                val_accuracy += accuracy
                val_loss += loss.item()

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



    def fit(self, dataset_name, train_dl, val_dl):
        
        check_best_path = f'{self.check_filename}/{dataset_name}_{self.name}.pth.tar'
        
        actual_epoch = 0
        best_val_loss = float('inf')
        actual_patience = 0

        if os.path.exists(check_best_path):
            actual_patience, actual_epoch, best_val_loss = self.__load_best_checkpoint(check_best_path)
        
        
        if actual_epoch + 1 == self.epochs:
            print('Already Finished Training\n')
            return 
  
        for epoch in range(actual_epoch, self.epochs):  # loop over the dataset multiple times	
      
            self.model.train()		

            train_accuracy, train_loss = 0.0, 0.0


            for bert_ebmbeds, labels in train_dl:
                                
                bert_ebmbeds, labels = bert_ebmbeds.to(self.device), labels.to(self.device)
                                                
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                if self.name == 'LayerAggregation': outputs, _ = self.model(bert_ebmbeds)
                else: outputs = torch.squeeze(self.model(bert_ebmbeds))
                
                loss = self.loss_fn(outputs, labels)
                
                loss.backward()
                self.optimizer.step()

                accuracy = self.score_fn(outputs, labels)

                train_accuracy += accuracy
                train_loss += loss.item()

    

            train_accuracy /= len(train_dl)
            train_loss /= len(train_dl)
   

            # Validation step
            val_accuracy, val_loss = self.evaluate(val_dl)

            print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss))


            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                actual_patience = 0
                self.__save_best_checkpoint(check_best_path, actual_patience, epoch, best_val_loss)
            else:
                actual_patience += 1
                if actual_patience >= self.patience:
                    print(f'Early stopping, validation accuracy do not decreased for {self.patience} epochs')
                    break
                                

        self.__load_best_checkpoint(check_best_path)

        print('Finished Training\n')
  