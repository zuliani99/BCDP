
from utils import read_embbedings, write_csv


class Baselines(object):
    def __init__(self, datasets_name, timestamp, approaches_names):
        self.datasets_name = datasets_name
        self.timestamp = timestamp
        self.approaches_names = approaches_names
        
        
    def save_results(self, method, embeddings_from, ds_name):
        write_csv(
            ts_dir=self.timestamp,
            head = ['method', 'embedding_from', 'dataset'],#, 'test_accuracy'],
            values = [method, embeddings_from, ds_name],#, test_accuracy],
            categoty_type='baselines'
        )    
    
        
    def run_svm(self, embeddings_from, ds_name):
        print('\nRunning SVM')
        
        print(' -> DONE\n')
        #self.save_results()
    
        
        
    def run_knn(self, embeddings_from, ds_name):
        print('\nRunning SVM')
        
        print(' -> DONE\n')
        #self.save_results()
        
        
        
    def run_naive_bayes(self, embeddings_from, ds_name):
        print('\nRunning SVM')
        
        print(' -> DONE\n')
        #self.save_results()
        
        
        
    def run_logistic_regression(self, embeddings_from, ds_name):
        print('\nRunning SVM')
        
        print(' -> DONE\n')
        #self.save_results()
        
        
        
    def run_baselines(self, embeddings_from, ds_name):
        # run SVM
        self.run_svm(embeddings_from, ds_name)
                
        # run NaiveBayes
        self.run_naive_bayes(embeddings_from, ds_name)
                
        # run LogisticRegression
        self.run_logistic_regression(embeddings_from, ds_name)
                
        # run Knn
        self.run_knn(embeddings_from, ds_name)
        
    
    def run(self):
        
        print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
        
        for ds_name in self.datasets_name:
            
            print(f'--------------- {ds_name} ---------------')
            
            for embeddings_from in self.approaches_names:
                
                print(f'------- EMBEDDINGS FROM {embeddings_from} -------')
                
                self.x_train, self.x_test, self.y_train, self.y_test = read_embbedings(ds_name, embeddings_from)

                self.run_baselines(embeddings_from, ds_name)