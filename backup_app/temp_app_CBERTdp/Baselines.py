
import random
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from utils import read_embbedings, write_csv, accuracy_result


class Baselines(object):
    def __init__(self, datasets_name, timestamp, approaches_names):
        self.datasets_name = datasets_name
        self.timestamp = timestamp
        self.approaches_names = approaches_names
        self.sklearn_baseline = [('SVM', SVC()), ('KNN', KNeighborsClassifier()), ('Naive_Bayes', GaussianNB()), ('Logistic_Regression', LogisticRegression())]
        
        
    def save_results(self, method, embeddings_from, ds_name, test_accuracy):
        write_csv(
            ts_dir=self.timestamp,
            head = ['method', 'embedding_from', 'dataset', 'test_accuracy'],
            values = [method, embeddings_from, ds_name, test_accuracy],
            categoty_type='baselines'
        )    
        
        
    def rand_baseline(self, embeddings_from, ds_name):
        print(f'\nRunning Random Baseline')
        res = [random.choice([-1, 1]) for _ in self.x_test]
        test_accuracy = accuracy_result(res, self.y_test)
        print(f'Random - test accuracy: {test_accuracy}')
        print(' -> DONE\n')
        self.save_results('Random', embeddings_from, ds_name, test_accuracy)

        
    
    
    def dum_baseline(self, embeddings_from, ds_name):
        print(f'\nRunning Dummy Baseline')
        pos, neg = 0, 0
        for el in self.y_train:
            if el < 0: neg += 1
            else: pos += 1

        if pos > neg: res = np.full(self.x_test.shape[0], 1)
        else: res = np.full(self.x_test.shape[0], -1)
        test_accuracy = accuracy_result(res, self.y_test)
        print(f'Dummy - test accuracy: {test_accuracy}')
        print(' -> DONE\n')
        self.save_results('Dummy', embeddings_from, ds_name, test_accuracy)
        
        
    

    def run_baselines(self, embeddings_from, ds_name):
        
        self.rand_baseline(embeddings_from, ds_name)
        
        self.dum_baseline(embeddings_from, ds_name)
        
        for name, method in self.sklearn_baseline:
            print(f'\nRunning {name}')
            method.fit(self.x_train, self.y_train)
            res = method.predict(self.x_test)
            test_accuracy = accuracy_result(res, self.y_test)
            print(f'{name} - test accuracy: {test_accuracy}')
            print(' -> DONE\n')
            self.save_results(name, embeddings_from, ds_name, test_accuracy)
            
        
            
            
    
    def run(self):
        
        print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
        
        for ds_name in self.datasets_name:
            
            print(f'--------------- {ds_name} ---------------')
            
            for embeddings_from in self.approaches_names:
                
                print(f'------- EMBEDDINGS FROM {embeddings_from} -------')
                
                self.x_train, self.x_test, self.y_train, self.y_test = read_embbedings(ds_name, embeddings_from)

                self.run_baselines(embeddings_from, ds_name)
                