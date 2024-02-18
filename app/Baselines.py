
import random
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from utils import read_embbedings, write_csv, accuracy_result


class Baselines(object):
    def __init__(self, common_parmas):
        self.datasets_name = common_parmas['datasets_name']
        self.timestamp = common_parmas['timestamp']
        self.choosen_model_embedding = common_parmas['choosen_model_embedding']
        self.sklearn_baseline = [('SVM', SVC()), ('KNN', KNeighborsClassifier()), ('Naive_Bayes', GaussianNB()), ('Logistic_Regression', LogisticRegression())]
        
        
    def save_results(self, method, ds_name, test_accuracy, elapsed):
        write_csv(
            ts_dir=self.timestamp,
            head = ['method', 'dataset', 'test_accuracy', 'elapsed'],
            values = [method, ds_name, test_accuracy, elapsed],
            categoty_type = 'baselines'
        )    
        
        
    def rand_baseline(self, ds_name):
        print(f'\nRunning Random Baseline')
        
        start = time.time()
        res = [random.choice([-1, 1]) for _ in self.x_test]
        end = time.time()
        
        test_accuracy = accuracy_result(res, self.y_test)
        print(f'Random - test accuracy: {test_accuracy}')
        print(' -> DONE\n')
        self.save_results('Random', ds_name, test_accuracy, end-start)

        
    
    
    def dum_baseline(self, ds_name):
        print(f'\nRunning Dummy Baseline')
        pos, neg = 0, 0
        
        start = time.time()
        for el in self.y_train:
            if el < 0: neg += 1
            else: pos += 1

        if pos > neg: res = np.full(self.x_test.shape[0], 1)
        else: res = np.full(self.x_test.shape[0], -1)
        end = time.time()
        
        test_accuracy = accuracy_result(res, self.y_test)
        print(f'Dummy - test accuracy: {test_accuracy}')
        print(' -> DONE\n')
        self.save_results('Dummy', ds_name, test_accuracy, end-start)
        
        
    

    def run_baselines(self, ds_name):
        
        self.rand_baseline(ds_name)
        
        self.dum_baseline(ds_name)
        
        for name, method in self.sklearn_baseline:
            print(f'\nRunning {name}')
            method.fit(self.x_train, self.y_train)
            
            start = time.time()
            res = method.predict(self.x_test)
            end = time.time()
            
            test_accuracy = accuracy_result(res, self.y_test)
            print(f'{name} - test accuracy: {test_accuracy}')
            print(' -> DONE\n')
            self.save_results(name, ds_name, test_accuracy, end-start)
            
        
    
    def run(self):
        
        print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
        
        for ds_name in self.datasets_name:
            
            print(f'--------------- {ds_name} ---------------')
                
            x_train, x_test, self.y_train, self.y_test = read_embbedings(ds_name, self.choosen_model_embedding)
                
            self.x_train = np.squeeze(x_train[:,-1,:])
            self.x_test = np.squeeze(x_test[:,-1,:])
                
            self.run_baselines(ds_name)
                