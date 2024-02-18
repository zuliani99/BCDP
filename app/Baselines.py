
import random
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

from utils import read_embbedings, write_csv, accuracy_result


class Baselines(object):
    def __init__(self, common_parmas):
        self.datasets_name = common_parmas['datasets_name']
        self.timestamp = common_parmas['timestamp']
        self.choosen_model_embedding = common_parmas['choosen_model_embedding']
        self.sklearn_baseline = [('SVM', SVC()), ('KNN', KNeighborsClassifier()), ('Naive_Bayes', GaussianNB()), ('Logistic_Regression', LogisticRegression())]
        
        
    def save_results_stats(self, method, ds_name, report, elapsed):
        test_accuracy, neg_f1, pos_f1 = report['accuracy'], report['-1']['f1-score'], report['1']['f1-score']
        write_csv(
            ts_dir=self.timestamp,
            head = ['method', 'dataset', 'test_accuracy', 'negative_f1', 'positive_f1', 'elapsed'],
            values = [method, ds_name, test_accuracy, neg_f1, pos_f1, elapsed],
            categoty_type = 'baselines'
        )
        
        
    def rand_baseline(self, ds_name):
        print(f'\nRunning Random Baseline')
        
        start = time.time()
        res = [random.choice([-1, 1]) for _ in self.x_test]
        end = time.time()
        
        report = classification_report(self.y_test, res, output_dict=True, zero_division=0)
        print(f'Random - test accuracy: {report["accuracy"]}')
        print(' -> DONE\n')
        self.save_results_stats('Random', ds_name, report, end-start)

        
    
    
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
        
        report = classification_report(self.y_test, res, output_dict=True, zero_division=0)
        print(f'Dummy - test accuracy: {report["accuracy"]}')
        print(' -> DONE\n')
        self.save_results_stats('Dummy', ds_name, report, end-start)
        
        
    

    def run_baselines(self, ds_name):
        
        self.rand_baseline(ds_name)
        
        self.dum_baseline(ds_name)
        
        for name, method in self.sklearn_baseline:
            print(f'\nRunning {name}')
            method.fit(self.x_train, self.y_train)
            
            start = time.time()
            res = method.predict(self.x_test)
            end = time.time()
            
            report = classification_report(self.y_test, res, output_dict=True, zero_division=0)
            print(f'{name} - test accuracy: {report["accuracy"]}')
            print(' -> DONE\n')
            self.save_results_stats(name, ds_name, report, end-start)
            
        
    
    def run(self):
        
        print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
        
        for ds_name in self.datasets_name:
            
            print(f'--------------- {ds_name} ---------------')
                
            x_train, x_test, self.y_train, self.y_test = read_embbedings(ds_name, self.choosen_model_embedding)
                
            self.x_train = np.squeeze(np.copy(x_train[:,-1,:]))
            self.x_test = np.squeeze(np.copy(x_test[:,-1,:]))
                
            self.run_baselines(ds_name)
                