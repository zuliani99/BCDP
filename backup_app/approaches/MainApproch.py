
#from ClusteringEmbeddings import ClusteringEmbeddings
#import torch.nn as nn
from utils import read_embbedings_pt
'''from utils import read_embbedings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np'''


'''class BertLastLayer(nn.Module):
	def __init__(self, bert):
		super(BertLastLayer, self).__init__()
		self.pre_trained_bert = bert
		
	def forward(self, x):
		return self.pre_trained_bert(**x, output_hidden_states=True).last_hidden_state[:, 0, :]
		'''
	

class MainApproch(object): #(ClusteringEmbeddings):
	def __init__(self, datasets_name, timestamp):
		'''ClusteringEmbeddings.__init__(self, self.__class__.__name__, embedding_split_perc,
                         device, BertLastLayer(model).to(device),
                         embeddings_dim = 768)'''
		
		self.datasets_name = datasets_name
		self.timestamp = timestamp

  
	def run(self):
	
		print(f'---------------------------------- START {self.__class__.__name__} ----------------------------------')	    
     
		#colors = ['red', 'green']
     
		for ds_name in self.datasets_name:

			print(f'--------------- {ds_name} ---------------')

			#self.get_embeddings(ds_name, dls)
			x_train, x_test, y_train, y_test = read_embbedings_pt(ds_name)
			x_train = x_train[-1][:, 0, :]
			x_test = x_test[-1][:, 0, :]
   
			# run clusering
			#self.faiss_clusering.run_faiss_kmeans(ds_name, self.__class__.__name__, self.timestamp)

		print(f'\n---------------------------------- END {self.__class__.__name__ } ----------------------------------\n\n')


'''x_train, x_test, y_train, y_test = read_embbedings(ds_name, self.__class__.__name__)
   
			tsne = TSNE().fit_transform(x_train)
			x = tsne[:,0]
			y = tsne[:,1]
			plt.figure(figsize=(16,10))
			for i in range(len(tsne)):
				plt.scatter(x[i], y[i], marker='o', color=colors[1 if y_train[i] == 1 else 0])
			plt.title(f'Train TSNE {ds_name}')
			plt.savefig(f'results/tsne_embedding_{ds_name}_{self.__class__.__name__}.png')
   
   
			pca = PCA().fit_transform(x_train)
			x = pca[:,0]
			y = pca[:,1]
			plt.figure(figsize=(16,10))
			for i in range(len(pca)):
				plt.scatter(x[i], y[i], marker='o', color=colors[1 if y_train[i] == 1 else 0])
			plt.title(f'Train PCA {ds_name}')
			plt.savefig(f'results/pca_embedding_{ds_name}_{self.__class__.__name__}.png')'''