
import numpy as np
#from sklearn.model_selection import train_test_split
import faiss
from tqdm.auto import tqdm

from utils import accuracy_result, read_embbedings, write_csv

class FaissClustering():
    def __init__(self):
        self.n_clusters_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.top_k_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.size_split = 0.2



    # standard_k_means: (k-means clustering algorithm)
    # input: sentences [real matrix, where for each row we have the embedding of the sentence], n_clusters [int, number of clusters], shperica: boolean
    # output: centroids [real matrix, where for each row we have the centroid of the cluster], label_clustering [int vector, for each cell the cluster of the doc]
    def standard_k_means(self, sentences, n_clusters, spherical=False):

        clustering = faiss.Kmeans(sentences.shape[1], n_clusters,
                                spherical=spherical,
                                gpu=True)
        clustering.train(sentences)

        _, label_clustering = clustering.index.search(sentences, 1)

        return clustering.centroids, label_clustering


    # label_centroids: (give the sentiment of each centroid)
    # input: n_clusters [int, number of clusters], centroids [real matrix, where each row is a cluster embedding],
    # label_clustering [int vector, for each cell the cluster of the sentence], sentiment [-1 or +1 vector, that is the sentiment of each sentence]
    # output: [-1 or 1 vector, that is the sentiment of each centroid]
    def label_centroids(self, n_clusters, label_clustering, sentiment):
        centroids_sentiment = []
        for i in tqdm(range(n_clusters)):
            indices = np.where(label_clustering == i)[0]
            if np.sum(sentiment[indices]) >= 0:
                centroids_sentiment.append(1)
            else:
                centroids_sentiment.append(-1)
        return np.array(centroids_sentiment)


    # get_result: (give the result sentiment for each query)
    # input: queries [real matrix, where each row is a query embedding],
    # centroids [real matrix, where each row is a centroid embedding],
    # centroids_labeled [-1 or 1 vector, that is the sentiment for each centroid],
    # top_k [int, top k cluster centroids that we consider for our final result]
    # output: [-1 or 1 vector, where for each query we have the related label]
    def get_result(self, queries, centroids, centroids_labeled, top_k=1):

        def compute_result(query, centroids=centroids,
                        centroids_labeled=centroids_labeled, top_k=top_k):
            dp_score = centroids.dot(query)
            top_k_ind = np.argpartition(dp_score, -top_k)[-top_k:]
            if np.sum(centroids_labeled[top_k_ind]) >= 0: return 1
            else: return -1

        return np.apply_along_axis(compute_result, axis=1, arr=queries)




    def run_faiss_kmeans(self, dataset_name, methods_name, spherical = False):

        x_train, x_test, y_train, y_test = read_embbedings(dataset_name, methods_name)

        for n_clusters in self.n_clusters_list:
            centroids, label_clustering = self.spherical_k_means(x_train, n_clusters) if spherical else self.standard_k_means(x_train, n_clusters)
            sentiment_centroids = self.label_centroids(n_clusters, centroids,
                                                label_clustering, y_train)
            for top_k in self.top_k_list:
                if top_k < n_clusters:
                    query_result = self.get_result(x_test, centroids,
                                            sentiment_centroids, top_k=top_k)
                    test_accuracy = accuracy_result(query_result, y_test)
                    #print('Result (n. clusters = {0} and k = {1}): {2}'.format(n_clusters, top_k, test_accuracy))

                    write_csv(
                        ts_dir=self.timestamp,
                        head = ['method', 'dataset', 'test_accuracy'],
                        values = [methods_name, dataset_name, test_accuracy],
                        categoty_type='our_approaches'
                    )
                    