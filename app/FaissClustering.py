
import numpy as np
import faiss
from sklearn.metrics import classification_report
import time

from utils import write_csv

class Faiss_KMEANS():
    
    def __init__(self):
        
        self.n_clusters_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.top_k_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Corresponds to the standard k-means clustering algorithm
    def k_means(self, sentences, n_clusters, spherical=False):
        # sentences -> real matrix where each row corresponds to the embeddings of the sentence
        # n_clusters -> desired number of clusters (k)
        
        clustering = faiss.Kmeans(sentences.shape[1], n_clusters, spherical=spherical, gpu=True)
        
        clustering.train(sentences)

        _, label_clustering = clustering.index.search(sentences, 1)
        
        # centroids -> real matrix, each row correponds to the centroid of one cluster
        # label_clustering -> Integer vector where each element represents the cluster label for the corresponding sentence
        return clustering.centroids, label_clustering


    # Determines the sentiment of each cluster centroid based on the majority sentiment of its assigned sentences
    def label_centroids(self, n_clusters, label_clustering, sentiment):
        # n_clusters -> desired number of clusters (k)
        #label_clustering -> vector where each element represents the cluster label for the corresponding sentence
        #sentiment -> vector, only values -1 or 1, corresponding sentiment of each sentence

        centroids_sentiment = []
        for i in range(n_clusters):
            indices = np.where(label_clustering == i)[0]
            if np.sum(sentiment[indices]) >= 0: centroids_sentiment.append(1)
            else: centroids_sentiment.append(-1)
            
        # Only values -1 or 1, contains the sentiment of each centroid
        return np.array(centroids_sentiment)


    # Measures the purity of each cluster as proposed by @Vascon 2013
    def confidence(self, n_clusters, label_clustering, sentiment):

        confidence = np.zeros(n_clusters)

        for i in range(n_clusters):
            count_neg = 0
            count_pos = 0

            indices = np.where(label_clustering == i)[0]
            count_pos = np.sum(sentiment[indices] == 1)
            count_neg = len(indices) - count_pos

            confidence[i] = abs(count_neg - count_pos)/(count_neg + count_pos)

        # Average confidence measure of all clusters in our space
        return np.mean(confidence)


    # Computes the sentiment for each query based on the clustering
    def get_result(self, queries, centroids, centroids_labeled, top_k=1):
        
        # queries -> matrix storing in each row a query embedding
        # centroids -> eal matrix storing in each row a centroid embedding
        # centroids_labeled -> Integer vector, only values -1 or 1, contains the sentiment of each centroid
        # top_k: int, number of cluster centroids that should be considered for the final result, default is 1

        def compute_result(query, centroids=centroids, centroids_labeled=centroids_labeled, top_k=top_k):
            dp_score = centroids.dot(query)
            top_k_ind = np.argpartition(dp_score, -top_k)[-top_k:]
            if np.sum(centroids_labeled[top_k_ind]) >= 0: return 1
            else: return -1

        # Vector only values -1 or 1, the corresponding sentiment for each query
        return np.apply_along_axis(compute_result, axis=1, arr=queries)



    # Calculates precision, recall, F1-measure for each class and the accuracy
    def report(self, y_true, y_pred):
        
        results = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return results['accuracy'], results['-1']['f1-score'], results['1']['f1-score']


    # erforms the clustering and the predictions of the model, evaluating its accuracy and F1-measure and the confidence of the clusters
    def run_faiss_kmeans(self, dataset_name, methods_name, timestamp, categoty_type, data, spherical = False):

        x_train, x_test, y_train, y_test = data
                        
        for n_clusters in self.n_clusters_list:
            
            centroids, label_clustering = self.k_means(x_train, n_clusters, spherical)
            confidence = self.confidence(n_clusters, label_clustering, y_train)
            sentiment_centroids = self.label_centroids(n_clusters, label_clustering, y_train)

            for top_k in self.top_k_list:
                if top_k < n_clusters:
                    start = time.time()
                    query_result = self.get_result(x_test, centroids, sentiment_centroids, top_k=top_k).astype(np.int8)
                    end = time.time()
                                        
                    accuracy, neg_f1, pos_f1 = self.report(y_test, query_result)

                    print('Result -> \t (n. clusters = {0} and k = {1}): \t {2}'.format(n_clusters, top_k, accuracy))

                    write_csv(
                        ts_dir = timestamp,
                        head = ['method', 'dataset', 'n_clusters', 'top_k', 'test_accuracy', 'confidence', 'negative_f1', 'positive_f1', 'elapsed'],
                        values = [methods_name, dataset_name, n_clusters, top_k, accuracy, confidence, neg_f1, pos_f1, end-start],
                        categoty_type = categoty_type
                    )