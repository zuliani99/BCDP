
import numpy as np
import faiss
from sklearn.metrics import classification_report
import time

from utils import write_csv

class Faiss_KMEANS():
    
    def __init__(self):
        self.n_clusters_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.top_k_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]



    def k_means(self, sentences, n_clusters, spherical=False):
        """ corresponds to the standard k-means clustering algorithm
        @param sentences: numpy.ndarray, real matrix where each row corresponds to the embeddings of the sentence
        @param n_clusters: int, desired number of clusters (k)
        @param spherical: boolean,  controls whether to use the spherical k-means, default is False
            
        @return:
            centroids: numpy.ndarray, real matrix, each row correponds to the centroid of one cluster
            label_clustering: Integer vector where each element represents the cluster label for the corresponding sentence
        """

        clustering = faiss.Kmeans(sentences.shape[1], n_clusters, spherical=spherical, gpu=True)
        
        clustering.train(sentences)

        _, label_clustering = clustering.index.search(sentences, 1)

        return clustering.centroids, label_clustering


    def label_centroids(self, n_clusters, label_clustering, sentiment):
        """ Determines the sentiment of each cluster centroid based on the majority sentiment of its assigned sentences
        @param n_clusters: int, number of distinct clusters
        @param label_clustering: Integer vector where each element represents the cluster label for the corresponding sentence
        @param sentiment: Integer vector, only values -1 or 1, corresponding sentiment of each sentence
        
        @return:
            Integer vector, only values -1 or 1, contains the sentiment of each centroid

        """
        centroids_sentiment = []
        for i in range(n_clusters):
            indices = np.where(label_clustering == i)[0]
            if np.sum(sentiment[indices]) >= 0: centroids_sentiment.append(1)
            else: centroids_sentiment.append(-1)
            
        return np.array(centroids_sentiment)



    def confidence(self, n_clusters, label_clustering, sentiment):
        """" measures the purity of each cluster as proposed by @Vascon 2013

        @param n_clusters: int, number of clusters
        @param label_clustering: Integer vector, corresponding cluster of each sentence
        @param sentiment: Integer vector, only values -1 or 1, corresponding sentiment of each sentence

        @return:
            confidence: int, average confidence measure of all clusters in our space"""

        confidence = np.zeros(n_clusters)

        for i in range(n_clusters):
            count_neg = 0
            count_pos = 0

            indices = np.where(label_clustering == i)[0]
            count_pos = np.sum(sentiment[indices] == 1)
            count_neg = len(indices) - count_pos

            confidence[i] = abs(count_neg - count_pos)/(count_neg + count_pos)

        return np.mean(confidence)



    def get_result(self, queries, centroids, centroids_labeled, top_k=1):
        """ computes the sentiment for each query based on the clustering
        @param queries: numpy.ndarray, Real matrix storing in each row a query embedding
        @param centroids: numpy.ndarray, Real matrix storing in each row a centroid embedding
        @param centroids_labeled: Integer vector, only values -1 or 1, contains the sentiment of each centroid
        @param top_k: int, number of cluster centroids that should be considered for the final result, default is 1

        @return:
            Integer vector, only values -1 or 1, the corresponding sentiment for each query
        """

        def compute_result(query, centroids=centroids, centroids_labeled=centroids_labeled, top_k=top_k):
            dp_score = centroids.dot(query)
            top_k_ind = np.argpartition(dp_score, -top_k)[-top_k:]
            if np.sum(centroids_labeled[top_k_ind]) >= 0: return 1
            else: return -1

        return np.apply_along_axis(compute_result, axis=1, arr=queries)



    def accuracy_result(self, model_results, ground_truth):
        """ calculates the accuarcy of the model's predictions

        @param model_results: Integer vector, contains only the values -1 and 1, predicted sentiment for all sentences
        @param ground_truth: Integer vector, contains only the values -1 and 1, actual sentiment for all sentences

        @return:
            the accuracy of the predictions
        """
        result_list = 0
        for i in range(ground_truth.shape[0]):
            if model_results[i] == ground_truth[i]: result_list += 1
        return result_list/ground_truth.shape[0]



    
    def report(self, model_results, ground_truth):
        """ calculates precision, recall, F1-measure for each class and the accuracy
        @param model_results: Integer vector, contains only the values -1 and 1, predicted sentiment for all sentences
        @param ground_truth: Integer vector, contains only the values -1 and 1, actual sentiment for all sentences
            
        @return:
            tuple containing a dictionary with precision, recall, F1-measure for the negative class, one dictionary for the positive class and the accuracy """

        results = classification_report(ground_truth, model_results)

        return results['-1'], results['1'], results['accuracy']




    def run_faiss_kmeans(self, dataset_name, methods_name, timestamp, data, categoty_type, spherical = False):
        """ performs the clustering and the predictions of the model, evaluating its accuracy and F1-measure and the confidence of the clusters

        @param dataset_name: str, dataset to be used in the run
        @param methods_name: str, method to be used in the run
        @param timestamp: str, timestamp to identify and store the run
        @param spherical: boolean, determines whether spherical k-means is performed or not, default is False

        @return: None
        """

        x_train, x_test, y_train, y_test = data

        for n_clusters in self.n_clusters_list:
            
            centroids, label_clustering = self.k_means(x_train, n_clusters, spherical)
            confidence = self.confidence(n_clusters, label_clustering, y_train)
            sentiment_centroids = self.label_centroids(n_clusters, label_clustering, y_train)

            for top_k in self.top_k_list:
                if top_k < n_clusters:
                    start = time.time()
                    query_result = self.get_result(x_test, centroids, sentiment_centroids, top_k=top_k)
                    end = time.time()
                    
                    evaluations = self.report(query_result, y_test)

                    print('Result (n. clusters = {0} and k = {1}): {2}'.format(n_clusters, top_k, evaluations[-1]))

                    write_csv(
                        ts_dir = timestamp,
                        head = ['method', 'dataset', 'test_accuracy', 'confidence', 'F1-measure negative', 'F1-measure positive', 'elapsed'],
                        values = [methods_name, dataset_name, evaluations[-1], confidence, evaluations[0], evaluations[1], end-start],
                        categoty_type = categoty_type
                    )