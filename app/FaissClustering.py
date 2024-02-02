
import numpy as np
import faiss
from tqdm.auto import tqdm

from utils import read_embbedings, write_csv

class FaissClustering():
    def __init__(self):
        self.n_clusters_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.top_k_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.size_split = 0.2



    def k_means(self, sentences, n_clusters, spherical=False):
        """ corresponds to the standard k-means clustering algorithm
        @param sentences: numpy.ndarray, real matrix where each row corresponds to the embeddings of the sentence
        @param n_clusters: int, desired number of clusters (k)
        @param spherical: boolean,  controls whether to use the spherical k-means, default is False
            
        @return:
            centroids: numpy.ndarray, real matrix, each row correponds to the centroid of one cluster
            label_clustering: Integer vector where each element represents the cluster label for the corresponding sentence
        """

        clustering = faiss.Kmeans(sentences.shape[1], n_clusters,
                                spherical=spherical,
                                gpu=True)
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
        for i in tqdm(range(n_clusters)):
            indices = np.where(label_clustering == i)[0]
            if np.sum(sentiment[indices]) >= 0:
                centroids_sentiment.append(1)
            else:
                centroids_sentiment.append(-1)
        return np.array(centroids_sentiment)



    def confidence(self, n_clusters, label_clustering, sentiment):
        """" measures the purity of each cluster as proposed by @Vascon 2013

        @param n_clusters: int, number of clusters
        @param label_clustering: Integer vector, corresponding cluster of each sentence
        @param sentiment: Integer vector, only values -1 or 1, corresponding sentiment of each sentence

        @return:
            confidence: numpy.ndarray, purity value of the cluster; 0 if both classes are equally often present in the cluster, 1 if only one class is present"""

        confidence = np.zeros(n_clusters)

        for i in range(n_clusters):
            count_neg = 0
            count_pos = 0

            indices = np.where(label_clustering == i)[0]
            count_pos = np.sum(sentiment[indices] == 1)
            count_neg = len(indices) - count_pos

            confidence[i] = abs(count_neg - count_pos)/(count_neg + count_pos)

        return confidence



    def get_result(self, queries, centroids, centroids_labeled, top_k=1):
        """ computes the sentiment for each query based on the clustering
        @param queries: numpy.ndarray, Real matrix storing in each row a query embedding
        @param centroids: numpy.ndarray, Real matrix storing in each row a centroid embedding
        @param centroids_labeled: Integer vector, only values -1 or 1, contains the sentiment of each centroid
        @param top_k: int, number of cluster centroids that should be considered for the final result, default is 1

        @return:
            Integer vector, only values -1 or 1, the corresponding sentiment for each query
        """

        def compute_result(query, centroids=centroids,
                        centroids_labeled=centroids_labeled, top_k=top_k):
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
            if model_results[i] == ground_truth[i]:
                result_list += 1
        return result_list/ground_truth.shape[0]




    def precision(self, model_results, ground_truth):
        """ calculates the precision of the predictions, i. e. the accuracy of the positive predictions
        
        @param model_results: Integer vector, contains only the values -1 and 1, predicted sentiment for all sentences
        @param ground_truth: Integer vector, contains only the values -1 and 1, actual sentiment for all sentences

        @return:
            the corresponding precision as float, returns 0 if the denominator evaluates to 0
        """
        true_p = 0
        false_p = 0
        indices = np.where(model_results == 1)[0]
        for i in indices:
            if model_results[i] == ground_truth[i]: true_p += 1
            else: false_p += 1
        return true_p/(true_p + false_p) if true_p + false_p != 0 else 0
    
    
    def recall(self, model_results, ground_truth):
        """ measures the ability of the model to correctly identify all relevant instances

        @param model_results: Integer vector, contains only the values -1 and 1, predicted sentiment for all sentences
        @param ground_truth: Integer vector, contains only the values -1 and 1, actual sentiment for all sentences

        @return:
            the corresponding recall as float, 0 if the denominator evaluates to 0
        """
        true_p = 0
        false_n = 0
        indices = np.where(model_results == 1)[0]
        for i in indices:
            if model_results[i] == ground_truth[i]: true_p += 1
        
        indices_n = np.where(model_results == 0)[0]
        for i in indices_n:
            if model_results[i] != ground_truth[i]: false_n += 1
        
        return true_p /(true_p+false_n) if true_p + false_n != 0 else 0




    def F1(self, model_results, ground_truth):
        """ calculates the F1-measure, i.e the harmonic mean between precision and recall, of the predictions
        @param model_results: Integer vector, contains only the values -1 and 1, predicted sentiment for all sentences
        @param ground_truth: Integer vector, contains only the values -1 and 1, actual sentiment for all sentences
            
        @return:
            precision, recall, F1 of the model """
        precision = self.precision(model_results, ground_truth)
        recall = self.recall(model_results, ground_truth)
        F1 = 2*precision*recall/(precision + recall)

        return (precision, recall, F1)




    def run_faiss_kmeans(self, dataset_name, methods_name, timestamp, spherical = False):
        """ performs the clustering and the predictions of the model, evaluating its accuracy and F1-measure and the confidence of the clusters

        @param dataset_name: str, dataset to be used in the run
        @param methods_name: str, method to be used in the run
        @param timestamp: str, timestamp to identify and store the run
        @param spherical: boolean, determines whether spherical k-means is performed or not, default is False

        @return: None
        """

        x_train, x_test, y_train, y_test = read_embbedings(dataset_name, methods_name)

        for n_clusters in self.n_clusters_list:
            
            centroids, label_clustering = self.k_means(x_train, n_clusters, spherical)
            confidence = self.confidence(n_clusters, label_clustering, y_train)
            sentiment_centroids = self.label_centroids(n_clusters, label_clustering, y_train)

            for top_k in self.top_k_list:
                if top_k < n_clusters:
                    query_result = self.get_result(x_test, centroids, sentiment_centroids, top_k=top_k)
                    test_accuracy = self.accuracy_result(query_result, y_test)
                    harmonic_mean = self.F1(query_result, y_test)
                    
                    print('Result (n. clusters = {0} and k = {1}): {2}'.format(n_clusters, top_k, test_accuracy))

                    write_csv(
                        ts_dir = timestamp,
                        head = ['method', 'dataset', 'test_accuracy', 'confidence', 'F1-measure'],
                        values = [methods_name, dataset_name, test_accuracy, confidence, harmonic_mean],
                        categoty_type='our_approaches'
                    )