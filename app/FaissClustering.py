
import numpy as np
import faiss
from tqdm.auto import tqdm

from utils import read_embbedings, write_csv

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



    def confidence(n_clusters, label_clustering, sentiment):
        """" measures the purity of each cluster as proposed by @Vascon 2013
        @input:
            int n_clusters: number of clusters
            int vector label_clustering: corresponding cluster of each sentence
            int vector: only values -1 or 1; corresponding sentiment of each sentence
        @output:
            np array confidence: purity value of the cluster; 0 if both classes are equally often present in the cluster, 1 if only one class is present"""

        confidence = np.zeros(n_clusters)

        for i in range(n_clusters):
            count_neg = 0
            count_pos = 0

            indices = np.where(label_clustering == i)[0]
            count_pos = np.sum(sentiment[indices] == 1)
            count_neg = len(indices) - count_pos

            confidence[i] = abs(count_neg - count_pos)/(count_neg + count_pos)

        return confidence



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



    def accuracy_result(model_results, ground_truth):
        result_list = 0
        for i in range(ground_truth.shape[0]):
            if model_results[i] == ground_truth[i]:
                result_list += 1
        return result_list/ground_truth.shape[0]




    def precision(model_results, ground_truth):
        """ calculates the precision of the predictions, i. e. the accuracy of the positive predictions
        @input:
            int vector model_results: contains only the values -1 and 1, predicted sentiment for all sentences
            int vector ground_truth: contains only the values -1 and 1, actual sentiment for all sentences
        @output:
            int: the corresponding prediction
        """
        true_p = 0
        false_p = 0
        indices = np.where(model_results == 1)
        for i in indices:
            if model_results[i] == ground_truth[i]: true_p += 1
            else: false_p += 1
        return true_p/(true_p + false_p)
    
    def recall(model_results, ground_truth):
        """ measures the ability of the model to correctly identify all relevant instances
            @input:
                int vector model_results: contains only the values -1 and 1, predicted sentiment for all sentences
                    int vector ground_truth: contains only the values -1 and 1, actual sentiment for all sentences
                @output:
                    int: the corresponding recall
        """
        true_p = 0
        false_n = 0
        indices = np.where(model_results == 1)
        for i in indices:
            if model_results[i] == ground_truth[i]: true_p += 1
        
        indices_n = np.where(model_results == 0)
        for i in indices_n:
            if model_results[i] != ground_truth[i]: false_n += 1
        
        return true_p /(true_p+false_n)




    def F1(model_results, ground_truth):
        """ calculates the F1-measure, i.e the harmonic mean between precision and recall, of the predictions
        @input:
            int vector model_results: contains only the values -1 and 1, predicted sentiment for all sentences
            int vector ground_truth: contains only the values -1 and 1, actual sentiment for all sentences
        @output:
            int tuple: (precision, recall, F1-measure) """
        precision = precision(model_results, ground_truth)
        recall = recall(model_results, ground_truth)
        F1 = 2*precision*recall/(precision + recall)

        return (precision, recall, F1)




    def run_faiss_kmeans(self, dataset_name, methods_name, timestamp, spherical = False):

        x_train, x_test, y_train, y_test = read_embbedings(dataset_name, methods_name)

        for n_clusters in self.n_clusters_list:
            centroids, label_clustering = self.spherical_k_means(x_train, n_clusters) if spherical else self.standard_k_means(x_train, n_clusters)
            confidence = self.confidence(n_clusters, label_clustering, y_train)
            sentiment_centroids = self.label_centroids(n_clusters, centroids,
                                                label_clustering, y_train)

            for top_k in self.top_k_list:
                if top_k < n_clusters:
                    query_result = self.get_result(x_test, centroids,
                                            sentiment_centroids, top_k=top_k)
                    test_accuracy = self.accuracy_result(query_result, y_test)
                    harmonic_mean = self.F1(query_result, y_test)
                    #print('Result (n. clusters = {0} and k = {1}): {2}'.format(n_clusters, top_k, test_accuracy))

                    write_csv(
                        ts_dir = timestamp,
                        head = ['method', 'dataset', 'test_accuracy', 'confidence', 'F1-measure'],
                        values = [methods_name, dataset_name, test_accuracy, confidence, harmonic_mean],
                        categoty_type='our_approaches'
                    )