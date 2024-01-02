
import numpy as np
from sklearn.model_selection import train_test_split
import faiss
from tqdm.auto import tqdm

class FaissClustering():
    def __init__(self):
        self.n_clusters_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.top_k_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.size_split = 0.2
 

    def get_embbedings_and_labels(self, dataset_name, methods_name):

        path = f'app/embeddings/{dataset_name}/{methods_name}'
        embds = np.concatenate([np.load(f'{path}/train_embeddings.npy'), np.load(f'{path}/val_embeddings.npy'), np.load(f'{path}/test_embeddings.npy')], 0, dtype=np.float32)
        sents = np.concatenate([np.load(f'{path}/train_labels.npy'), np.load(f'{path}/val_labels.npy'), np.load(f'{path}/test_labels.npy')], 0, dtype=np.float32)
        x_train, x_test, y_train, y_test = train_test_split(embds, sents, test_size=self.size_split, random_state=42)

        return x_train, x_test, y_train, y_test



    # standard_k_means: (standard k-means clustering algorithm)
    # input: sentences [real matrix, where for each row we have the embedding of the sentence], n_clusters [int, number of clusters]
    # output: centroids [real matrix, where for each row we have the centroid of the cluster], label_clustering [int vector, for each cell the cluster of the doc]
    def standard_k_means(self, sentences, n_clusters):

        clustering = faiss.Kmeans(sentences.shape[1], n_clusters,
                                spherical=False,
                                gpu=True)
        clustering.train(sentences)

        _, label_clustering = clustering.index.search(sentences, 1)

        return clustering.centroids, label_clustering


    # spherical_k_means: (spherical k-means clustering algorithm)
    # input: sentences [real matrix, where for each row we have the embedding of the sentence], n_clusters [int, number of clusters]
    # output: centroids [real matrix, where for each row we have the centroid of the cluster], label_clustering [int vector, for each cell the cluster of the doc]
    def spherical_k_means(self, sentences, n_clusters):

        clustering = faiss.Kmeans(sentences.shape[1], n_clusters,
                                spherical=True,
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



    def accuracy_result(model_results, ground_truth):
        result_list = 0
        for i in range(ground_truth.shape[0]):
            if model_results[i] == ground_truth[i]:
                result_list += 1
        return result_list/ground_truth.shape[0]




    def run_faiss_kmeans(self, dataset_name, methods_name, spherical = False):

        x_train, x_test, y_train, y_test = self.get_embbedings_and_labels(dataset_name, methods_name)

        for n_clusters in self.n_clusters_list:
            centroids, label_clustering = self.spherical_k_means(x_train, n_clusters) if spherical else self.standard_k_means(x_train, n_clusters)
            sentiment_centroids = self.label_centroids(n_clusters, centroids,
                                                label_clustering, y_train)
            for top_k in self.top_k_list:
                if top_k < n_clusters:
                    query_result = self.get_result(x_test, centroids,
                                            sentiment_centroids, top_k=top_k)
                    print('Result (n. clusters = {0} and k = {1}): {2}'.format(n_clusters, top_k, self.accuracy_result(query_result, y_test)))

            print('\n\n')