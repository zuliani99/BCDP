# Clustering BERT Eembedding via Dot Product (CBERTdp)

We aim to simplify classification tasks by employing K- Means clustering and performing dot-product. As an example we make use of Sentiment Analy- sis on different kinds of reviews. We utilize a pretrained BERT (Devlin et al., 2019) model to obtain embeddings for the reviews, subsequently subjecting these to K-Means clustering and per- forming dot-product operations to obtain the final label. 

## Approaches
### Main Approaches
for each sentence or passage of our training set we execute BERT (Bidirectional Encoder Representations from Transformers) in order to obtain the embedding vector of that sentence. Once we have all the embeddings, we cluster these embeddings (using FAISS Spherical/Standard K-Means. The value of K for K-Means can vary and computational analyses will be carried out based on which is the best K, also considering the accuracy and the complexity. Each centroid of each cluster is labelled either as positive or as negative (pre-processing for Sentiment Analysis) using majority vote as a criterion. This last part concludes the offline phase which is necessary for us to assign a label when we have a new query.
The second step is the online step involving a new sentence or passage. First of all, given the query we are going to execute the BERT model to get the vector representation. Once obtained, we compute the dot-product between the query and the centroids calculated in the offline step. In the case of two centroids, we assign to the query the label corresponding to the centroid with larger dot-product value. This idea will be extended to having multiple centroids where the final label is assigned according to majority vote.
Using this technique we can obtain expressiveness and calculation speed whilst keeping the technique extremely scalable, because in the online phase it is only necessary to run BERT on a single query and a few dot-products to obtain the final solution. In this way, we are able to achieve faster results without employing another neural network for classification or having to use BERT several times. At the same time, using BERT to obtain the embeddings should preserve expressiveness and accuracy.

### 2nd & 3th Approaches
For further improvement of the model's accuracy, we propose a second approach that utilizes layer-wise embeddings of BERT for the benefit of greater expressiveness. Each layer captures the sentence's meaning at a different level of abstraction resulting in multiple embeddings that provide a richer representation of the meaning of the sentence. 
The second approach is in turn divided into two sub-projects: in the first one, we consider each layer-embedding separately and for each of them we perform what we have described in the previous approach. As regards the second sub-project, an extra attention layer is added to incorporate all the layer-wise embeddings given by BERT (layer-aggregation). In this variation we decided to fine-tune the model by freezing the BERT layers while allowing the newer attention layer to train its weights. This procedure aids to avoid the phenomenon of Catastrophic Forgetting. Another different solution is to fine-tune the complete models with a very low learning rate (2e-5). Continuing the whole model is done to achieve a better vector representation in the output that considers all the different nuances of the different layers in order to solve our problem optimally. Once having extracted the embedding, the remaining procedure is equal to the first approach.

## Packages Installation
```
!pip install torch
!pip install transformers
!pip install datasets
!pip install faiss-gpu
```

## Run
```
python app/main.PY
```
