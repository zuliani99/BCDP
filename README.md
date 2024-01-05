# Clustering BERT Eembedding via Dot Product (CBERTdp)

We aim to simplify classification tasks by employing K- Means clustering and performing dot-product. As an example we make use of Sentiment Analy- sis on different kinds of reviews. We utilize a pretrained BERT (Devlin et al., 2019) model to obtain embeddings for the reviews, subsequently subjecting these to K-Means clustering and per- forming dot-product operations to obtain the final label. 

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
