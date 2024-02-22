# Clustering BERT Eembedding via Dot Product (CBERTdp)

In this study, we explore strategies to reduce the computational complexity of sentiment analysis methods, which commonly rely on resource-intensive neural networks. Our investigation focuses on leveraging BERT-extracted embeddings and clustering techniques to streamline the sentiment classification process. Specifically, we propose a novel approach where we cluster BERT embeddings and classify sentiment by computing the dot product between a new sentence's embedding and cluster centroids. We present three variants of this approach, each offering a different trade-off between computational efficiency and accuracy.

Our findings reveal that only the variant incorporating an attention layer achieves satisfactory results in terms of sentiment classification accuracy. This approach demonstrates moderate computational costs compared to other baselines.

Overall, our study sheds light on promising avenues for reducing the computational overhead of sentiment analysis, highlighting the potential of leveraging clustering techniques and attention mechanisms for more efficient and effective sentiment classification.


## Used Dataseets
1. [imdb](https://huggingface.co/datasets/imdb) 
2. [sst2](https://huggingface.co/datasets/sst2)
3. [yelp_polarity](https://huggingface.co/datasets/yelp_polarity)


## Enviroment Setup
```
conda create --name <env> --file requirements.txt
```

## Note before running the application
We would like to inform the user that the obtaining of the base embedding from all of three datasets woould require much time since its dimension expecially for *yelp_polarity*.

## How to Run
```
usage: main.py [-h] -s {our_approaches,competitors,baselines}
               [{our_approaches,competitors,baselines} ...] -a ABLATIONS -m
               {BERT,DISTILBERT}

optional arguments:
  -h, --help            show this help message and exit
  -s {our_approaches,competitors,baselines} [{our_approaches,competitors,baselines} ...], --strategies {our_approaches,competitors,baselines} [{our_approaches,competitors,baselines} ...]
                        Possible strategies to run
  -a ABLATIONS, --ablations ABLATIONS
                        Bool ablations
  -m {BERT,DISTILBERT}, --model {BERT,DISTILBERT}
                        Pretreined BERT model from Huggingface
```

## Results
We have uploaded the **.csv** file containig our results for every strategies including baselines, competitors, ablations and of course our approacches, you can see them following this [link](https://github.com/zuliani99/CBERTdp/tree/main/app/results/our_results).

## Cite Us
```
@online{CBERTdp,
    author = "Thomas Vecchiato, Riccardo Zuliani, Alice Schirrmeister, Isabel Marie Ritter",
    title = "Clustering BERT Eembedding via Dot Product (CBERTdp)",
    url  = "https://github.com/zuliani99/CBERTdp/blob/main/Project_Report_NLP.pdf",
}
```
