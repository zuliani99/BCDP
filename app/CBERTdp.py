# -*- coding: utf-8 -*-

from utils import get_datasets
from transformers import BertTokenizer, BertModel

from MainApproch import MainApproch
from LayerWise import LayerWise
from LayerAggregation import LayerAggregation

import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
	embedding_split_perc = 0.1
    
	datasets_dict = get_datasets()

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained("bert-base-uncased").to(device)
 
	main_approach = MainApproch(device, datasets_dict, model, tokenizer, embedding_split_perc)
	layer_wise = LayerWise(device, datasets_dict, model, tokenizer, embedding_split_perc)
	layer_aggregation = LayerAggregation(device, datasets_dict, model, tokenizer, embedding_split_perc)
 
	methods = [main_approach, layer_wise, layer_aggregation]
 
	for method in methods:
		method.run()


if __name__ == "__main__":
    main()
    

