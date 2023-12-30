# -*- coding: utf-8 -*-

from utils import accuracy_score, get_datasets
from transformers import BertTokenizer, BertModel

from approaches.MainApproch import MainApproch
from approaches.LayerWise import LayerWise
from approaches.LayerAggregation import LayerAggregation

from competitros.BertLinears import BertLinears

import torch
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    
	embedding_split_perc = 0.1
    
	datasets_dict = get_datasets()

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained("bert-base-uncased").to(device)

	loss_fn = nn.CrossEntropyLoss()

	# our approaches
	main_approach = MainApproch(device, datasets_dict, model, tokenizer, embedding_split_perc)
	layer_wise = LayerWise(device, datasets_dict, model, tokenizer, embedding_split_perc)
	layer_aggregation = LayerAggregation(
		device=device,
		batch_size=16,
		datasets_dict=datasets_dict,
  		model=model,
    	tokenizer=tokenizer,
     	embedding_split_perc=embedding_split_perc,
      	loss_fn=loss_fn,
		score_fn=accuracy_score,
		patience = 3,
		epochs = 10,
		dim_embedding = 768 * 12
    )
 
	
	# competitors
	bert_linears = BertLinears(
      	device=device,
		batch_size=16,
		datasets_dict=datasets_dict,
  		model=model,
    	tokenizer=tokenizer,
     	embedding_split_perc=embedding_split_perc,
      	loss_fn=loss_fn,
		score_fn=accuracy_score,
		patience = 3,
		epochs = 10
  	)
 
 
 
	methods = [
		# our approaches
		main_approach,
		layer_wise,
		layer_aggregation,

		# competitors
		#bert_linears,
  
		# baselines
	]
 
	for method in methods:
		method.run()


if __name__ == "__main__":
    main()
    

