class LayerWise():
	def __init__(self, device, datasets_dict, model, tokenizer, embedding_split_perc):
		self.device = device
		self.datasets_dict = datasets_dict
		self.model = model
		self.tokenizer = tokenizer
		self.embedding_split_perc = embedding_split_perc
