import re
import sys

import numpy as np
from transformers import AutoTokenizer, BertConfig
import aquilign.segmenter.utils as utils
import aquilign.segmenter.models as models
import torch
import tqdm
import os


class Tagger:
	def  __init__(self,
				  model_dir):
		"""
		Main Class trainer
		"""
		config_file = utils.read_to_dict(f"{model_dir}/config/config.json")
		vocab_path = f"{model_dir}/vocab"
		self.saved_model = f"{model_dir}/{config_file['global']['model_path']}"
		if ".safetensors" in self.saved_model:
			use_safetensors = True
		else:
			use_safetensors = False
		architecture = config_file["architecture"]["name"]
		device = config_file["global"]["device"]
		workers = config_file["global"]["workers"]
		base_model_name = config_file["global"]["base_model_name"]
		use_bert_tokenizer = config_file["global"]["use_bert_tokenizer"]
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		include_lang_metadata = config_file["global"]["include_lang_metadata"]
		lang_emb_dim = config_file["global"]["lang_emb_dim"]
		linear_layers = config_file["global"]["linear_layers"]
		self.use_pretrained_embeddings = config_file["global"]["use_pretrained_embeddings"]
		linear_layers_hidden_size = config_file["global"]["linear_layers_hidden_size"]
		emb_dim = config_file["global"]["emb_dim"]
		if use_bert_tokenizer or architecture in ["BERT", "DISTILBERT"]:
			self.tokenizer = AutoTokenizer.from_pretrained(config_file["global"]["base_model_name"])
		if architecture == "lstm":
			add_attention_layer = config_file["architecture"]["add_attention_layer"]
			lstm_hidden_size = config_file["architecture"]["lstm_hidden_size"]
			num_lstm_layers = config_file["architecture"]["num_lstm_layers"]
			lstm_dropout = config_file["architecture"]["lstm_dropout"]
			linear_dropout = config_file["architecture"]["linear_dropout"]
			bidirectional = config_file["architecture"]["bidirectional"]
			keep_bert_dimensions = config_file["architecture"]["keep_bert_dimensions"]
		elif architecture == "gru":
			add_attention_layer = config_file["architecture"]["add_attention_layer"]
			hidden_size = config_file["architecture"]["hidden_size"]
			num_layers = config_file["architecture"]["num_layers"]
			dropout = config_file["architecture"]["dropout"]
			bidirectional = config_file["architecture"]["bidirectional"]
		elif architecture == "transformer":
			num_heads = config_file["architecture"]["num_heads"]
			num_transformers_layers = config_file["architecture"]["num_transformers_layers"]
		elif architecture == "cnn":
			dropout = config_file["architecture"]["dropout"]
			cnn_scale = config_file["architecture"]["cnn_scale"]
			add_attention_layer = config_file["architecture"]["add_attention_layer"]
			keep_bert_dimensions = config_file["architecture"]["keep_bert_dimensions"]
			hidden_size = config_file["architecture"]["hidden_size"]
			kernel_size = config_file['architectures'][architecture]["kernel_size"]
			linear_dropout = config_file["architecture"]["linear_dropout"]
			positional_embeddings = config_file['architectures'][architecture]["positional_embeddings"]
			num_heads = config_file["architecture"]["num_heads"]
			num_cnn_layers = config_file["architecture"]["num_cnn_layers"]



		# First we prepare the corpus
		self.device = device
		if self.device != "cpu":
			device_name = torch.cuda.get_device_name(self.device)
			print(f"Device name: {device_name}")
		self.workers = workers
		self.all_dataset_on_device = False
		print("Loading data")
		self.use_bert_tokenizer = use_bert_tokenizer
		if architecture in ["BERT", "DISTILBERT"] or self.use_bert_tokenizer or self.use_pretrained_embeddings:
			self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
		self.tokens_regexp = re.compile(r"\s+|([\.“\?\!—\"/:;,\-¿«\[\]»])")
		self.base_model_name = base_model_name





		self.input_vocab = utils.read_to_dict(f"{vocab_path}/input_vocab.json")
		self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}
		self.lang_vocab = utils.read_to_dict(f"{vocab_path}/lang_vocab.json")
		assert self.input_vocab != {}, "Error with input vocabulary"

		self.target_classes = utils.read_to_dict(f"{vocab_path}/target_classes.json")
		self.reverse_target_classes = {value:key for key, value in self.target_classes.items()}


		self.tgt_PAD_IDX = self.target_classes["[PAD]"]
		self.output_dim = len(self.target_classes)
		self.include_lang_metadata = include_lang_metadata
		self.best_model = None
		self.input_dim = len(self.input_vocab)
		self.architecture = architecture



		# Ici on choisit quelle architecture on veut tester. À faire: CNN et RNN

		if architecture == "transformer":
			self.model = models.TransformerModel(input_dim=self.input_dim,
												 emb_dim=emb_dim,
												 num_heads=num_heads,
												 num_layers=num_transformers_layers,
												 output_dim=self.output_dim,
												 num_langs=len(self.lang_vocab),
												 lang_emb_dim=lang_emb_dim,
												 include_lang_metadata=True,
												 device=device,
												 linear_layers=linear_layers,
												 linear_layers_hidden_size=linear_layers_hidden_size,
												 load_pretrained_embeddings=False,
												 use_bert_tokenizer=use_bert_tokenizer,
												 pretrained_weights=None)
		elif architecture == "lstm":
			self.model = models.LSTM_Encoder(input_dim=self.input_dim,
											 emb_dim=emb_dim,
											 bidirectional=bidirectional,
											 lstm_dropout=lstm_dropout,
											 positional_embeddings=False,
											 device=self.device,
											 lstm_hidden_size=lstm_hidden_size,
											 batch_size=1,
											 num_langs=len(self.lang_vocab),
											 num_lstm_layers=num_lstm_layers,
											 include_lang_metadata=include_lang_metadata,
											 out_classes=self.output_dim,
											 attention=add_attention_layer,
											 lang_emb_dim=lang_emb_dim,
											 load_pretrained_embeddings=False,
											 pretrained_weights=None,
											 linear_layers=linear_layers,
											 linear_layers_hidden_size=linear_layers_hidden_size,
											 use_bert_tokenizer=use_bert_tokenizer,
											 keep_bert_dimensions=keep_bert_dimensions,
											 linear_dropout=linear_dropout)
		elif architecture == "gru":
			self.model = models.GRU_Encoder(input_dim=self.input_dim,
											 emb_dim=emb_dim,
											 bidirectional=bidirectional,
											 dropout=dropout,
											 positional_embeddings=False,
											 device=self.device,
											 hidden_size=hidden_size,
											 batch_size=batch_size,
											 num_langs=len(self.lang_vocab),
											 num_layers=num_layers,
											 include_lang_metadata=include_lang_metadata,
											 out_classes=self.output_dim,
											 attention=add_attention_layer,
											 lang_emb_dim=lang_emb_dim,
											 load_pretrained_embeddings=False,
											 pretrained_weights=None
					)
		elif architecture == "cnn":
			self.model = models.CnnEncoder(input_dim=self.input_dim,
									   emb_dim=emb_dim,
									   dropout=dropout,
									   kernel_size=kernel_size,
									   positional_embeddings=positional_embeddings,
									   device=self.device,
									   hidden_size=hidden_size,
									   num_langs=len(self.lang_vocab),
									   num_conv_layers=num_cnn_layers,
									   include_lang_metadata=include_lang_metadata,
									   out_classes=self.output_dim,
									   attention=add_attention_layer,
									   lang_emb_dim=lang_emb_dim,
									   load_pretrained_embeddings=False,
										use_bert_tokenizer=use_bert_tokenizer,
										linear_layers_hidden_size=linear_layers_hidden_size,
										linear_layers=linear_layers,
									   pretrained_weights=None,
									   cnn_scale=cnn_scale,
									   keep_bert_dimensions=keep_bert_dimensions,
									   linear_dropout=linear_dropout
									   )
		elif architecture in ["BERT", "DISTILBERT"]:
			from transformers import AutoModelForTokenClassification
			with torch.no_grad():
				print("Warning, BERT tokenizer not fully implemented yet")
				self.model = AutoModelForTokenClassification.from_pretrained(base_model_name, num_labels=3)
		if not use_safetensors:
			with torch.no_grad():
				self.model.load_state_dict(torch.load(self.saved_model, map_location=torch.device(self.device)))
		else:
			from safetensors import safe_open
			params = {}
			with safe_open(self.saved_model, framework="pt", device=self.device) as f:
				for k in f.keys():
					params[k] = f.get_tensor(k)
			self.model.load_state_dict(params)
		self.model.to(self.device)


	def tag(self, data:str, lang:str, read_file:bool=False) -> list[str]:
		"""
		The main tagging function. Takes a text as string and the lang, returns a list of segments.
		Parameters:
			data: str: the text to be tagged, as a path or a string.
			lang: str: the language code of the text.
			read_file: whether to read the file to a string or to process a string directly
		"""
		if read_file:
			with open(data, "r") as f:
				data = f.read()
		if lang not in self.lang_vocab:
			print("Lang should be represented as in the lang vocabulary json file. Please check its encoding")
			print("Representing lang as [UNK]. Results might be unsatisfactory.")
			lang = "la"
		segmented = []
		data = utils.format_examples(text=data,
								 tokens_per_example=100,
								 regexp=self.tokens_regexp,
								 lang=lang)
		for formatted_example in tqdm.tqdm(data):
			example, lang = formatted_example
			example_as_words = [item for item in re.split(self.tokens_regexp, example) if item]
			if self.architecture in ["BERT", "DISTILBERT"] or self.use_bert_tokenizer or self.use_pretrained_embeddings:
				# In the case we use a BERT subword tokenizer
				tokenized = self.tokenizer(example, truncation=True, padding="max_length", return_tensors="pt", max_length=380)
				if self.architecture in ["BERT", "DISTILBERT"]:
					tokenized_inputs = tokenized['input_ids'].to(self.device)
					masks = tokenized['attention_mask'].to(self.device)
					preds = self.model(input_ids=tokenized_inputs, attention_mask=masks).logits.tolist()
				else:
					lang = torch.tensor([self.lang_vocab[lang]]).to(self.device)
					preds = self.model(src=tokenized['input_ids'], lang=lang).tolist()

				# On convertit les tokens
				bert_labels = utils.get_labels_from_preds(preds)
				human_to_bert, bert_to_human = utils.get_correspondence(example_as_words, self.tokenizer)
				tokenized = utils.unalign_labels(human_to_bert=human_to_bert, predicted_labels=bert_labels,
											  splitted_text=example_as_words)
				segmented.extend(tokenized)

			else:
				# In the case we use a homemade tokenizer
				tokenized, lang = self.tokenize(example, lang)
				tokenized = torch.tensor(tokenized).to(self.device)
				lang = torch.tensor(lang).to(self.device)
				with torch.no_grad():
					preds = self.model(tokenized, lang)
				as_labels = utils.get_labels_from_preds(preds)
				segmented_text = utils.apply_labels(example_as_words, as_labels)
				segmented.extend(segmented_text)
		return segmented

	def tokenize(self, example, lang, debug=True) -> tuple:
		"""
		This function takes the targets and creates the examples.
		"""
		as_tokens = [item for item in re.split(self.tokens_regexp, example) if item]
		as_tokens_ids = []
		for token in as_tokens:
			try:
				as_tokens_ids.append(self.input_vocab[token.lower()])
			except KeyError:
				as_tokens_ids.append(self.input_vocab["[UNK]"])
		lang = [self.lang_vocab[lang]]
		ids = np.asarray([as_tokens_ids])
		return ids, lang



if __name__ == '__main__':
	model_dir = sys.argv[1]
	with open("data/DeRegiminePrincipum/cat_3_3_11.txt", "r") as input_txt:
		text_as_string = input_txt.read()
	lang = "ca"
	tagger = Tagger(model_dir=model_dir)
	segmented_text = tagger.tag(text_as_string, lang)

	with open(f"/home/mgl/Documents/output.txt", "w") as output:
		output.write("\n".join(segmented_text))