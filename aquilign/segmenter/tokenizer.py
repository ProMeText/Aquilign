import re
import sys

from transformers import AutoTokenizer
import aquilign.segmenter.utils as utils
import aquilign.segmenter.models as models
import torch
import tqdm
import os


class Tagger:
	def  __init__(self,
				  model_dir,
				  batch_size):
		"""
		Main Class trainer
		"""
		config_file = utils.read_to_dict(f"{model_dir}/config/config.json")
		self.saved_model = f"{model_dir}/models/best/best.pt"
		architecture = config_file["architecture"]["name"]
		device = config_file["global"]["device"]
		workers = config_file["global"]["workers"]
		vocab_dir = config_file["global"]["vocab_dir"]
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





		self.input_vocab = utils.read_to_dict(f"{vocab_dir}/input_vocab.json")
		self.lang_vocab = utils.read_to_dict(f"{vocab_dir}/lang_vocab.json")
		assert self.input_vocab != {}, "Error with input vocabulary"
		self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}

		self.target_classes = utils.read_to_dict(f"{vocab_dir}/target_classes.json")
		self.reverse_target_classes = {value:key for key, value in self.target_classes.items()}


		self.tgt_PAD_IDX = self.target_classes["[PAD]"]
		self.batch_size = int(batch_size)
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
											 batch_size=batch_size,
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
			self.model = AutoModelForTokenClassification.from_pretrained(base_model_name, num_labels=3)

		self.model.load_state_dict(torch.load(self.saved_model, map_location=torch.device(self.device)))
		self.model.to(self.device)


	def tag(self, data:str, lang:str) -> list[str]:
		"""
		The main tagging function. Takes a text as string, returns a list of segments.
		"""
		segmented = []
		new_labels = []
		data = utils.format_examples(text=data,
								 tokens_per_example=100,
								 regexp=self.tokens_regexp,
								 lang=lang)
		for formatted_example in tqdm.tqdm(data):
			example, lang = formatted_example
			if self.architecture in ["BERT", "DISTILBERT"] or self.use_bert_tokenizer or self.use_pretrained_embeddings:
				tokenized = self.tokenizer.encode(example, truncation=True, padding=True, return_tensors="pt", max_length=380)
				if self.architecture in ["BERT", "DISTILBERT"]:
					tokenized_inputs = tokenized['input_ids'].to(self.device)
					masks = tokenized['attention_mask'].to(self.device)
					preds = self.model(input_ids=tokenized_inputs, attention_mask=masks).tolist()
				else:
					lang = torch.tensor([self.lang_vocab[lang]]).to(self.device)
					preds = self.model(src=tokenized, lang=lang).tolist()

				# On convertit les tokens
				splitted_pred = [item for item in re.split(self.tokens_regexp, example) if item]
				bert_labels = utils.get_labels_from_preds(preds)
				human_to_bert, bert_to_human = utils.get_correspondence(splitted_pred, self.tokenizer)
				labels = utils.unalign_labels(human_to_bert=human_to_bert, predicted_labels=bert_labels,
											  splitted_text=splitted_pred)
				tokenized = labels.split("\n")
				segmented.extend(tokenized)




			else:
				tokenized = self.tokenize(example).to(self.device)
				lang = lang.to(self.device)
				preds = self.model(tokenized, lang)[0]
				preds = preds.view(-1, self.output_dim)
				print(preds.shape)




		return segmented

	def tokenize(self, example, lang, debug=True) -> tuple:
		"""
		This function takes the targets and creates the examples.
		"""
		assert data != [], "Error with the data when producing the corpus"
		examples = []
		attention_masks = []
		targets = []
		langs = []
		ids = []
		text = example['example']
		lang = example['lang']
		as_tokens = [item for item in re.split(self.delimiters_regex, text) if item]
		as_tokens_ids = [self.input_vocab[token] for token in as_tokens]
		lang = self.lang_vocabulary[lang]



		# On doit convertir la liste d'arrays vers un arrays, on concatène sur la dimension 0 (lignes)
		ids = np.concatenate(ids, axis=0)
		# targets = np.concatenate(targets, axis=0)
		targets = torch.stack(targets, dim=0)
		if self.architecture in ["BERT", "DISTILBERT"]:
			return ids, attention_masks, langs, targets
		else:
			return ids, langs, targets



if __name__ == '__main__':
	model_dir = sys.argv[1]
	batch_size = int(sys.argv[2])
	with open("data/DeRegiminePrincipum/cat_3_3_11.txt", "r") as input_txt:
		text_as_string = input_txt.read()
	lang = "ca"
	Tagger = Tagger(model_dir=model_dir,
					batch_size=batch_size)
	Tagger.tag(text_as_string, lang)