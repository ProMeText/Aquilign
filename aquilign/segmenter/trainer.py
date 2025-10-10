import copy
import random
import sys
import json
import argparse
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="train",
					help="Mode (test, train)")
parser.add_argument("-md", "--model", default="train",
 						help="Model path")
parser.add_argument("-a", "--architecture", default="lstm",
					help="Architecture to be tested")
parser.add_argument("-p", "--parameters", default=None,
					help="Path to parameters file")
parser.add_argument("-d", "--debug", default=False,
					help="Debug mode")
parser.add_argument("-n", "--out_name", default="",
					help="Debug mode")
args = parser.parse_args()
architecture = args.architecture
parameters = args.parameters
mode = args.mode
model = args.model
debug = args.debug
out_dir_suffix = args.out_name
with open(parameters, "r") as input_json:
	config_file = json.load(input_json)

# Gestion des imports
if config_file["global"]["import"] != "":
	sys.path.append(config_file["global"]["import"])

from transformers import AutoTokenizer, BertTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, set_seed, TrainerCallback, EarlyStoppingCallback
import aquilign.segmenter.utils as utils
import aquilign.segmenter.models as models
import aquilign.segmenter.eval as eval
import aquilign.segmenter.datafy as datafy
import torch
import datetime
from torch.utils.data import DataLoader
import tqdm
import os
import glob
import shutil

class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every):
        self.save_every = save_every

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.save_every == 0:
            control.should_save = True  # Forces saving
        else:
            control.should_save = False  # Skips saving



class SegmenterTrainer:
	def  __init__(self,
				  config_file,
				  out_dir_suffix):
		"""
		Main Class trainer
		"""
		self.date_hour = datetime.datetime.now().isoformat()
		self.debug = debug
		fine_tune = False
		epochs = config_file["global"]["epochs"]
		batch_size = config_file["global"]["batch_size"]
		lr = config_file["global"]["lr"]
		device = config_file["global"]["device"]
		workers = config_file["global"]["workers"]
		train_path = config_file["global"]["train"]
		test_path = config_file["global"]["test"]
		dev_path = config_file["global"]["dev"]
		output_dir = config_file["global"]["out_dir"]
		base_model_name = config_file["global"]["base_model_name"]
		use_pretrained_embeddings = config_file["global"]["use_pretrained_embeddings"]
		use_bert_tokenizer = config_file["global"]["use_bert_tokenizer"]
		if use_pretrained_embeddings or use_bert_tokenizer or "BERT" in architecture or "SaT" in architecture:
			os.environ["TOKENIZERS_PARALLELISM"] = "false"
		data_augmentation = config_file["global"]["data_augmentation"]
		self.freeze_embeddings = config_file["global"]["freeze_embeddings"]
		self.freeze_lang_embeddings = config_file["global"]["freeze_lang_embeddings"]
		self.balance_class_weights = config_file["global"]["balance_class_weights"]
		self.eval_batch_size = config_file["global"]["eval_batch_size"]
		include_lang_metadata = config_file["global"]["include_lang_metadata"]
		lang_emb_dim = config_file["global"]["lang_emb_dim"]
		linear_layers = config_file["global"]["linear_layers"]
		linear_layers_hidden_size = config_file["global"]["linear_layers_hidden_size"]
		self.segments_max_length = config_file["global"]["segments_max_length"]
		emb_dim = config_file["global"]["emb_dim"]
		if architecture == "lstm":
			self.use_char_embeddings = config_file["global"]["use_char_embeddings"]
			add_attention_layer = config_file["architectures"][architecture]["add_attention_layer"]
			lstm_hidden_size = config_file["architectures"][architecture]["lstm_hidden_size"]
			num_lstm_layers = config_file["architectures"][architecture]["num_lstm_layers"]
			lstm_dropout = config_file["architectures"][architecture]["lstm_dropout"]
			linear_dropout = config_file["architectures"][architecture]["linear_dropout"]
			char_dropout_prob = config_file["architectures"][architecture]["char_dropout_prob"]
			char_embedding_dim = config_file["architectures"][architecture]["char_embedding_dim"]
			bidirectional = config_file["architectures"][architecture]["bidirectional"]
			keep_bert_dimensions = config_file["architectures"][architecture]["keep_bert_dimensions"]
		if architecture == "Baseline":
			add_attention_layer = config_file["architectures"][architecture]["add_attention_layer"]
			linear_dropout = config_file["architectures"][architecture]["linear_dropout"]
			keep_bert_dimensions = config_file["architectures"][architecture]["keep_bert_dimensions"]
		elif architecture == "gru":
			add_attention_layer = config_file["architectures"][architecture]["add_attention_layer"]
			hidden_size = config_file["architectures"][architecture]["hidden_size"]
			num_layers = config_file["architectures"][architecture]["num_layers"]
			dropout = config_file["architectures"][architecture]["dropout"]
			bidirectional = config_file["architectures"][architecture]["bidirectional"]
		elif architecture == "transformer":
			num_heads = config_file["architectures"][architecture]["num_heads"]
			num_transformers_layers = config_file["architectures"][architecture]["num_transformers_layers"]
		elif architecture == "cnn":
			dropout = config_file["architectures"][architecture]["dropout"]
			cnn_scale = config_file["architectures"][architecture]["cnn_scale"]
			add_attention_layer = config_file["architectures"][architecture]["add_attention_layer"]
			keep_bert_dimensions = config_file["architectures"][architecture]["keep_bert_dimensions"]
			hidden_size = config_file["architectures"][architecture]["hidden_size"]
			kernel_size = config_file['architectures'][architecture]["kernel_size"]
			linear_dropout = config_file["architectures"][architecture]["linear_dropout"]
			positional_embeddings = config_file['architectures'][architecture]["positional_embeddings"]
			num_heads = config_file["architectures"][architecture]["num_heads"]
			num_cnn_layers = config_file["architectures"][architecture]["num_cnn_layers"]



		# First we prepare the corpus
		self.device = device
		if self.device != "cpu":
			device_name = torch.cuda.get_device_name(self.device)
			print(f"Device name: {device_name}")
		self.workers = workers
		self.all_dataset_on_device = False
		print("Loading data")
		self.use_bert_tokenizer = use_bert_tokenizer
		if use_pretrained_embeddings or "BERT" in architecture or self.use_bert_tokenizer or "SaT" in architecture:
			create_vocab = False
			self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
		else:
			create_vocab = True
			self.tokenizer = None

		self.train_path = train_path
		self.test_path = test_path
		self.dev_path = dev_path
		self.fine_tune = fine_tune
		if out_dir_suffix != "":
			out_dir_suffix = f"_{out_dir_suffix}"
		self.output_dir = output_dir + f"/{self.date_hour}" + out_dir_suffix
		self.logs_dir = f"{self.output_dir}/logs"
		self.vocab_dir = f"{self.output_dir}/vocab"
		self.config_dir = f"{self.output_dir}/config"
		os.makedirs(self.config_dir, exist_ok=True)
		out_conf_dict = copy.deepcopy(config_file)
		if "BERT" not in architecture and "SaT" not in architecture:
			out_conf_dict["architecture"] = out_conf_dict["architectures"][architecture]
			out_conf_dict["architecture"]["name"] = architecture
		else:
			out_conf_dict["architecture"] = {"name": architecture}
		out_conf_dict["global"]["model_path"] = "models/best/best.pt"
		utils.serialize_dict(out_conf_dict, f"{self.config_dir}/config.json")
		os.makedirs(self.logs_dir, exist_ok=True)
		os.makedirs(self.vocab_dir, exist_ok=True)
		self.use_pretrained_embeddings = use_pretrained_embeddings
		self.base_model_name = base_model_name


		self.data_augmentation = data_augmentation
		print(f"Creating vocab is {create_vocab}")
		if "BERT" not in architecture and "SaT" not in architecture:
			self.train_dataloader = datafy.CustomTextDataset("train",
														train_path=train_path,
														test_path=test_path,
														dev_path=dev_path,
														delimiter="£",
														output_dir=self.output_dir,
														create_vocab=create_vocab,
														use_pretrained_embeddings=use_pretrained_embeddings,
														debug=self.debug,
														data_augmentation=self.data_augmentation,
														tokenizer_name=base_model_name,
														use_bert_tokenizer=use_bert_tokenizer,
													    use_char_embeddings=self.use_char_embeddings,
														architecture=architecture)
			self.test_dataloader = datafy.CustomTextDataset(mode="test",
													   train_path=train_path,
													   test_path=test_path,
														dev_path=dev_path,
													   delimiter="£",
													   output_dir=self.output_dir,
													   create_vocab=False,
													   input_vocab=self.train_dataloader.datafy.input_vocabulary,
													   lang_vocab=self.train_dataloader.datafy.lang_vocabulary,
														use_pretrained_embeddings=use_pretrained_embeddings,
														debug=self.debug,
														data_augmentation=self.data_augmentation,
														tokenizer_name=base_model_name,
														use_bert_tokenizer=use_bert_tokenizer,
													    use_char_embeddings=self.use_char_embeddings,
														   architecture=architecture)

			self.dev_dataloader = datafy.CustomTextDataset(mode="dev",
													   train_path=train_path,
													   test_path=test_path,
														dev_path=dev_path,
													   delimiter="£",
													   output_dir=self.output_dir,
													   create_vocab=False,
													   input_vocab=self.train_dataloader.datafy.input_vocabulary,
													   lang_vocab=self.train_dataloader.datafy.lang_vocabulary,
														use_pretrained_embeddings=use_pretrained_embeddings,
														debug=self.debug,
														data_augmentation=self.data_augmentation,
														tokenizer_name=base_model_name,
														use_bert_tokenizer=use_bert_tokenizer,
													    use_char_embeddings=self.use_char_embeddings,
														   architecture=architecture)

			self.loaded_test_data = DataLoader(self.test_dataloader,
											   batch_size=batch_size,
											   shuffle=False,
											   num_workers=self.workers,
											   pin_memory=False,
											   drop_last=True)
			self.loaded_train_data = DataLoader(self.train_dataloader,
												batch_size=batch_size,
												shuffle=True,
												num_workers=self.workers,
												pin_memory=False,
											   drop_last=True)
			self.loaded_dev_data = DataLoader(self.dev_dataloader,
												batch_size=batch_size,
												shuffle=True,
												num_workers=self.workers,
												pin_memory=False,
											   drop_last=True)

			print(f"Number of train examples: {len(self.train_dataloader.datafy.train_padded_examples)}")
			print(f"Number of test examples: {len(self.test_dataloader.datafy.test_padded_examples)}")
			print(f"Total length of examples (with padding): {self.train_dataloader.datafy.max_length_examples}")
			self.lang_vocab = self.train_dataloader.datafy.lang_vocabulary
			self.target_classes = self.train_dataloader.datafy.target_classes
			self.reverse_target_classes = self.train_dataloader.datafy.reverse_target_classes

			self.corpus_size = self.train_dataloader.__len__()
			self.steps = self.corpus_size // batch_size

			self.test_steps = self.test_dataloader.__len__() // batch_size
			# Ici on choisit quelle architecture on veut tester. À faire: CNN et RNN
			if self.use_pretrained_embeddings:
				weights = torch.load("aquilign/segmenter/embeddings.npy")
			else:
				weights = None

			self.tgt_PAD_IDX = self.target_classes["[PAD]"]
			self.output_dim = len(self.target_classes)

			if self.balance_class_weights:
				self.train_dataloader.datafy.deduce_weights(weight_factor=2)
				weights = self.train_dataloader.datafy.target_weights.to(self.device)
				self.criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=self.tgt_PAD_IDX)
			else:
				self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_PAD_IDX)

		# If we use bert
		else:
			train_lines = utils.json_corpus_to_lines(train_path, keep_punct=True)
			dev_lines = utils.json_corpus_to_lines(dev_path, keep_punct=True)
			eval_lines, delimiter = utils.json_corpus_to_lines(test_path, keep_punct=True, return_delimiter=True)
			if self.data_augmentation:
				train_lines = utils.augment_data([train_lines])
			print(self.tokenizer)
			train_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(train_lines, tokenizer=self.tokenizer,
																			   delimiter=delimiter)

			self.train_dataset = utils.SentenceBoundaryDataset(train_texts_and_labels)
			assert self.train_dataset.texts_and_labels is not None, f"Error with dataset production: {self.train_dataset.texts_and_labels}"

			# Dev corpus
			print("Dev corpus preparation")
			dev_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(dev_lines, tokenizer=self.tokenizer,
																			 delimiter=delimiter)
			self.dev_dataset = utils.SentenceBoundaryDataset(dev_texts_and_labels)


			# Dev corpus
			print("Test corpus preparation")
			test_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(eval_lines, tokenizer=self.tokenizer,
																			 delimiter=delimiter)
			self.test_data = utils.SentenceBoundaryDataset(test_texts_and_labels)

			self.loaded_test_data = DataLoader(self.test_data,
											   batch_size=self.eval_batch_size,
											   shuffle=False,
											   num_workers=self.workers,
											   pin_memory=False,
											   drop_last=True)

			self.target_classes = {"[SC]": 0,  # Segment content > no split
								   "[SB]": 1,  # Segment boundary > split before
								   "[PAD]": 2
								   }
			self.reverse_target_classes = {id: label for label, id in self.target_classes.items()}
			self.tgt_PAD_IDX = self.target_classes["[PAD]"]



		os.makedirs(f"{self.output_dir}/models/.tmp", exist_ok=True)

		if "BERT" in architecture or "SaT" in architecture or self.use_pretrained_embeddings or self.use_bert_tokenizer:
			self.input_vocab = self.tokenizer.get_vocab()
		else:
			self.input_vocab = self.train_dataloader.datafy.input_vocabulary
		assert self.input_vocab != {}, "Error with input vocabulary"
		utils.serialize_dict(self.input_vocab, f"{self.vocab_dir}/input_vocab.json")
		self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}


		self.epochs = epochs
		self.batch_size = batch_size
		self.include_lang_metadata = include_lang_metadata
		self.best_model = None
		self.input_dim = len(self.input_vocab)
		self.architecture = architecture

		self.epochs_log_file = f"{self.logs_dir}/train_logs.txt"
		self.final_results_file = f"{self.logs_dir}/best_model.txt"
		utils.remove_files(
			[self.epochs_log_file, self.final_results_file]
		)




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
												 load_pretrained_embeddings=use_pretrained_embeddings,
												 use_bert_tokenizer=use_bert_tokenizer,
												 pretrained_weights=weights)
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
											 load_pretrained_embeddings=use_pretrained_embeddings,
											 pretrained_weights=weights,
											 linear_layers=linear_layers,
											 linear_layers_hidden_size=linear_layers_hidden_size,
											 use_bert_tokenizer=use_bert_tokenizer,
											 keep_bert_dimensions=keep_bert_dimensions,
											 use_character_embeddings=self.use_char_embeddings,
											 linear_dropout=linear_dropout,
											 char_dropout_prob=char_dropout_prob,
											 char_embedding_dim=char_embedding_dim)
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
											 load_pretrained_embeddings=use_pretrained_embeddings,
											 pretrained_weights=weights
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
									   load_pretrained_embeddings=use_pretrained_embeddings,
										use_bert_tokenizer=use_bert_tokenizer,
										linear_layers_hidden_size=linear_layers_hidden_size,
										linear_layers=linear_layers,
									   pretrained_weights=weights,
									   cnn_scale=cnn_scale,
									   keep_bert_dimensions=keep_bert_dimensions,
									   linear_dropout=linear_dropout
									   )

		elif architecture == "Baseline":
			self.model = models.BaseLineModel(input_dim=self.input_dim,
											 emb_dim=emb_dim,
											 positional_embeddings=False,
											 device=self.device,
											 batch_size=batch_size,
											 num_langs=len(self.lang_vocab),
											 include_lang_metadata=include_lang_metadata,
											 out_classes=self.output_dim,
											 attention=add_attention_layer,
											 lang_emb_dim=lang_emb_dim,
											 load_pretrained_embeddings=use_pretrained_embeddings,
											 pretrained_weights=weights,
											 linear_layers=linear_layers,
											 linear_layers_hidden_size=linear_layers_hidden_size,
											 use_bert_tokenizer=use_bert_tokenizer,
											 keep_bert_dimensions=keep_bert_dimensions,
											 linear_dropout=linear_dropout)
		else:
			if architecture in ["BERT", "TinyBERT"]:
				from transformers import AutoModelForTokenClassification
				self.model = AutoModelForTokenClassification.from_pretrained(base_model_name, num_labels=3)
			elif architecture == "DistilBERT":
				from transformers import DistilBertTokenizer, DistilBertForTokenClassification
				self.tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
				self.model = DistilBertForTokenClassification.from_pretrained(base_model_name, num_labels=3)
			elif architecture == "SaT":
				import aquilign.segmenter.sat_models as SaT
				config = SaT.SubwordXLMConfig.from_pretrained(base_model_name)
				config.num_labels = 3
				config.num_hidden_layers = 12
				config.lookahead = 48
				config.lookahead_split_layers = 6
				self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
				self.model = SaT.SubwordXLMForTokenClassification.from_pretrained(base_model_name, config=config)
				print("SaT model loaded.")
		self.architecture = architecture
		self.model.to(self.device)
		self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
		self.use_pretrained_embeddings = use_pretrained_embeddings

		# Les classes étant distribuées de façons déséquilibrée, on donne + d'importance à la classe <SB>
		# qu'aux deux autres pour le calcul de la loss. On désactive pour l'instant
		print(self.model.__repr__())
		self.accuracies = []
		self.results = []

	def save_model(self, epoch):
		torch.save(self.model.state_dict(), f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{epoch}.pt")
		print(f"Model saved to {self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{epoch}.pt")


	def get_best_model(self):
		"""
		We choose the best model based on a weighted average of precision and recall.
		"""
		f1_averages = []
		weighted_averages = []
		for result in self.results:
			recall = result["recall"][1]
			precision = result["precision"][1]
			f1 = result["f1"][1]
			weighted = (precision + (recall*1.3) ) / 2.3
			weighted_averages.append(weighted.item())
			f1_averages.append(f1.item())

		max_average = max(weighted_averages)
		best_epoch = weighted_averages.index(max_average)
		message = f"Best model: {best_epoch} with {max_average} weighted precision and recall on dev data."
		utils.append_to_file(message, self.final_results_file)
		print(message)
		models = glob.glob(f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_*.pt")
		try:
			os.mkdir(f"{self.output_dir}/models/best")
		except OSError:
			pass
		model_found = False
		for model in models:
			if model == f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{best_epoch}.pt":
				model_found = True
				shutil.copy(model, f"{self.output_dir}/models/best/best.pt")
				print(f"Saving best model to {self.output_dir}/models/best/best.pt")
			os.remove(model)
		assert model_found is True, "No best model found. Somethign went wrong."
		self.best_model = f"{self.output_dir}/models/best/best.pt"

	def Bert_Train(self):

		training_args = TrainingArguments(
			output_dir=f"results_{self.output_dir}/epoch{self.epochs}_bs{self.batch_size}",
			num_train_epochs=self.epochs,
			logging_strategy="epoch",
			per_device_train_batch_size=self.batch_size,
			per_device_eval_batch_size=self.batch_size,
			evaluation_strategy="epoch",
			dataloader_num_workers=8,
			dataloader_prefetch_factor=4,
			bf16=True,
			use_cpu=self.device == "cpu",
			save_strategy="epoch",
			load_best_model_at_end=True
			#best model is evaluated on loss
		)
		self.trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=self.train_dataset,
			eval_dataset=self.dev_dataset,
			compute_metrics=eval.compute_metrics,
			callbacks=[SaveEveryNEpochsCallback(save_every=1),
					   EarlyStoppingCallback(early_stopping_patience=5)]

		)

		print("Starting training")
		self.trainer.train()
		print("End of training")

	def train(self, clip=0.1):
		# Ici on va faire en sorte que les plongements de mots ne soient pas entraînables, si c'est des plongements pré-entraînés
		# Une possibilité serait de dégeler les paramètres en fin d'entraînement, quelques epochs avant la fin (3-4?)
		if self.use_pretrained_embeddings:
			if self.freeze_embeddings:
				for param in self.model.embedding.parameters():
					param.requires_grad = False

		# Idem pour les plongements de langue.
		if self.include_lang_metadata:
			if self.freeze_lang_embeddings:
				for param in self.model.lang_embedding.parameters():
					param.requires_grad = False
		utils.remove_file(f"{self.output_dir}/accuracies.txt")
		utils.remove_files(f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_*.pt")
		print("Starting training")
		os.makedirs(f"{self.output_dir}/models/best", exist_ok=True)
		torch.save(self.input_vocab, f"{self.output_dir}/models/best/vocab.voc")
		print("Evaluating randomly initiated model")
		recall, precision, f1 = self.evaluate(last_epoch=False)
		utils.append_to_file(
			f"Randomly initiated model:\n" +
			utils.format_results(
				results=[precision, recall, f1], header=["", "Segment Content", "Segment Boundary"], print_to_term=False
			),
			self.epochs_log_file
		)
		utils.append_to_file("---", self.epochs_log_file)
		for epoch in range(self.epochs):
			self.model.train()
			epoch_number = epoch
			last_epoch = epoch == range(self.epochs)[-1]
			print(f"Epoch {str(epoch_number)}")
			for data in tqdm.tqdm(self.loaded_train_data, unit_scale=self.batch_size):
				if "BERT" in self.architecture:
					examples, masks, targets = data
					masks = masks.to(self.device)
				else:
					examples, langs, targets = data
					langs = langs.to(self.device)
				examples = examples.to(self.device)
				targets = targets.to(self.device)
				# Shape [batch_size, max_length]
				# tensor_examples = examples.to(device)
				# Shape [batch_size, max_length]
				# tensor_targets = targets.to(device)

				# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
				# for param in model.parameters():
				# param.grad = None
				self.optimizer.zero_grad()
				# Shape [batch_size, max_length, output_dim]
				if "BERT" not in self.architecture:
					output = self.model(examples, langs)
					output = output.view(-1, self.output_dim)
					tgt = targets.view(-1)
					loss = self.criterion(output, tgt)
				else:
					output = self.model(input_ids=examples, attention_mask=masks, labels=targets)
					loss = output.loss
				# output_dim = output.shape[-1]
				# Shape [batch_size*max_length, output_dim]
				# Shape [batch_size*max_length]

				# output = [batch size * tgt len - 1, output dim]
				# tgt = [batch size * tgt len - 1]

				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
				self.optimizer.step()

			# self.model.eval()
			self.scheduler.step()
			recall, precision, f1 = self.evaluate()
			utils.append_to_file(
				f"Epoch: {str(epoch_number)}\n" +
				utils.format_results(
				results=[precision, recall, f1], header=["", "Segment Content", "Segment Boundary"], print_to_term=False
				),
				self.epochs_log_file
			)
			utils.append_to_file("---", self.epochs_log_file)
			self.save_model(epoch_number)
		self.get_best_model()
		self.evaluate_best_model()
		print(f"End of training. Results written to {self.final_results_file}")
		# self.evaluate_best_model_per_lang()

	def evaluate_best_model(self, max_length=None):
		"""
				Cette fonction produit les métriques d'évaluation (justesse, précision, rappel)
				"""
		print("Evaluating best model on test data")
		all_preds = []
		all_targets = []
		all_examples = []
		eval_device = self.device
		if "BERT" in self.architecture:
			best_model_path = self.trainer.state.best_model_checkpoint
			self.model = AutoModelForTokenClassification.from_pretrained(best_model_path, num_labels=3)
		else:
			self.model.load_state_dict(torch.load(self.best_model, weights_only=True))
		print("Model loaded.")
		self.model.to(eval_device)
		self.model.eval()
		print("Starting evaluation")
		for data in tqdm.tqdm(self.loaded_test_data, unit_scale=self.eval_batch_size):
			if "BERT" in self.architecture:
				examples, masks, targets = data['input_ids'], data['attention_mask'], data['labels']
				masks = masks.to(eval_device)
			else:
				examples, langs, targets = data
				langs = langs.to(eval_device)
			examples = examples.to(eval_device)
			with torch.no_grad():
				# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
				if "BERT" not in self.architecture:
					preds = self.model(examples, langs)
				else:
					emissions = self.model(input_ids=examples, attention_mask=masks).logits
					C = emissions.size(-1)
					device = "cpu"
					emissions = emissions.to(device)
					masks = masks.to(device)
					transitions = torch.zeros(C, C, device=device)
					start_transitions = torch.zeros(C, device=device)
					end_transitions = torch.zeros(C, device=device)
					mask = data["attention_mask"].bool()
					L_O, L_B, L_I = 2, 1, 0
					if max_length is None:
						preds = emissions
					else:
						preds = utils.constrained_viterbi(emissions,
														  transitions,
														  start_transitions,
														  end_transitions,
														  mask,
														  device,
														  ideal_segments_length=max_length,
														  L_O=L_O,
														  L_B=L_B,
														  L_I=L_I)
				all_preds.append(preds)
				all_targets.append(targets)
				examples = examples.to(self.device)
				all_examples.append(examples)
		# On crée une seul vecteur, en concaténant tous les exemples sur la dimension 0 (= chaque exemple individuel)
		cat_preds = torch.cat(all_preds, dim=0)
		cat_targets = torch.cat(all_targets, dim=0)
		cat_examples = torch.cat(all_examples, dim=0)
		# eval.compute_ambiguity_metrics(tokens=cat_examples,
		# 										   labels=cat_targets,
		# 										   predictions=cat_preds,
		# 										   id_to_word=self.reverse_input_vocab,
		# 										   word_to_id=self.input_vocab,
		# 										   log_dir = self.logs_dir,
		# 										   name="global")

		results = eval.compute_metrics(predictions=cat_preds,
									   labels=cat_targets,
									   examples=cat_examples,
									   id_to_word=self.reverse_input_vocab,
									   # idx_to_class=self.reverse_target_classes,
									   # padding_idx=self.tgt_PAD_IDX,
									   # batch_size=self.batch_size,
									   last_epoch=True,
									   bert_training=False,
									   tokenizer=self.tokenizer,
									   log_file=self.final_results_file)



		recall = ["Recall", results["recall"][0], results["recall"][1]]
		precision = ["Precision", results["precision"][0], results["precision"][1]]
		f1 = ["F1", results["f1"][0], results["f1"][1]]
		header = ["", "Segment Content", "Segment Boundary"]
		print(f"Results for all langs:")
		utils.format_results(results=[precision, recall, f1], header=header)
		utils.append_to_file(
			"Best model on test data\n" +
			utils.format_results(
				results=[precision, recall, f1], header=["", "Segment Content", "Segment Boundary"], print_to_term=False
			),
		self.final_results_file)

	def evaluate_best_model_per_lang(self):
		"""
				Cette fonction produit les métriques d'évaluation (justesse, précision, rappel)
				"""
		print("Evaluating best model on test data")

		# On crée un dernier dataloader: un dictionnaire avec division des langues pour avoir des résultats par langue.
		loaded_test_data_per_lang = {}
		# We change the batch size for really small sub-corpuses (ex. english for now)
		for lang in self.lang_vocab:
			if lang == "[UNK]":
				continue
			if self.architecture not in "BERT":
				current_dataloader = datafy.CustomTextDataset(mode="test",
															  train_path=self.train_path,
															  test_path=self.test_path,
															  dev_path=self.dev_path,
															  device=self.device,
															  delimiter="£",
															  output_dir=self.output_dir,
															  create_vocab=False,
															  input_vocab=self.train_dataloader.datafy.input_vocabulary,
															  lang_vocab=self.train_dataloader.datafy.lang_vocabulary,
															  use_pretrained_embeddings=self.use_pretrained_embeddings,
															  debug=self.debug,
															  data_augmentation=self.data_augmentation,
															  filter_by_lang=lang,
															 tokenizer_name=self.base_model_name,
															  architecture=self.architecture,
															  use_bert_tokenizer=self.use_bert_tokenizer)
				loaded_test_data_per_lang[lang] = DataLoader(current_dataloader,
																  batch_size=self.batch_size,
																  shuffle=False,
																  num_workers=self.workers,
																  pin_memory=False,
																  drop_last=True)
				self.model.load_state_dict(torch.load(self.best_model, weights_only=True))
			else:

				test_lines, delimiter = utils.json_corpus_to_lines(self.test_path, keep_punct=True, return_delimiter=True)
				text_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(test_lines,
																				   tokenizer=self.tokenizer,
																				   delimiter=delimiter)
				self.test_dataset = utils.SentenceBoundaryDataset(text_texts_and_labels)
				self.model = AutoModelForTokenClassification.from_pretrained(self.best_model_name)



		self.model.eval()
		results_per_lang = {}
		for lang in self.lang_vocab:
			if lang == "[UNK]":
				continue
			print(f"Testing {lang}")
			all_preds = []
			all_targets = []
			all_examples = []
			for data in tqdm.tqdm(loaded_test_data_per_lang[lang], unit_scale=self.batch_size):
				if "BERT" in self.architecture:
					examples, masks, targets = data
					masks = masks.to(self.device)
				else:
					examples, langs, targets = data
					langs = langs.to(self.device)
				examples = examples.to(self.device)
				targets = targets.to(self.device)
				with torch.no_grad():
					# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
					if self.architecture != "BERT":
						preds = self.model(examples, langs)
					else:
						preds = self.model(input_ids=examples, attention_mask=masks, labels=targets).logits
					all_preds.append(preds)
					all_targets.append(targets)
					all_examples.append(examples)

			# On crée une seul vecteur, en concaténant tous les exemples sur la dimension 0 (= chaque exemple individuel)
			try:
				cat_preds = torch.cat(all_preds, dim=0)
			except RuntimeError:
				print(f"Not enough data for lang {lang}")
				continue
			cat_targets = torch.cat(all_targets, dim=0)
			cat_examples = torch.cat(all_examples, dim=0)
			eval.compute_ambiguity_metrics(tokens=cat_examples,
			 										   labels=cat_targets,
			 										   predictions=cat_preds,
			 										   id_to_word=self.reverse_input_vocab,
			 										   word_to_id=self.input_vocab,
			 										   log_dir=self.logs_dir,
			 										   name=lang)
			results = eval.compute_metrics(predictions=cat_preds,
										   labels=cat_targets,
										   examples=cat_examples,
										   id_to_word=self.reverse_input_vocab,
										   last_epoch=False,
										   bert_training=False,
										   tokenizer=self.tokenizer)
			results_per_lang[lang] = results

			recall = ["Recall", results["recall"][0], results["recall"][1]]
			precision = ["Precision", results["precision"][0], results["precision"][1]]
			f1 = ["F1", results["f1"][0], results["f1"][1]]
			header = ["", "Segment Content", "Segment Boundary"]
			print(f"Results for {lang}:")
			utils.format_results(results=[precision, recall, f1], header=header)
			utils.append_to_file(
				f"Best model on test data for {lang}:\n" +
				utils.format_results(
					results=[precision, recall, f1], header=["", "Segment Content", "Segment Boundary"], print_to_term=False
				),
				self.final_results_file.replace(".txt", f".{lang}.txt"))




	def evaluate(self, loss_calculation:bool=False, last_epoch:bool=False):
		"""
		Cette fonction produit les métriques d'évaluation (justesse, précision, rappel)
		"""
		print("Evaluating model on dev data")
		debug = False
		epoch_accuracy = []
		epoch_loss = []
		# Timer = utils.Timer()
		all_preds = []
		all_targets = []
		all_examples = []
		self.model.eval()
		with torch.no_grad():
			for data in tqdm.tqdm(self.loaded_dev_data, unit_scale=self.batch_size):
				if "BERT" in self.architecture:
					examples, masks, targets = data['input_ids'], data['attention_mask'], data['labels']
					masks = masks.to(self.device)
				else:
					examples, langs, targets = data
					langs = langs.to(self.device)
				examples = examples.to(self.device)
				targets = targets.to(self.device)
				with torch.no_grad():
					# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
					if self.architecture != "BERT":
						preds = self.model(examples, langs)
					else:
						preds = self.model(input_ids=examples, attention_mask=masks, labels=targets).logits
					all_preds.append(preds)
					all_targets.append(targets)
					all_examples.append(examples)

		# On supprime les batchs:
		cat_preds = torch.cat(all_preds, dim=0) # [num_examples, max_dim, num_classes]
		cat_targets = torch.cat(all_targets, dim=0) # [num_examples, max_dim]
		cat_examples = torch.cat(all_examples, dim=0) # [num_examples, max_dim]
		results = eval.compute_metrics(predictions=cat_preds,
									   labels=cat_targets,
									   examples=cat_examples,
									   id_to_word=self.reverse_input_vocab,
									   # idx_to_class=self.reverse_target_classes,
									   # padding_idx=self.tgt_PAD_IDX,
									   # batch_size=self.batch_size,
									   last_epoch=last_epoch,
									   tokenizer=self.tokenizer,
									   bert_training=False)
		self.results.append(results)

		recall = ["Recall", results["recall"][0], results["recall"][1]]
		precision = ["Precision", results["precision"][0], results["precision"][1]]
		f1 = ["F1", results["f1"][0], results["f1"][1]]
		print(f"Results for all langs:")
		utils.format_results(results=[precision, recall, f1], header=["", "Segment Content", "Segment Boundary"])
		return (recall, precision, f1)



if __name__ == '__main__':
	random.seed(1234)
	trainer = SegmenterTrainer(config_file=config_file,
					  out_dir_suffix=out_dir_suffix)
	if mode != "test":
		if "BERT" in architecture or "SaT" in architecture:
			trainer.Bert_Train()
			# for i in range(trainer.segments_max_length - 5, trainer.segments_max_length + 5):
				# trainer.evaluate_best_model(max_length=i)
			# trainer.evaluate_best_model(max_length=100)
			trainer.evaluate_best_model()
		else:
			trainer.train()
	else:
		trainer.best_model = model
		trainer.evaluate_best_model_per_lang()
