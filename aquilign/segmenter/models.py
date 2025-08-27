import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import transformers


def save_bert_embeddings():
	myBertModel = transformers.BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
	word_embeddings = myBertModel.get_input_embeddings().weight.data
	# word_embeddings = myBertModel.get_input_embeddings().weight.data.half()
	torch.save(word_embeddings, "aquilign/segmenter/embeddings.npy")

class GRU_Encoder(nn.Module):
	def __init__(self,
				 input_dim,
				 emb_dim,
				 load_pretrained_embeddings,
				 pretrained_weights,
				 include_lang_metadata,
				 attention,
				 num_langs,
				 lang_emb_dim,
				 bidirectional,
				 dropout,
				 positional_embeddings,
				 device,
				 hidden_size,
				 out_classes,
				 num_layers,
				 batch_size
				 ):
		super().__init__()

		# On peut utiliser des embeddings pré-entraînés pour vérifier si ça améliore les résultats
		if load_pretrained_embeddings:
			# Hard-codé, il vaudrait mieux récupérer à partir des données des embeddings
			self.input_dim = 119547
			emb_dim = 768
			# Ici on vérifiera le paramètre _freeze
			self.embedding = torch.nn.Embedding(num_embeddings=self.input_dim, embedding_dim=emb_dim)
			# Censé initialiser les paramètres avec les poids pré-entraînés
			self.embedding.weight.data = torch.tensor(pretrained_weights)
		else:
			# Sinon on utilise l'initialisation normale
			self.embedding = nn.Embedding(input_dim, emb_dim)
		self.include_lang_metadata = include_lang_metadata
		self.attention = attention
		self.num_langs = num_langs
		self.hidden_size = hidden_size
		self.batch_size = batch_size

		# Possibilité de produire des embeddings de langue que l'on va concaténer aux plongements de mots
		if self.include_lang_metadata:
			self.lang_embedding = nn.Embedding(self.num_langs, lang_emb_dim)  # * self.scale
			# Si on concatène les embeddings, la dimension de sortie après concaténation est la somme de
			# la dimension des deux types de plongements
			input_size = emb_dim + lang_emb_dim
		else:
			input_size = emb_dim

		self.bidi = bidirectional
		# self.dropout = nn.Dropout(dropout)
		self.rnn = nn.GRU(input_size=input_size,
						  bidirectional=self.bidi,
						  batch_first=True,
						  dropout=dropout,
						  num_layers=num_layers,
						  hidden_size=self.hidden_size)
		self.positional_embeddings = positional_embeddings

		# On peut ajouter des plongements positionnels mais avec un lstm c'est probablement moins utile
		if positional_embeddings:
			self.pos1Dsum = Summer(PositionalEncoding1D(emb_dim))
		self.device = device
		self.input_dim = input_dim
		self.num_layers = num_layers

		# Une couche d'attention multitête
		if self.attention:
			if self.bidi:
				# La sortie du LSTM est doublée en taille si c'est bidirectionnel (lr et rl)
				self.multihead_attn = nn.MultiheadAttention(self.hidden_size * 2, 8)
			else:
				self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 8)

		if self.bidi:
			self.linear_layer = nn.Linear(self.hidden_size * 2, out_classes)
		else:
			self.linear_layer = nn.Linear(self.hidden_size, out_classes)


	def forward(self, src, lang):
		batch_size, seq_length = src.size()
		# On plonge le texte [batch_size, max_length, embeddings_dim]
		embedded = self.embedding(src)
		if self.include_lang_metadata:

			# Shape: [batch_size, lang_metadata_dimensions]
			lang_embedding = self.lang_embedding(lang)
			# On augmente de dimension pour pouvoir concaténer chaque token et la langue:
			# [batch_size, max_length, lang_metadata_dimensions]
			projected_lang = lang_embedding.unsqueeze(1).expand(-1, seq_length,
																-1)  # (batch_size, seq_length, embedding_dim)
			# On concatène chaque token avec le vecteur de langue, c'est-à-dire qu'on augmente la
			# dimensionnalité de chaque vecteur de mot dont la dimension sera la somme des deux dimensions:
			# [batch_size, max_length, lang_metadata_dimensions + word_embedding_dimension]
			embedded = torch.cat((embedded, projected_lang), 2)
		else:
			embedded = self.embedding(src)

		if self.positional_embeddings:
			embedded = self.pos1Dsum(embedded)  #
		if self.bidi:
			init_state = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size).to(self.device)
		else:
			init_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
		rnn_out, _ = self.rnn(embedded, init_state)


		# Attention et classification
		if self.attention:
			attn_output, _ = self.multihead_attn(rnn_out, rnn_out, rnn_out)
			outs = self.linear_layer(attn_output + rnn_out)
		else:
			outs = self.linear_layer(rnn_out)
		# dimension: [batch_size, max_length, 3] pour [SC], [SB], [PAD]
		return outs




class TransformerModel(nn.Module):
	def __init__(self,
				 input_dim:int,
				 emb_dim:int,
				 num_heads:int,
				 num_layers:int,
				 device:str,
				 output_dim:int,
				 num_langs:int,
				 lang_emb_dim:int,
				 include_lang_metadata:bool,
				 linear_layers:int,
				 linear_layers_hidden_size:int,
				 load_pretrained_embeddings:bool,
				 use_bert_tokenizer:bool,
				 pretrained_weights:np.ndarray):
		super(TransformerModel, self).__init__()

		self.num_langs = num_langs
		self.lang_emb_dim = lang_emb_dim
		self.include_lang_metadata = include_lang_metadata
		# Couche d'embedding pour transformer les entrées dans l'espace de dimension hidden_dim
		if load_pretrained_embeddings or use_bert_tokenizer:
			# Hard-codé, il vaudrait mieux récupérer à partir des données des embeddings
			self.input_dim = 119547
			emb_dim = 768
			# Ici on vérifiera le paramètre _freeze
			self.embedding = torch.nn.Embedding(num_embeddings=self.input_dim, embedding_dim=emb_dim)
			# Censé initialiser les paramètres avec les poids pré-entraînés
			self.embedding.weight.data = torch.tensor(pretrained_weights)
			print(f"Pretrained embeddings loaded dtype: {pretrained_weights.dtype}")
		else:
			# Sinon on utilise l'initialisation normale
			self.embedding = nn.Embedding(input_dim, emb_dim)

		if self.include_lang_metadata:
			self.lang_embedding = nn.Embedding(self.num_langs, lang_emb_dim)  # * self.scale
			hidden_dim = emb_dim + lang_emb_dim
		else:
			hidden_dim = emb_dim


		# Encoder du Transformer
		encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		layers = []
		if linear_layers == 1:
			layers.append(nn.Linear(hidden_dim, output_dim))
		else:
			layers.append(nn.Linear(hidden_dim, linear_layers_hidden_size))
			layers.append(nn.ReLU())
			for layer in range(linear_layers):
				if layer != linear_layers - 2:
					layers.append(nn.Linear(linear_layers_hidden_size, linear_layers_hidden_size))
					layers.append(nn.ReLU())
				else:
					layers.append(nn.Linear(linear_layers_hidden_size, output_dim))
					break
		self.layers = nn.Sequential(*layers)

	def forward(self, src, lang):
		# x : (batch_size, seq_length, input_dim)
		batch_size, seq_length = src.size()
		# Passer à travers la couche d'embedding
		src = self.embedding(src)  # (batch_size, seq_length, hidden_dim)
		if self.include_lang_metadata:
			lang_embedding = self.lang_embedding(lang)
			projected_lang = lang_embedding.unsqueeze(1).expand(-1, seq_length,
																-1)
			src = torch.cat((src, projected_lang), 2)

		# Transposer pour avoir (seq_length, batch_size, hidden_dim)
		src = src.permute(1, 0, 2)

		# Passer à travers le Transformer
		transformer_out = self.transformer_encoder(src)


		transformer_out = transformer_out.permute(1, 0, 2)

		# Ce que l'on ferait si on voulait classifier la phrase entière
		# last_out = transformer_out[-1, :, :]  # (batch_size, hidden_dim)

		# Passer à travers la couche de sortie
		output = self.layers(transformer_out)  # (batch_size, output_dim)

		return output


class LSTM_Encoder(nn.Module):
	def __init__(self,
				 input_dim: int,
				 emb_dim: int,
				 bidirectional: bool,
				 lstm_dropout: float,
				 positional_embeddings: bool,
				 device: str,
				 lstm_hidden_size: int,
				 num_lstm_layers: int,
				 batch_size: int,
				 out_classes: int,
				 include_lang_metadata: bool,
				 num_langs: int,
				 attention: bool,
				 lang_emb_dim: int,
				 load_pretrained_embeddings:bool,
				 pretrained_weights:np.ndarray,
				 linear_layers:int,
				 linear_layers_hidden_size:int,
				 use_bert_tokenizer:bool,
				 keep_bert_dimensions:bool,
				 linear_dropout:float):
		super().__init__()

		# On peut utiliser des embeddings pré-entraînés pour vérifier si ça améliore les résultats
		if load_pretrained_embeddings or use_bert_tokenizer:
			# Hard-codé, il vaudrait mieux récupérer à partir des données des embeddings
			self.input_dim = 119547
			if keep_bert_dimensions:
				emb_dim = 768
			# Ici on vérifiera le paramètre _freeze
			self.embedding = torch.nn.Embedding(num_embeddings=self.input_dim, embedding_dim=emb_dim)
			# Censé initialiser les paramètres avec les poids pré-entraînés
			if load_pretrained_embeddings:
				self.embedding.weight.data = torch.tensor(pretrained_weights)
				print(f"Pretrained embeddings loaded dtype: {pretrained_weights.dtype}")
		else:
			# Sinon on utilise l'initialisation normale
			self.embedding = nn.Embedding(input_dim, emb_dim)
		self.include_lang_metadata = include_lang_metadata
		self.attention = attention
		self.num_langs = num_langs
		self.linear_layers = linear_layers

		# Possibilité de produire des embeddings de langue que l'on va concaténer aux plongements de mots
		if self.include_lang_metadata:
			self.lang_embedding = nn.Embedding(self.num_langs, lang_emb_dim)  # * self.scale
			# Si on concatène les embeddings, la dimension de sortie après concaténation est la somme de
			# la dimension des deux types de plongements
			lstm_input_size = emb_dim + lang_emb_dim
		else:
			lstm_input_size = emb_dim
		self.bidi = bidirectional
		# self.dropout = nn.Dropout(dropout)
		if not lstm_dropout:
			lstm_dropout = 0
		print(lstm_input_size)
		self.lstm = nn.LSTM(input_size=lstm_input_size,
							hidden_size=lstm_hidden_size,
							num_layers=num_lstm_layers,
							batch_first=True,
							dropout=lstm_dropout,
							bidirectional=bidirectional)
		self.positional_embeddings = positional_embeddings

		# On peut ajouter des plongements positionnels mais avec un lstm c'est probablement moins utile
		if positional_embeddings:
			self.pos1Dsum = Summer(PositionalEncoding1D(emb_dim))
		self.device = device
		self.input_dim = input_dim
		self.hidden_dim = lstm_hidden_size
		self.batch_size = batch_size
		self.num_lstm_layers = num_lstm_layers
		self.linear_dropout = linear_dropout


		# Une couche d'attention multitête
		if self.attention:
			if self.bidi:
				# La sortie du LSTM est doublée en taille si c'est bidirectionnel (lr et rl)
				self.multihead_attn = nn.MultiheadAttention(self.hidden_dim * 2, 8)
			else:
				self.multihead_attn = nn.MultiheadAttention(self.hidden_dim, 8)

		layers = []
		if self.linear_layers == 1:
			if self.bidi:
				layers.append(nn.Linear(lstm_hidden_size * 2, out_classes))
			else:
				layers.append(nn.Linear(lstm_hidden_size, out_classes))
		else:
			if self.bidi:
				layers.append(nn.Linear(lstm_hidden_size * 2, linear_layers_hidden_size))
			else:
				layers.append(nn.Linear(lstm_hidden_size, linear_layers_hidden_size))
			layers.append(nn.ReLU())
			for layer in range(self.linear_layers):
				if layer != self.linear_layers - 2:
					layers.append(nn.Linear(linear_layers_hidden_size, linear_layers_hidden_size))
					layers.append(nn.ReLU())
					layers.append(nn.Dropout(self.linear_dropout))
				else:
					layers.append(nn.Linear(linear_layers_hidden_size, out_classes))
					break

		self.layers = nn.Sequential(*layers)

	def forward(self, src, lang):
		batch_size, seq_length = src.size()
		# On plonge le texte [batch_size, max_length, embeddings_dim]
		embedded = self.embedding(src)
		if self.include_lang_metadata:

			# Shape: [batch_size, lang_metadata_dimensions]
			lang_embedding = self.lang_embedding(lang)
			# On augmente de dimension pour pouvoir concaténer chaque token et la langue:
			# [batch_size, max_length, lang_metadata_dimensions]
			projected_lang = lang_embedding.unsqueeze(1).expand(-1, seq_length,
																-1)  # (batch_size, seq_length, embedding_dim)
			# On concatène chaque token avec le vecteur de langue, c'est-à-dire qu'on augmente la
			# dimensionnalité de chaque vecteur de mot dont la dimension sera la somme des deux dimensions:
			# [batch_size, max_length, lang_metadata_dimensions + word_embedding_dimension]
			embedded = torch.cat((embedded, projected_lang), 2)
		else:
			embedded = self.embedding(src)
		if self.positional_embeddings:
			embedded = self.pos1Dsum(embedded)  #
		if self.bidi:
			(h, c) = (torch.zeros(2 * self.num_lstm_layers, self.batch_size, self.hidden_dim).to(self.device),
					  torch.zeros(2 * self.num_lstm_layers, self.batch_size, self.hidden_dim).to(self.device))
		else:
			(h, c) = (torch.zeros(1 * self.num_lstm_layers, self.batch_size, self.hidden_dim).to(self.device),
					  torch.zeros(1 * self.num_lstm_layers, self.batch_size, self.hidden_dim).to(self.device))
		lstm_out, (h, c) = self.lstm(embedded, (h, c))
		# La sortie est de forme [batch_size, max_length, hidden_size (*2 si c'est bidirectionnel)]:
		# le LSTM va retourner autant de tokens en sortie qu'en entrée

		# Attention et classification
		if self.attention:
			attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
			outs = self.layers(attn_output + lstm_out)
		else:
			outs = self.layers(lstm_out)
		# dimension: [batch_size, max_length, 3] pour [SC], [SB], [PAD]
		return outs


# embedded = self.dropout(tok_embedded + pos_embedded)

class CnnEncoder(nn.Module):
	def __init__(self,
				 input_dim,
				 emb_dim,
				 kernel_size,
				 dropout,
				 device,
				 linear_layers_hidden_size,
				 linear_layers,
				 out_classes,
				 positional_embeddings,
				 hidden_size,
				 num_langs,
				 num_conv_layers,
				 include_lang_metadata,
				 attention,
				 lang_emb_dim,
				 load_pretrained_embeddings,
				 use_bert_tokenizer,
				 pretrained_weights,
				 cnn_scale,
				 keep_bert_dimensions,
				 linear_dropout):
		super().__init__()

		assert kernel_size % 2 == 1, "Kernel size must be odd!"
		self.linear_layers = linear_layers
		self.scale = cnn_scale
		self.hidden_dim = hidden_size
		self.num_langs = num_langs
		self.attention = attention
		self.out_classes = out_classes
		self.include_lang_metadata = include_lang_metadata
		self.device = device
		self.positional_embeddings = positional_embeddings
		self.linear_dropout = linear_dropout

		# Possibilité de produire des embeddings de langue que l'on va concaténer aux plongements de mots

		if load_pretrained_embeddings or use_bert_tokenizer:
			# Hard-codé, il vaudrait mieux récupérer à partir des données des embeddings
			self.input_dim = 119547
			if keep_bert_dimensions:
				emb_dim = 768
			self.embedding = torch.nn.Embedding(num_embeddings=self.input_dim, embedding_dim=emb_dim)
			# Censé initialiser les paramètres avec les poids pré-entraînés
			if load_pretrained_embeddings:
				self.embedding.weight.data = torch.tensor(pretrained_weights)
				print(f"Pretrained embeddings loaded dtype: {pretrained_weights.dtype}")
		else:
			# Sinon on utilise l'initialisation normale
			self.embedding = nn.Embedding(input_dim, emb_dim)

		if self.include_lang_metadata:
			self.lang_embedding = nn.Embedding(self.num_langs, lang_emb_dim)  # * self.scale
			# Si on concatène les embeddings, la dimension de sortie après concaténation est la somme de
			# la dimension des deux types de plongements
			cnn_emb_dim = emb_dim + lang_emb_dim
		else:
			cnn_emb_dim = emb_dim

		if self.positional_embeddings:
			self.pos1Dsum = Summer(PositionalEncoding1D(cnn_emb_dim))

		if include_lang_metadata:
			self.emb2hid = nn.Linear(cnn_emb_dim, self.hidden_dim + lang_emb_dim)
			self.hid2emb = nn.Linear(self.hidden_dim + lang_emb_dim, cnn_emb_dim)
		else:
			self.emb2hid = nn.Linear(cnn_emb_dim, self.hidden_dim)
			self.hid2emb = nn.Linear(self.hidden_dim, cnn_emb_dim)

		if self.include_lang_metadata:
			in_channel = self.hidden_dim + lang_emb_dim
		else:
			in_channel = self.hidden_dim
		self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channel,
											  out_channels=2 * in_channel,
											  kernel_size=kernel_size,
											  padding=(kernel_size - 1) // 2)
									for _ in range(num_conv_layers)])

		self.dropout = nn.Dropout(dropout)


		# Une couche d'attention multitête
		if self.attention:
			self.multihead_attn = nn.MultiheadAttention(cnn_emb_dim, 8)


		layers = []

		if self.linear_layers == 1:
			layers.append(nn.Linear(cnn_emb_dim, self.out_classes))
		else:
			layers.append(nn.Linear(cnn_emb_dim, linear_layers_hidden_size))
			layers.append(nn.ReLU())
			for layer in range(self.linear_layers):
				if layer != self.linear_layers - 2:
					layers.append(nn.Linear(linear_layers_hidden_size, linear_layers_hidden_size))
					layers.append(nn.ReLU())
					layers.append(nn.Dropout(linear_dropout))
				else:
					layers.append(nn.Linear(linear_layers_hidden_size, self.out_classes))
					break

		self.decoder = nn.Sequential(*layers)

	def forward(self, src, lang):
		# src = [batch size, src len]
		batch_size, seq_length = src.size()

		# batch_size = src.shape[0]
		# src_len = src.shape[1]

		# create position tensor
		# pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

		# pos = [0, 1, 2, 3, ..., src len - 1]

		# pos = [batch size, src len]

		# embed tokens
		if self.include_lang_metadata:
			embedded = self.embedding(src)
			# Shape: [batch_size, lang_metadata_dimensions]
			lang_embedding = self.lang_embedding(lang)
			# On augmente de dimension pour pouvoir concaténer chaque token et la langue:
			# [batch_size, max_length, lang_metadata_dimensions]
			projected_lang = lang_embedding.unsqueeze(1).expand(-1, seq_length,
																-1)  # (batch_size, seq_length, embedding_dim)
			# On concatène chaque token avec le vecteur de langue, c'est-à-dire qu'on augmente la
			# dimensionnalité de chaque vecteur de mot dont la dimension sera la somme des deux dimensions:
			# [batch_size, max_length, lang_metadata_dimensions + word_embedding_dimension]
			embedded = torch.cat((embedded, projected_lang), 2)
		else:
			embedded = self.embedding(src)
		# pos_embedded = torch.zeros(tok_embedded.shape)

		if self.positional_embeddings:
			embedded = self.pos1Dsum(embedded)
		# tok_embedded = pos_embedded = [batch size, src len, emb dim]

		embedded = self.dropout(embedded)

		# embedded = [batch size, src len, emb dim]

		# pass embedded through linear layer to convert from emb dim to hid dim
		conv_input = self.emb2hid(embedded)

		# conv_input = [batch size, src len, hid dim]

		# permute for convolutional layer
		conv_input = conv_input.permute(0, 2, 1)

		# conv_input = [batch size, hid dim, src len]

		# begin convolutional blocks...

		for i, conv in enumerate(self.convs):
			# pass through convolutional layer
			conved = conv(self.dropout(conv_input))

			# conved = [batch size, 2 * hid dim, src len]

			# pass through GLU activation function
			conved = F.glu(conved, dim=1)

			# conved = [batch size, hid dim, src len]

			# apply residual connection
			conved = (conved + conv_input) * self.scale

			# conved = [batch size, hid dim, src len]

			# set conv_input to conved for next loop iteration
			conv_input = conved

		# ...end convolutional blocks

		# permute and convert back to emb dim
		conved = self.hid2emb(conved.permute(0, 2, 1))

		# conved = [batch size, src len, emb dim]

		# transformed = self.transformerEncoder(conved)
		# Attention et classification
		if self.attention:
			attn_output, _ = self.multihead_attn(conved, conved, conved)
			outs = self.decoder(attn_output + conved)
		else:
			outs = self.decoder(conved)


		return outs



if __name__ == '__main__':
    save_bert_embeddings()