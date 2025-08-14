import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import aquilign.segmenter.utils as utils
import re
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import transformers


def save_bert_embeddings():
	myBertModel = transformers.BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
	print(sys.getsizeof(myBertModel))
	BertEmbeddings = myBertModel.embeddings.word_embeddings.weight.detach().numpy().ravel()
	print(type(BertEmbeddings))
	np.save("aquilign/segmenter/embeddings.pckl", BertEmbeddings)

class RNN_Encoder(nn.Module):
	pass


import torch
import torch.nn as nn


class TransformerModel(nn.Module):
	def __init__(self,
				 input_dim,
				 hidden_dim,
				 num_heads,
				 num_layers,
				 output_dim,
				 num_langs,
				 lang_emb_dim,
				 include_lang_metadata):
		super(TransformerModel, self).__init__()

		self.num_langs = num_langs
		self.lang_emb_dim = lang_emb_dim
		self.include_lang_metadata = include_lang_metadata
		# Couche d'embedding pour transformer les entrées dans l'espace de dimension hidden_dim
		self.embedding = nn.Embedding(input_dim, hidden_dim)

		if self.include_lang_metadata:
			self.lang_embedding = nn.Embedding(self.num_langs, lang_emb_dim)  # * self.scale
			hidden_dim = hidden_dim + lang_emb_dim


		# Encoder du Transformer
		encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# Couche de sortie
		self.fc_out = nn.Linear(hidden_dim, output_dim)

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
		output = self.fc_out(transformer_out)  # (batch_size, output_dim)

		return output


class LSTM_Encoder(nn.Module):
	def __init__(self,
				 input_dim: int,
				 emb_dim: int,
				 bidirectional_lstm: bool,
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
				 load_pretrained_embeddings:bool=True):
		super().__init__()
		if load_pretrained_embeddings:
			self.input_dim = 119547
			emb_dim = 768
			self.tok_embedding = torch.nn.Embedding(num_embeddings=self.input_dim, embedding_dim=emb_dim)
			weights = np.load("aquilign/segmenter/embeddings.pckl")
			self.tok_embedding.weights = weights["embeddings/data.pkl"]
		else:
			self.tok_embedding = nn.Embedding(input_dim, emb_dim)
		self.include_lang_metadata = include_lang_metadata
		self.attention = attention
		self.num_langs = num_langs
		if self.include_lang_metadata:
			# Voir si c'est la meilleure méthode. Les embeddings sont de la même dimension que ceux du texte, pas forcément ouf.
			# Autre possibilité, one-hot encoding et couche linéaire
			# self.scale = torch.sqrt(torch.FloatTensor([0.5]))
			self.lang_embedding = nn.Embedding(self.num_langs, lang_emb_dim)  # * self.scale
			lstm_input_size = emb_dim + lang_emb_dim
		else:
			lstm_input_size = emb_dim

		self.bidi = bidirectional_lstm
		# self.dropout = nn.Dropout(dropout)
		if not lstm_dropout:
			lstm_dropout = 0
		self.lstm = nn.LSTM(input_size=lstm_input_size,
							hidden_size=lstm_hidden_size,
							num_layers=num_lstm_layers,
							batch_first=True,
							dropout=lstm_dropout,
							bidirectional=bidirectional_lstm)
		self.positional_embeddings = positional_embeddings
		if positional_embeddings:
			self.pos1Dsum = Summer(PositionalEncoding1D(emb_dim))
		self.device = device
		self.input_dim = input_dim
		self.hidden_dim = lstm_hidden_size
		self.batch_size = batch_size




		if self.attention:
			if self.bidi:
				self.multihead_attn = nn.MultiheadAttention(self.hidden_dim * 2, 8)
			else:
				self.multihead_attn = nn.MultiheadAttention(self.hidden_dim, 8)

		# On peut aussi ajouter une couche d'attention.
		if self.bidi:
			self.linear_layer = nn.Linear(lstm_hidden_size * 2, out_classes)
		else:
			self.linear_layer = nn.Linear(lstm_hidden_size, out_classes)


	def forward(self, src, lang):
		batch_size, seq_length = src.size()
		# On plonge le texte
		embedded = self.tok_embedding(src)
		if self.include_lang_metadata:
			lang_embedding = self.lang_embedding(lang)
			# On augmente de dimension pour pouvoir concaténer chaque token et la langue
			projected_lang = lang_embedding.unsqueeze(1).expand(-1, seq_length,
																-1)  # (batch_size, seq_length, embedding_dim)
			# On concatène chaque token avec le vecteur de langue.
			embedded = torch.cat((embedded, projected_lang), 2)
		else:
			embedded = self.tok_embedding(src)

		if self.positional_embeddings:
			embedded = self.pos1Dsum(embedded)  #
		batch_size = self.batch_size
		if self.bidi:
			(h, c) = (torch.zeros(2, batch_size, self.hidden_dim).to(self.device),
					  torch.zeros(2, batch_size, self.hidden_dim).to(self.device))
		else:
			(h, c) = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
					  torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
		lstm_out, (h, c) = self.lstm(embedded, (h, c))

		if self.attention:
			attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
			outs = self.linear_layer(attn_output + lstm_out)
		else:
			outs = self.linear_layer(lstm_out)
		return outs


# embedded = self.dropout(tok_embedded + pos_embedded)

class CnnEncoder(nn.Module):
	def __init__(self,
				 input_dim,
				 emb_dim,
				 hid_dim,
				 n_layers,
				 kernel_size,
				 dropout,
				 device):
		super().__init__()

		assert kernel_size % 2 == 1, "Kernel size must be odd!"

		self.device = device

		self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

		self.tok_embedding = nn.Embedding(input_dim, emb_dim)

		self.emb2hid = nn.Linear(emb_dim, hid_dim)
		self.hid2emb = nn.Linear(hid_dim, emb_dim)

		self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
											  out_channels=2 * hid_dim,
											  kernel_size=kernel_size,
											  padding=(kernel_size - 1) // 2)
									for _ in range(n_layers)])

		self.dropout = nn.Dropout(dropout)

	def forward(self, src):
		# src = [batch size, src len]

		# batch_size = src.shape[0]
		# src_len = src.shape[1]

		# create position tensor
		# pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

		# pos = [0, 1, 2, 3, ..., src len - 1]

		# pos = [batch size, src len]

		# embed tokens
		tok_embedded = self.tok_embedding(src)
		# pos_embedded = torch.zeros(tok_embedded.shape)

		# tok_embedded = pos_embedded = [batch size, src len, emb dim]

		# combine embeddings by elementwise summing
		embedded = self.dropout(tok_embedded)
		# embedded = self.dropout(tok_embedded + pos_embedded)

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

		# elementwise sum output (conved) and input (embedded) to be used for attention
		combined = (conved + embedded) * self.scale

		# transformed = self.transformerEncoder(conved)

		# combined = [batch size, src len, emb dim]

		return combined, combined


class LinearDecoder(nn.Module):
	"""
    Simple Linear Decoder that outputs a probability distribution
    over the vocabulary
    Parameters
    ===========
    label_encoder : LabelEncoder
    in_features : int, input dimension
    """

	def __init__(self, enc_dim, out_dim):
		super().__init__()
		self.decoder = nn.Linear(enc_dim, out_dim)

	def forward(self, enc_outs):
		return self.decoder(enc_outs)


if __name__ == '__main__':
    save_bert_embeddings()