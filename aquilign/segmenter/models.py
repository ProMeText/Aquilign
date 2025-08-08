import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import aquilign.segmenter.utils as utils
import re
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

class RNN_Encoder(nn.Module):
	pass

class LSTM_Encoder(nn.Module):
	def __init__(self,
				 input_dim:int,
				 emb_dim:int,
				 bidirectional_lstm:bool,
				 dropout:float,
				 positional_embeddings:bool,
				 device:str,
				 lstm_hidden_size:int,
				 num_lstm_layers:int=1,
				 batch_size:int=32,
				 tagset_size:int=3,
				 include_lang_metadata:bool=True):

		super().__init__()
		self.tok_embedding = nn.Embedding(input_dim, emb_dim)
		self.include_lang_metadata = include_lang_metadata
		if self.include_lang_metadata:
			# Voir si c'est la meilleure méthode. Les embeddings sont de la même dimension que ceux du texte, pas forcément ouf.
			self.scale = torch.sqrt(torch.FloatTensor([0.5]))
			self.lang_embedding = nn.Embedding(num_lang, emb_dim) * self.scale
		self.bidi = bidirectional_lstm
		print(self.tok_embedding)
		# self.dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(input_size=emb_dim,
							hidden_size=lstm_hidden_size,
							num_layers=num_lstm_layers,
							batch_first=True,
							dropout=dropout,
							bidirectional=bidirectional_lstm)
		self.positional_embeddings = positional_embeddings


		if positional_embeddings:
			self.pos1Dsum = Summer(PositionalEncoding1D(emb_dim))
		self.device = device
		self.input_dim = input_dim
		self.hidden_dim = lstm_hidden_size
		self.batch_size = batch_size
		if self.bidi:
			self.linear_layer = nn.Linear(lstm_hidden_size * 2, tagset_size)
		else:
			self.linear_layer = nn.Linear(lstm_hidden_size, tagset_size)

		# On normalise le long de la dimension 2 (sur chaque ligne)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, src):
		if self.include_lang_metadata:
			txt, langs = src
			lang_embedding = self.lang_embedding(langs)
			embedded = self.tok_embedding(src)
			embedded = torch.sum(embedded + lang_embedding)
		else:
			embedded = self.tok_embedding(src)

		if self.positional_embeddings:
			embedded = self.pos1Dsum(embedded)   #

		batch_size = self.batch_size
		if self.bidi:
			(h, c) = (torch.zeros(2, batch_size, self.hidden_dim), torch.zeros(2, batch_size, self.hidden_dim))
		else:
			(h, c) = (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))
		lstm_out, (h, c) = self.lstm(embedded, (h, c))

		outs = self.linear_layer(lstm_out)
		norms = self.softmax(outs)

		return norms
		# tok_embedded = pos_embedded = [batch size, src len, emb dim]

		# combine embeddings by elementwise summing
		# embedded = self.dropout(embedded)
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


