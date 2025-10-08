# Functions are imported/adapted from AllenAI's AllenNLP library:
# https://github.com/allenai/allennlp/

"""Defines the character embedding module (adapted from ELMo)"""

import json
from typing import Dict, Callable

import numpy
import torch


PADDING_VALUE = 0


def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int):

    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


def pad_sequence_to_length(
    sequence: list,
    desired_length: int,
    default_value: Callable[[], any] = lambda: 0,
    padding_on_right: bool = True,
) -> list:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.
    The original list is not modified.
    """

    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


class CharacterMapper:
    """
    Maps individual tokens to sequences of character ids.
    """

    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding>
    mask_character = 261  # <mask>

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    mask_characters = _make_bos_eos(
        mask_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    pad_characters = [PADDING_VALUE - 1] * max_word_length

    bos_token = "[CLS]"
    eos_token = "[SEP]"
    pad_token = "[PAD]"
    mask_token = "[MASK]"

    def __init__(self, tokens_to_add: Dict[str, int] = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) -> list[int]:
        if word in self.tokens_to_add:
            char_ids = [CharacterMapper.padding_character] * CharacterMapper.max_word_length
            char_ids[0] = CharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = CharacterMapper.end_of_word_character
        elif word == CharacterMapper.bos_token:
            char_ids = CharacterMapper.beginning_of_sentence_characters
        elif word == CharacterMapper.eos_token:
            char_ids = CharacterMapper.end_of_sentence_characters
        elif word == CharacterMapper.mask_token:
            char_ids = CharacterMapper.mask_characters
        elif word == CharacterMapper.pad_token:
            char_ids = CharacterMapper.pad_characters
        else:
            word_encoded = word.encode("utf-8", "ignore")[
                : (CharacterMapper.max_word_length - 2)
            ]
            char_ids = [CharacterMapper.padding_character] * CharacterMapper.max_word_length
            char_ids[0] = CharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = CharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class CharacterIndexer:
    def  __init__(self) -> None:
        self._mapper = CharacterMapper()

    def tokens_to_indices(self, tokens: list[str]) -> list[list[int]]:
        return [self._mapper.convert_word_to_char_ids(token) for token in tokens]

    def _default_value_for_padding(self):
        return [PADDING_VALUE] * CharacterMapper.max_word_length

    def as_padded_tensor(self, batch: list[list[str]], as_tensor=True, maxlen=None) -> torch.Tensor:
        if maxlen is None:
            maxlen = max(map(len, batch))
        batch_indices = [self.tokens_to_indices(tokens) for tokens in batch]
        padded_batch = [
            pad_sequence_to_length(
                indices, maxlen,
                default_value=self._default_value_for_padding)
            for indices in batch_indices
        ]
        if as_tensor:
            return torch.LongTensor(padded_batch)
        else:
            return padded_batch


class Highway(torch.nn.Module):
    """
    A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    # Parameters

    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class CharacterCNN(torch.nn.Module):
    """
    Computes context insensitive token representations from each token's characters.
    """

    def __init__(self,
                 char_embedding_dim,
                 lang_emb_dim=None,
            output_dim: int = 768,
            requires_grad: bool = True,
            vocab_length=179,
            max_tokens=29) -> None:
        super().__init__()

        self._options = {
            'char_cnn': {
                'activation': 'relu',
                'filters': [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]
                    ],
                'n_highway': 2,
                'embedding': {'dim': char_embedding_dim},
                'total': {'dim': char_embedding_dim + lang_emb_dim
                if lang_emb_dim else char_embedding_dim},
                'n_characters': vocab_length,
                'max_characters_per_token': max_tokens
            }
        }
        self.output_dim = output_dim
        self.requires_grad = requires_grad
        self.lang_embedding_dim = lang_emb_dim
        self._init_weights()
        self.char_embedding_dim = char_embedding_dim
        if lang_emb_dim:
            self.lang_embeddings = torch.nn.Embedding(8, lang_emb_dim)


    def _init_weights(self):
        self._init_char_embedding()
        self._init_cnn_weights()
        self._init_highway()
        self._init_projection()

    def _init_char_embedding(self):
        weights = numpy.zeros(
            (
                self._options["char_cnn"]["n_characters"] + 1,
                self._options["char_cnn"]["embedding"]["dim"]
            ),
            dtype="float32")
        weights[-1, :] *= 0.  # padding
        self._char_embedding_weights = torch.nn.Parameter(
            torch.FloatTensor(weights), requires_grad=self.requires_grad
        )

    def _init_cnn_weights(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        char_embed_dim = cnn_options["total"]["dim"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim, out_channels=num,
                kernel_size=width, bias=True)
            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad
            convolutions.append(conv)
            self.add_module("char_conv_{}".format(i), conv)
        self._convolutions = convolutions

    def _init_highway(self):
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options["n_highway"]

        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            self._highways._layers[k].weight.requires_grad = self.requires_grad
            self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _init_projection(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        self._projection.weight.requires_grad = self.requires_grad
        self._projection.bias.requires_grad = self.requires_grad

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs: torch.Tensor, langs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.
        Returns
        -------
        embeddings: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, embedding_dim)`` tensor with context
            insensitive token representations.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        #character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
        #    inputs, mask, self._beginning_of_sentence_characters, self._end_of_sentence_characters
        #)
        character_ids_with_bos_eos, mask_with_bos_eos = inputs, mask

        # the character id embedding
        max_chars_per_token = self._options["char_cnn"]["max_characters_per_token"]
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            inputs.view(-1, max_chars_per_token + 2), self._char_embedding_weights
        )
        if self.lang_embedding_dim:
            batch_size, sequence_length, max_chars = inputs.size()
            language_vectors = self.lang_embeddings(langs)
            language_vectors = language_vectors.unsqueeze(1)  # [48, 1, 32]
            language_vectors = language_vectors.expand(-1, sequence_length, -1)  # [48, 163, 32]
            language_vectors_flat = language_vectors.reshape(-1, self.char_embedding_dim)  # [7824, 32]
            language_vectors_expanded = language_vectors_flat.unsqueeze(1).expand(-1, max_chars, -1)
            character_embedding = torch.cat([character_embedding, language_vectors_expanded], dim=-1)
        # run convolutions
        cnn_options = self._options["char_cnn"]
        if cnn_options["activation"] == "tanh":
            activation = torch.tanh
        elif cnn_options["activation"] == "relu":
            activation = torch.nn.functional.relu
        else:
            raise Exception("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_{}".format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return token_embedding.view(batch_size, sequence_length, -1)
