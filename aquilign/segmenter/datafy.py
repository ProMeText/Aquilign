import random

import tokenizers
from torch.utils.data import Dataset
import torch
import aquilign.segmenter.utils as utils
from transformers import AutoTokenizer
import json
import re
import numpy as np



class SentenceBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, texts_and_labels, tokenizer):
        self.texts_and_labels = texts_and_labels

    def __len__(self):
        return len(self.texts_and_labels)

    def __getitem__(self, idx):
        return self.texts_and_labels[idx]

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomTextDataset(Dataset):
    def __init__(self,
                 set_type,
                 mode,
                 train_path,
                 test_path,
                 dev_path,
                 delimiter,
                 output_dir,
                 create_vocab,
                 data_augmentation,
                 tokenizer_name,
                 input_vocab=None,
                 lang_vocab=None,
                 use_pretrained_embeddings=False,
                 debug=False,
                 filter_by_lang=None,
                 use_bert_tokenizer=False,
                 use_char_embeddings=False,
                 architecture="lstm",
                 tuning_mode=False,
                 weight_factor=2):
        print(f"Loading new dataset: {set_type}")
        self.datafy = Datafier(
                                set_type,
                                train_path,
                               test_path,
                               dev_path,
                               delimiter,
                               output_dir,
                               create_vocab,
                               input_vocab,
                               lang_vocab,
                               use_pretrained_embeddings,
                               debug,
                               data_augmentation,
                               tokenizer_name=tokenizer_name,
                               filter_by_lang=filter_by_lang,
                               use_bert_tokenizer=use_bert_tokenizer,
                               use_char_embeddings=use_char_embeddings,
                               architecture=architecture,
                               tuning_mode=tuning_mode,
                               weight_factor=weight_factor,
                               mode=mode)
        self.use_char_embeddings = use_char_embeddings
        self.architecture = architecture
        self.mode = mode
        self.set_type = set_type
        if self.set_type == "train":
            print("Creating train corpus")
            self.datafy.create_train_corpus()
            print("Train corpus created.")
            assert self.datafy.train_padded_examples.shape[1] == self.datafy.train_padded_targets.shape[1] , (f"Something went wrong with corpus creation.\n"
             f"Padded examples shape: {self.datafy.train_padded_examples.shape}\n"
             f"Padded targets shape: {self.datafy.train_padded_targets.shape}.")
        elif self.set_type == "test":
            print("Creating test corpus")
            self.datafy.create_test_corpus()
        else:
            print("Creating dev corpus")
            self.datafy.create_dev_corpus()

    def __len__(self):
        if self.set_type == "train":
            return len(self.datafy.train_padded_examples)
        elif self.set_type == "test":
            return len(self.datafy.test_padded_examples)
        else:
            return len(self.datafy.dev_padded_examples)

    def __getitem__(self, idx):
        if "BERT" in self.architecture:
            if self.set_type == "train":
                examples = self.datafy.train_padded_examples[idx]
                masks = self.datafy.train_attention_masks[idx]
                labels = self.datafy.train_padded_targets[idx]
            elif self.set_type == "test":
                examples = self.datafy.test_padded_examples[idx]
                masks = self.datafy.test_attention_masks[idx]
                labels = self.datafy.test_padded_targets[idx]
            else:
                examples = self.datafy.dev_padded_examples[idx]
                masks = self.datafy.dev_attention_masks[idx]
                labels = self.datafy.dev_padded_targets[idx]
            return examples, masks, labels
        else:
            if self.set_type == "train":
                examples = self.datafy.train_padded_examples[idx]
                labels = self.datafy.train_padded_targets[idx]
                langs = self.datafy.train_langs[idx]
            elif self.set_type == "test":
                examples = self.datafy.test_padded_examples[idx]
                labels = self.datafy.test_padded_targets[idx]
                langs = self.datafy.test_langs[idx]
            else:
                examples = self.datafy.dev_padded_examples[idx]
                labels = self.datafy.dev_padded_targets[idx]
                langs = self.datafy.dev_langs[idx]
            return examples, langs, labels


class Datafier:
    def __init__(self,
                 set_type,
                 train_path,
                 test_path,
                 dev_path,
                 delimiter,
                 output_dir,
                 create_vocab,
                 input_vocab,
                 lang_vocab=None,
                 use_pretrained_embeddings=False,
                 debug=False,
                 data_augmentation=False,
                 tokenizer_name="",
                 filter_by_lang=None,
                 use_bert_tokenizer=False,
                 use_char_embeddings=False,
                 architecture="lstm",
                 tuning_mode=False,
                 weight_factor=2,
                 mode="train"
                 ):
        self.max_length_examples = 0
        self.frequency_dict = {}
        self.output_dir = output_dir
        self.vocab_dir = f"{self.output_dir}/vocab"
        self.unknown_threshold = 14  # Under this frequency the tokens will be tagged as [UNK]
        self.input_vocabulary = {}
        self.target_weights = None
        self.reverse_target_classes = {}
        self.train_padded_examples = []
        self.train_padded_targets = []
        self.test_padded_examples = []
        self.test_padded_targets = []
        self.dev_padded_examples = []
        self.dev_padded_targets = []
        self.train_path = train_path
        self.test_path = test_path
        self.dev_path = dev_path
        self.delimiter = delimiter
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.set_type = set_type
        if self.mode == "train":
            self.train_data = self.import_json_corpus(train_path)
            self.dev_data = self.import_json_corpus(dev_path)
        self.test_data = self.import_json_corpus(test_path)
        self.previous_model_vocab = input_vocab
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.use_bert_tokenizer = use_bert_tokenizer
        self.debug = debug
        self.target_classes = {"[SC]": 0,  # Segment content > no split
                               "[SB]": 1,  # Segment boundary > split before
                               "[PAD]": 2
                               }
        self.filter_by_lang = filter_by_lang
        self.reverse_target_classes = {idx:token for token, idx in self.target_classes.items()}
        self.tuning_mode = tuning_mode
        # if self.tuning_mode is False:
            # utils.serialize_dict(self.target_classes, f"{self.vocab_dir}/target_classes.json")
        self.delimiters_regex = re.compile(r"\s+|([\.“\?\'!—\"/:;,\-¿«\[\]»])")
        if self.data_augmentation and self.mode == "train":
            # full_corpus = self.train_data + self.remove_punctuation(self.train_data) + utils.apply_noise()
            self.train_data = utils.augment_data([self.train_data])[0]
        if mode == "train":
            full_corpus = self.train_data + self.test_data + self.dev_data
            assert len(self.train_data) != len(self.test_data) != 0, "Some error here."
        self.architecture = architecture
        self.use_char_embeddings = use_char_embeddings
        self.weight_factor = weight_factor
        if self.architecture in ["BERT", "DISTILBERT"]:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if mode == "train":
                self.create_lang_vocab(full_corpus)
            else:
                self.lang_vocabulary = lang_vocab
            self.input_vocabulary = self.tokenizer.get_vocab()
        else:
            if set_type == "train":
                if self.use_char_embeddings:
                    self.get_max_length(full_corpus)
                if create_vocab:
                    print("Creating vocabulary.")
                    self.create_vocab(self.remove_punctuation(full_corpus) + full_corpus, use_char_embeddings)
                    self.create_lang_vocab(full_corpus)
                else:
                    self.lang_vocabulary = None

            # Si on utilise un tokéniseur BERT
            elif self.use_pretrained_embeddings or self.use_bert_tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.create_lang_vocab(full_corpus)
                self.input_vocabulary = self.tokenizer.get_vocab()

            # Dans tous les autres cas
            else:
                if self.use_char_embeddings:
                    self.get_max_length(self.test_data)
                self.input_vocabulary = input_vocab
                self.lang_vocabulary = lang_vocab
        assert self.input_vocabulary != {}, "Error with input vocab"


    def create_lang_vocab(self, data):
        langs = {item["lang"] for item in data}
        lang_vocab = {"[UNK]": 0}
        self.lang_vocabulary = {
            **lang_vocab, **{lang:idx + 1 for idx, lang in enumerate(langs)}
        }
        if self.tuning_mode is False:
            utils.serialize_dict(self.lang_vocabulary, f"{self.vocab_dir}/lang_vocab.json")

    def get_max_length(self, corpus):
        corpus_as_string = " ".join([example['example'] for example in corpus])
        splitted_text = re.split(self.delimiters_regex, corpus_as_string)
        self.max_token_length = max([len(item) for item in splitted_text if item])

    def create_vocab(self, data:list[dict], use_char_embeddings:bool=False):

        examples = [item["example"] for item in data]
        # On fusionne l'ensemble du corpus
        data_string = " ".join(examples).replace(self.delimiter, " ")
        # data_string = data_string[:100]
        if use_char_embeddings is False:
            input_vocabulary = {"[PAD]": 0,
                                "[UNK]": 1}
            n = 2
            splitted_text = re.split(self.delimiters_regex, data_string)
            for item in splitted_text:
                if item not in ["", None] and item.lower() not in input_vocabulary:
                    input_vocabulary[item.lower()] = n
                    n += 1
        else:
            input_vocabulary = {"[PAD]": 0,
                                "[UNK]": 1,
                                "[EOT]": 2,
                                "[SOT]": 3}
            n = 4
            input_vocabulary = {**input_vocabulary,
                                **{
                char: idx + n for idx, char
                in enumerate(
                    list(
                        set(data_string)
                    )
                )
            }
                                }
        self.input_vocabulary = input_vocabulary
        self.reverse_input_vocabulary = {idx:char for char, idx in self.input_vocabulary.items()}
        if self.tuning_mode is False:
            utils.serialize_dict(self.input_vocabulary, f"{self.vocab_dir}/input_vocab.json")

    def remove_punctuation(self, data) -> list[dict]:
        data_no_punct = []
        punctuation_regexp = re.compile(r"[\.,!?:;«”\"»\-\(\)\[\]]")
        for example in data:
            if not re.search(punctuation_regexp, example["example"]):
                continue
            text = example['example']
            lang = example['lang']
            text = re.sub(punctuation_regexp, '', text)
            data_no_punct.append({"example": text, "lang": lang})
        return data_no_punct

    def deduce_weights(self, weight_factor):
        """
        Fonction simple de pondération des classes:
        weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
        https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
        On ignore le padding.
        """
        # On supprime le batch
        as_single_vector = torch.flatten(self.train_padded_targets)

        # On supprime le padding
        examples_as_list = [item for item in as_single_vector.tolist() if item != 2]
        segment_content = examples_as_list.count(self.target_classes["[SC]"])
        segment_boundary = examples_as_list.count(self.target_classes["[SB]"])
        total_samples = len(examples_as_list)

        # Petite vérification
        assert segment_boundary + segment_content == total_samples, "Les calculs ne sont pas bons, Kévin"
        segment_content_weight = total_samples / (segment_content * 2)
        segment_boundary_weight = total_samples / (segment_boundary * 2)
        print(f"Balanced weights: {[segment_content_weight, segment_boundary_weight, 0]}")
        self.target_weights = torch.tensor([segment_content_weight, segment_boundary_weight, 0])

    def create_train_corpus(self):
        if self.architecture in ["BERT", "DISTILBERT"]:
            train_padded_examples, train_attention_masks, train_langs, train_padded_targets = self.produce_corpus(self.train_data, debug=self.debug)
            self.train_attention_masks = utils.tensorize(train_attention_masks)
        else:
            train_padded_examples, train_langs, train_padded_targets = self.produce_corpus(self.train_data, debug=self.debug)
        self.train_padded_examples = utils.tensorize(train_padded_examples)
        self.train_langs = utils.tensorize(train_langs)
        self.train_padded_targets = utils.tensorize(train_padded_targets)

    def create_test_corpus(self):
        """
        This function creates the test corpus, and uses the vocabulary of the train set to do so.
        Outputs: tensorized input, tensorized target, formatted input to ease accuracy computation.
        """
        full_corpus = self.test_data
        if self.architecture in ["BERT", "DISTILBERT"]:
            test_padded_examples, test_attention_masks, test_langs, test_padded_targets = self.produce_corpus(full_corpus, debug=self.debug)
            self.test_attention_masks = utils.tensorize(test_attention_masks)
        else:
            test_padded_examples, test_langs, test_padded_targets = self.produce_corpus(full_corpus, debug=self.debug)
        self.test_padded_examples = utils.tensorize(test_padded_examples)
        self.test_langs = utils.tensorize(test_langs)
        self.test_padded_targets = utils.tensorize(test_padded_targets)

    def create_dev_corpus(self):
        """
        This function creates the dev corpus, and uses the vocabulary of the train set to do so.
        Outputs: tensorized input, tensorized target, formatted input to ease accuracy computation.
        """
        full_corpus = self.dev_data
        if self.architecture in ["BERT", "DISTILBERT"]:
            dev_padded_examples, dev_attention_masks, dev_langs, dev_padded_targets = self.produce_corpus(full_corpus, debug=self.debug)
            self.dev_attention_masks = utils.tensorize(dev_attention_masks)
        else:
            dev_padded_examples, dev_langs, dev_padded_targets = self.produce_corpus(full_corpus, debug=self.debug)
        self.dev_padded_examples = utils.tensorize(dev_padded_examples)
        self.dev_langs = utils.tensorize(dev_langs)
        self.dev_padded_targets = utils.tensorize(dev_padded_targets)

    def get_frequency(self, data_as_string):
        for char in data_as_string:
            try:
                self.frequency_dict[char] += 1
            except:
                self.frequency_dict[char] = 1


    def import_json_corpus(self, path: str) -> list:
        '''
        Import data and normalize
        '''
        with open(path, "r") as file:
            corpus_as_dict = json.load(file)
        # self.get_frequency("".join(normalized))
        examples = [example for example in corpus_as_dict['examples']]
        return examples

    def get_txt_as_str(self, path) -> str:
        with open(path, "r") as training_file:
            imported_data = training_file.read()
            cleaned_text = [utils.remove_multiple_spaces(line) for line in imported_data.split("\n")]
            normalized = [utils.normalize(line) for line in cleaned_text]

        return "".join(normalized)


    def text_to_chars_ids(self, corpus, max_tokens, max_sentence_length):
        sentence_as_ids = []
        for token in corpus:
            try:
                token_as_ids = [self.input_vocabulary[char] for char in token]
            except KeyError:
                print("Error with chars: ", [char for char in token if char not in self.input_vocabulary])
                print("Tagging token as UNK")
                token_as_ids = [self.input_vocabulary["[UNK]"] if char not in self.input_vocabulary else self.input_vocabulary[char] for char in token]
            token_as_ids = token_as_ids + [self.input_vocabulary["[EOT]"]]
            token_as_ids = [self.input_vocabulary["[SOT]"]] + token_as_ids
            padding_number = max_tokens - len(token_as_ids)
            token_as_ids = token_as_ids + [self.input_vocabulary["[PAD]"] for _ in range(padding_number)]
            sentence_as_ids.append(token_as_ids)
        sentence_difference = max_sentence_length - len(sentence_as_ids)
        sentence_as_ids = sentence_as_ids + [[self.input_vocabulary["[PAD]"] for _ in range(max_tokens)] for sentence in
                                             range(sentence_difference)]
        return sentence_as_ids

    def produce_corpus(self, data:list, debug=True) -> tuple:
        """
        This function takes the targets and creates the examples.
        """
        assert data != [], "Error with the data when producing the corpus"
        examples = []
        attention_masks = []
        targets = []
        langs = []
        ids = []
        if debug:
            data = data[:100]
        for example in data:
            try:
                text = example['example']
            except TypeError as e:
                print("Error with example:", example)
                print(e)
                exit(0)
            lang = example['lang']
            if self.filter_by_lang and lang != self.filter_by_lang:
                continue
            # Si on veut utiliser des embeddings pré-entraînés, il faut tokéniser avec le tokéniseur maison
            if self.use_pretrained_embeddings or self.use_bert_tokenizer or self.architecture in ["BERT", "DISTILBERT"]:
                try:
                    if "BERT" in self.architecture:
                        example, masks, idents, target = utils.convertSentenceToSubWordsAndLabels(text, self.tokenizer, self.delimiter, max_length=400, output_masks=True)
                        attention_masks.append(masks.tolist())
                    else:
                        example, idents, target = utils.convertSentenceToSubWordsAndLabels(text, self.tokenizer, self.delimiter, max_length=400)
                    ids.append(idents)
                except TypeError as e:
                    print("Passing.")
                    continue
            else:
                target = []
                example = []
                text = text.replace(self.delimiter, " " + self.delimiter)
                as_tokens = [item for item in re.split(self.delimiters_regex, text) if item not in [None, ""]]
                for idx, token in enumerate(as_tokens):
                    if not token:
                        continue
                    if token == self.delimiter:
                        print("Oups.")
                        print(as_tokens)
                        continue
                    if self.delimiter in token:
                        target.append("[SB]")
                        if token == self.delimiter:
                            print("Problemo")
                            print(text)
                        example.append(token.replace(self.delimiter, "").lower())
                    else:
                        target.append("[SC]")
                        example.append(token.lower())
                assert len(example) == len(target), "Length inconsistency"
            examples.append(example)
            targets.append(target)
            if not "BERT" in self.architecture and self.lang_vocabulary is not None:
                langs.append(self.lang_vocabulary[lang])

        self.max_length_examples = max([len(example) for example in examples])
        max_length_targets = max([len(target) for target in targets])
        if max_length_targets > 500:
            print("There is a problem with some line way too long. Please check the datasets.")
            print(np.mean([len(target) for target in targets]))
            print(max_length_targets)
            exit(0)
        if "BERT" not in self.architecture:
            if self.use_char_embeddings is True:
                pad_value = "[PAD]"
                padded_examples = []
                padded_targets = []
                assert self.input_vocabulary != {}, "Error with input vocabulary"
                for example in examples:
                    ids = self.text_to_chars_ids(example, max_tokens=self.max_token_length + 2, max_sentence_length=self.max_length_examples)
                    padded_examples.append(ids)

                for target in targets:
                    target_length = len(target)
                    target = target + [pad_value for _ in range(max_length_targets - target_length)]
                    target = [self.target_classes[token] for token in target]
                    padded_targets.append(target)
                return padded_examples, langs, padded_targets
            elif self.use_pretrained_embeddings is False and self.use_bert_tokenizer is False:
                pad_value = "[PAD]"
                padded_examples = []
                padded_targets = []
                assert self.input_vocabulary != {}, "Error with input vocabulary"
                for example in examples:
                    example_length = len(example)
                    example = example + [pad_value for _ in range(self.max_length_examples - example_length)]
                    if "" in example:
                        exit(0)
                    example = ["[PAD]"] + example
                    try:
                        example = [self.input_vocabulary[token] if token in self.input_vocabulary else self.input_vocabulary["[UNK]"] for token in example]
                    except KeyError:
                        print(example)
                        print(len(example))
                        problem = next(item for item in example if item not in self.input_vocabulary)
                        print(f"|{problem}| Is not in vocabulary")
                        exit(0)
                    padded_examples.append(example)


                for target in targets:
                    target_length = len(target)
                    target = target + [pad_value for _ in range(max_length_targets - target_length)]
                    target = ["[PAD]"] + target
                    target = [self.target_classes[token] for token in target]
                    padded_targets.append(target)
                return padded_examples, langs, padded_targets

        # On doit convertir la liste d'arrays vers un arrays, on concatène sur la dimension 0 (lignes)
        ids = np.concatenate(ids, axis=0)
        # targets = np.concatenate(targets, axis=0)
        targets = torch.stack(targets, dim=0)
        if self.architecture in ["BERT", "DISTILBERT"]:
            return ids, attention_masks, langs, targets
        else:
            return ids, langs, targets

