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
        # get the max length of the training set in order to have the good feature to put in tokenizer
        # current text (one line, ie 12 tokens [before automatic BERT tokenization])
        return self.texts_and_labels[idx]

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomTextDataset(Dataset):
    def __init__(self, mode, train_path, test_path, dev_path, device, all_dataset_on_device, delimiter, output_dir, create_vocab, data_augmentation, tokenizer_name, input_vocab=None, lang_vocab=None, use_pretrained_embeddings=False, debug=False, filter_by_lang=None):
        self.datafy = Datafier(train_path,
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
                               tokenizer_name,
                               filter_by_lang=filter_by_lang)
        self.mode = mode
        if mode == "train":
            self.datafy.create_train_corpus()
        elif mode == "test":
            self.datafy.create_test_corpus()
        else:
            self.datafy.create_dev_corpus()
        if all_dataset_on_device:
            self.datafy.train_padded_examples = torch.LongTensor(self.datafy.train_padded_examples).to(device)
            self.datafy.test_padded_examples = torch.LongTensor(self.datafy.test_padded_examples).to(device)
            self.datafy.train_padded_targets = torch.LongTensor(self.datafy.train_padded_targets).to(device)
            self.datafy.test_padded_targets = torch.LongTensor(self.datafy.test_padded_targets).to(device)
            self.datafy.train_langs = torch.LongTensor(self.datafy.train_langs).to(device)
            self.datafy.test_langs = torch.LongTensor(self.datafy.test_langs).to(device)

    def __len__(self):
        if self.mode == "train":
            return len(self.datafy.train_padded_examples)
        elif self.mode == "test":
            return len(self.datafy.test_padded_examples)
        else:
            return len(self.datafy.dev_padded_examples)

    def __getitem__(self, idx):
        if self.mode == "train":
            examples = self.datafy.train_padded_examples[idx]
            labels = self.datafy.train_padded_targets[idx]
            langs = self.datafy.train_langs[idx]
        elif self.mode == "test":
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
                 tokenizer_name,
                 filter_by_lang=None,
                 ):
        self.max_length_examples = 0
        self.frequency_dict = {}
        self.output_dir = output_dir
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
        self.train_data = self.import_json_corpus(train_path)
        self.test_data = self.import_json_corpus(test_path)
        self.dev_data = self.import_json_corpus(dev_path)
        self.previous_model_vocab = input_vocab
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.debug = debug
        self.target_classes = {"[SC]": 0,  # Segment content > no split
                               "[SB]": 1,  # Segment boundary > split before
                               "[PAD]": 2
                               }
        self.filter_by_lang = filter_by_lang
        self.reverse_target_classes = {idx:token for token, idx in self.target_classes.items()}
        utils.serialize_dict(self.target_classes, f"{self.output_dir}/target_classes.json")
        utils.serialize_dict(self.reverse_target_classes, f"{self.output_dir}/reverse_target_classes.json")
        self.delimiters_regex = re.compile(r"\s+|([\.“\?\!—\"/:;,\-¿«\[\]»])")
        full_corpus = self.train_data + self.test_data + self.dev_data
        assert len(self.train_data) != len(self.test_data) != 0, "Some error here."
        if create_vocab:
            self.create_vocab(self.remove_punctuation(full_corpus) + full_corpus)
            self.create_lang_vocab(full_corpus)
        elif self.use_pretrained_embeddings:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.create_lang_vocab(full_corpus)
        else:
            self.input_vocabulary = input_vocab
            self.lang_vocabulary = lang_vocab


    def update_vocab(self, input_vocab):
        """
        This function updates the existing vocab with the new examples.
        """
        length_previous_vocab: int = len(input_vocab)
        orig_list_of_characters: set = {char for char, _ in input_vocab.items()}
        train_data_as_string: str = self.get_txt_as_str(self.train_path)
        new_set_of_characters: set = utils.get_vocab(train_data_as_string)

        # We want the new characters, no matter if some chars are not present in the new vocab.
        unseen_chars = new_set_of_characters - orig_list_of_characters

        # We compare the original vocab with the new one to create a merged vocab
        # but we need to keep the order, i.e. to append new chars at the end of the dict
        if len(unseen_chars) != 0:
            for index, new_char in enumerate(list(unseen_chars)):
                self.input_vocabulary[new_char] = (length_previous_vocab + index)

    def create_lang_vocab(self, data):
        langs = {item["lang"] for item in data}
        self.lang_vocabulary = {lang:idx for idx, lang in enumerate(langs)}
        utils.serialize_dict(self.lang_vocabulary, f"{self.output_dir}/lang_vocabulary.json")


    def create_vocab(self, data:list[dict]):
        input_vocabulary = {"[PAD]": 0,
                            "[UNK]": 1}
        examples = [item["example"] for item in data]
        # On fusionne l'ensemble du corpus
        data_string = " ".join(examples).replace(self.delimiter, " ")
        # data_string = data_string[:100]
        splitted_text = re.split(self.delimiters_regex, data_string)

        n = 2
        for item in splitted_text:
            if item not in ["", None] and item.lower() not in input_vocabulary:
                input_vocabulary[item.lower()] = n
                n += 1
        reverse_input_vocabulary = {idx + n: token for idx, token in enumerate(splitted_text)}

        self.input_vocabulary = input_vocabulary
        self.reverse_input_vocabulary = reverse_input_vocabulary

        utils.serialize_dict(self.reverse_input_vocabulary, f"{self.output_dir}/reverse_input_vocabulary.json")
        utils.serialize_dict(self.input_vocabulary, f"{self.output_dir}/input_vocabulary.json")

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

    def deduce_weights(self):
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
        self.target_weights = torch.tensor([segment_content_weight, segment_boundary_weight, 0])

    def create_train_corpus(self):
        if self.data_augmentation:
            full_corpus = self.train_data + self.remove_punctuation(self.train_data)
        else:
            full_corpus = self.train_data
        train_padded_examples, train_langs, train_padded_targets = self.produce_corpus(full_corpus, debug=self.debug)
        self.train_padded_examples = utils.tensorize(train_padded_examples)
        self.train_langs = utils.tensorize(train_langs)
        self.train_padded_targets = utils.tensorize(train_padded_targets)
        self.deduce_weights()

    def create_test_corpus(self):
        """
        This function creates the test corpus, and uses the vocabulary of the train set to do so.
        Outputs: tensorized input, tensorized target, formatted input to ease accuracy computation.
        """
        if self.data_augmentation:
            full_corpus = self.test_data + self.remove_punctuation(self.test_data)
        else:
            full_corpus = self.test_data
        test_padded_examples, test_langs, test_padded_targets = self.produce_corpus(full_corpus, debug=self.debug)
        self.test_padded_examples = utils.tensorize(test_padded_examples)
        self.test_langs = utils.tensorize(test_langs)
        self.test_padded_targets = utils.tensorize(test_padded_targets)

    def create_dev_corpus(self):
        """
        This function creates the dev corpus, and uses the vocabulary of the train set to do so.
        Outputs: tensorized input, tensorized target, formatted input to ease accuracy computation.
        """
        if self.data_augmentation:
            full_corpus = self.dev_data + self.remove_punctuation(self.dev_data)
        else:
            full_corpus = self.dev_data
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


    def produce_corpus(self, data:list, debug=True) -> tuple:
        """
        This function takes the targets and creates the examples.
        """
        assert data != [], "Error with the data when producing the corpus"
        examples = []
        targets = []
        langs = []
        ids = []
        if debug:
            data = data[:500]
        for example in data:
            text = example['example']
            lang = example['lang']
            if self.filter_by_lang and lang != self.filter_by_lang:
                continue
            # Si on veut utiliser des embeddings pré-entraînés, il faut tokéniser avec le tokéniseur maison
            if self.use_pretrained_embeddings:
                try:
                    example, idents, target = utils.convertSentenceToSubWordsAndLabels(text, self.tokenizer, self.delimiter, max_length=380)
                    ids.append(idents)
                except TypeError as e:
                    print("Passing.")
                    continue
            else:
                target = []
                example = []
                text = text.replace(self.delimiter, " " + self.delimiter)
                as_tokens = re.split(self.delimiters_regex, text)
                for idx, token in enumerate(as_tokens):
                    if not token:
                        continue
                    if self.delimiter in token:
                        target.append("[SB]")
                        example.append(token.replace(self.delimiter, "").lower())
                    else:
                        target.append("[SC]")
                        example.append(token.lower())
                assert len(example) == len(target), "Length inconsistency"

            examples.append(example)
            targets.append(target)
            langs.append(self.lang_vocabulary[lang])


        self.max_length_examples = max([len(example) for example in examples])
        max_length_targets = max([len(target) for target in targets])
        if max_length_targets > 500:
            print("There is a problem with some line way too long. Please check the datasets.")
            print(np.mean([len(target) for target in targets]))
            print(max_length_targets)
            exit(0)

        if self.use_pretrained_embeddings is False:
            pad_value = "[PAD]"
            padded_examples = []
            padded_targets = []
            for example in examples:
                example_length = len(example)
                example = example + [pad_value for _ in range(self.max_length_examples - example_length)]
                example = ["[PAD]"] + example
                example = [self.input_vocabulary[token] for token in example]
                padded_examples.append(example)


            for target in targets:
                target_length = len(target)
                target = target + [pad_value for _ in range(max_length_targets - target_length)]
                target = ["[PAD]"] + target
                target = [self.target_classes[token] for token in target]
                padded_targets.append(target)
            return padded_examples, langs, padded_targets

        else:
            # On doit convertir la liste d'arrays vers un arrays, on concatène sur la dimension 0 (lignes)
            ids = np.concatenate(ids, axis=0)
            # targets = np.concatenate(targets, axis=0)
            targets = torch.stack(targets, dim=0)
            return ids, langs, targets

