import glob
import json
import re
import traceback
import unicodedata
import torch
import os
import time
import tabulate
import statistics
import jsonschema
import random

def augment_data(corpuses:tuple, augment_proportion:float=1.0):
    assert augment_proportion >= 0.0 and augment_proportion <= 1.0, 'Augment proportion should be between 0 and 1'
    augmented_data = []
    for corpus in corpuses:
        noised_corpus = []
        for example in corpus:
            example_text = example['example']
            print(example_text)
            noised = apply_noise(example_text)
            print(noised)
            print("---")
            noised_example = {'example': noised, 'lang': example['lang']}
            noised_corpus.append(noised_example)
        random.shuffle(noised_corpus)
        summed_corpus = corpus + noised_corpus[:round(len(corpus)*augment_proportion)]
        augmented_data.append(summed_corpus)

    return augmented_data
# -------------------------------------------------------------------
# Character confusion map (simulate typical OCR/HTR confusions)
# -------------------------------------------------------------------
CONFUSION_MAP = {
    "u": "v", "v": "u",
    "i": "l", "l": "i",
    "m": "n", "n": "m",
    "c": "e", "e": "c",
    "t": "c", "s": "f"
}


def char_confusion(word, prob=0.1):
    """
    Replace characters in a word with common confusions.

    prob (float): probability between 0 and 1
        - prob=0.1 → each character has a 10% chance of being replaced
        - higher values → stronger corruption

    Example:
        "dominus" -> "dominvs"
    """
    return "".join(
        CONFUSION_MAP.get(c, c) if random.random() < prob else c
        for c in word
    )


def random_delete_char(word, prob=0.1):
    """
    Randomly delete characters in a word.

    prob (float): probability each character is dropped.

    Example:
        "dominus" -> "domins"
    """
    if len(word) <= 1 or word[-2] == "£":
        return word
    return "".join([c for c in word if random.random() > prob])


def drop_punctuation(text, prob=0.3):
    """
    Randomly remove punctuation marks from the text.

    prob (float): probability each punctuation mark is deleted.

    Example:
        "In principio, Deus." -> "In principio Deus"
    """
    return re.sub(r"([.,;:!?])",
                  lambda m: "" if random.random() < prob else m.group(1),
                  text)


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def apply_noise(text, noise_level="light"):
    """
    Apply a configurable set of noise transformations to text.

    Args:
        text (str): input text
        labels (list): segmentation labels (passed through unchanged)
        noise_level (str): 'light' | 'medium' | 'heavy'
            Controls the intensity of the noise:
                - light  = few modifications
                - medium = balanced corruption
                - heavy  = strong corruption

    Returns:
        noisy_text: noisy text string
    """

    # Define probabilities depending on noise level
    if noise_level == "light":
        p_del, p_conf, p_punct = 0.01, 0.05, 0.1
    elif noise_level == "medium":
        p_del, p_conf, p_punct = 0.1, 0.1, 0.3
    else:  # heavy
        p_del, p_conf, p_punct = 0.2, 0.2, 0.5
    noisy_words = []
    for w in text.split():
        w = random_delete_char(w, prob=p_del)
        w = char_confusion(w, prob=p_conf)
        noisy_words.append(w)

    noisy_text = " ".join(noisy_words)

    # Apply punctuation noise
    noisy_text = drop_punctuation(noisy_text, prob=p_punct)

    return noisy_text


# -------------------------------------------------------------------
# OPTIONAL FUTURE EXTENSION:
# If you want fine-grained control instead of only presets,
# change the function signature to:
#
# def apply_noise(text, labels=None, noise_level="medium",
#                 p_del=None, p_conf=None, p_punct=None):
#
# Then override defaults like this:
# p_del = p_del if p_del is not None else p_del_default
# ...
#
# This way you can call:
# apply_noise(text, p_del=0.2, p_conf=0.05, p_punct=0.4)
# -------------------------------------------------------------------




def remove_punctuation(text: str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text


def remove_punctuation_from_corpus(data:dict)-> dict:
    """
    This function removes the punctuation from the json formated corpus.
    """
    updated_list_of_examples = []
    for example in data["examples"]:
        without_punct = remove_punctuation(example["example"])
        new_example = {"example": without_punct,
                       "lang": example["lang"]}
        updated_list_of_examples.append(new_example)
    data["examples"] = updated_list_of_examples
    return data


def unicode_normalize_string(string:str) -> str:
    return unicodedata.normalize("NFC", string)


def unicode_normalize_corpus(data:list) -> str:
    normalized_examples = []
    for example in data["examples"]:
        normalized_examples.append({"example": unicode_normalize_string(example["example"]),
                                    "lang": example["lang"]})
    data["examples"] = normalized_examples
    return data


def test_data(data: dict, label: str, schema: dict) -> None:
    """
    This function tests if the training data can be correctly parsed. If not, it stops the training and exits.
    """
    # We first test if the data format is OK
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        print(f"The data is not valid. Please make sure the structure follows the example in the README. "
              f"Error: {e}")
        exit(0)

    delimiter = data['metadata']["delimiter"]
    regexp = re.compile(rf"{delimiter}([^A-Za-zẽ\d+çÇÉÁÍòãÓȝïÈèÚéçáíƷàÞóúýþ&\(\)\[\].·,,;¿?¦“…/’‘>«»'¡\-—–―\"])\s?")
    valid_list = []
    for idx, example in enumerate(data["examples"]):
        example_text = example["example"]
        search = re.search(regexp, example_text)
        if search:
            print("\n")
            print(f"Problem with some example (example {idx + 1}):\n{example_text}")
            print(search)
            print("\n")
            valid_list.append(False)

    if any([item is False for item in valid_list]):
        print(f"Test on {label} failed. Exiting")
    else:
        print(f"Test on {label} passed.")

# dataset class which fits the requirements
class SentenceBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, texts_and_labels):
        self.texts_and_labels = texts_and_labels

    def __len__(self):
        return len(self.texts_and_labels)

    def __getitem__(self, idx):
        # get the max length of the training set in order to have the good feature to put in tokenizer
        # current text (one line, ie 12 tokens [before automatic BERT tokenization])
        return self.texts_and_labels[idx]


# function who gets the max length of tokenized text, used then in the class SentenceBoundaryDataset
def get_token_max_length(train_texts, tokenizer):
    lengths_list = []
    for text in train_texts:
        tok_text = tokenizer(text, return_tensors='pt')
        # get the length for every tok text
        tensor_length = (tok_text['input_ids'].squeeze())
        length = tensor_length.shape[0]
        lengths_list.append(length)
    # get the max value from the list
    max_length = max(lengths_list)
    return max_length

# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertToSubWordsSentencesAndLabels(corpus, tokenizer, delimiter="£",  verbose=False):
    """
    This function takes a corpus and returns the tokenized corpus as subwords with their labels.
    :param corpus: A list of dicts of the shape
                            {"example": "tutti e tre £e domandarono quali armi il cavaliere ne portò £quand’e'",
                             "lang": "it"}
    """
    if verbose:
        print("Converting to sentences and labels")
    sentencesList = []
    sentencesAsLabels = []
    for example in corpus:
        text = example["example"]
        sentenceAsList = tokenize_words(text, delimiter)
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append(sentence)
    num_max_length = get_token_max_length(sentencesList, tokenizer)
    out_toks_and_labels = []
    for text, labels in zip(sentencesList, sentencesAsLabels):
        toks = tokenizer(text, padding="max_length", max_length=num_max_length, truncation=True,
                         return_tensors="pt")

        # get the text with the similar splits as for the creation of the data
        tokens = tokenize_words(text, delimiter)
        # get the index correspondences between text and tok text
        corresp = get_index_correspondence(tokens, tokenizer)
        # aligning the label
        new_labels = align_labels(corresp, labels, text)
        # get the length of the tensor
        sq = (toks['input_ids'].squeeze())
        ### insert 2 for in the new_labels in order to get tensors with the same size !
        if len(sq) == len(new_labels):
            pass
        else:
            diff = len(sq) - len(new_labels)
            for elem in range(diff):
                new_labels.append(2)
        assert len(sq) == len(new_labels), f"Mismatch.\n" \
                                           f"Text: {text}\n" \
                                           f"{(sq.tolist())}\n" \
                                           f"{(new_labels)}\n" \
                                           f"sq: {len(sq)}\n" \
                                           f"new labels: {len(new_labels)}"
        # tensorize the new labels
        label = torch.tensor(new_labels)
        out_toks_and_labels.append({'input_ids': toks['input_ids'].squeeze(),
                                    'attention_mask': toks['attention_mask'].squeeze(),
                                    'labels': label})
    return out_toks_and_labels


def json_corpus_to_lines(corpus: str, keep_punct, return_delimiter=False) -> list[dict]:
    """
    This function imports the json files and performs a first validation of the data structure. It returns
    the examples as a liste of dictionnaries with the example and its language information
    """
    with open(corpus, "r") as corpus_file:
        examples = json.load(corpus_file)
        if keep_punct is False:
            examples = remove_punctuation_from_corpus(examples)

    examples = unicode_normalize_corpus(examples)
    # Let's perform some tests
    with open("aquilign/segmenter/dataSchema.json", "r") as input_file:
        JsonSchema = json.load(input_file)
    test_data(examples, corpus, schema=JsonSchema)

    if return_delimiter:
        return examples["examples"], examples["metadata"]["delimiter"]
    else:
        return examples["examples"]


def write_accuracy(message, path):
    with open(f"{path}accuracies.txt", "a") as output_file:
        output_file.write(message)

def write_to_file(message, path):
    with open(path, "a") as output_file:
        output_file.write(message + "\n")
def remove_file(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def append_to_file(content, path):
    with open(path, "a") as output_file:
        output_file.write(content + "\n")

class Timer:
    def __init__(self):
        self.start = time.time()
        self.started_time = dict()
        self.stopped_time = dict()

    def start_timer(self, timer_name):
        self.started_time[timer_name] = time.time()

    def stop_timer(self, timer_name):
        self.stopped_time[timer_name] = time.time()
        print(self.stopped_time[timer_name] - self.started_time[timer_name])

    def lapse(self):
        lapse = time.time()
        print(lapse - self.start)

def remove_files(path:list|str):
    if isinstance(path, list):
        pass
    else:
        path = glob.glob(path)
    for file in path:
        try:
            os.remove(file)
        except OSError:
            pass

def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def format_results(results, header, print_to_term=True):
    to_print = tabulate.tabulate(results, headers=header, tablefmt='orgtbl')
    if print_to_term:
        print(to_print)
    else:
        return to_print


# function to get the index of the tokens after BERT tokenization
def get_index_correspondence(sent, tokenizer):
    correspondence = [(0,0)]
    for word in sent:
        (raw_end, expand_end) = correspondence[-1]
        tokenized_word = tokenizer.tokenize(word)
        correspondence.append((raw_end+1, expand_end+len(tokenized_word)))
    return correspondence


def tokenize_words(sentence:str, delimiter) -> list:
    """
    Cette fonction tokénise une phrase selon un certain nombre de marqueurs
    """
    if f"{delimiter} " in sentence:
        print("Problem with delimiter in sentence. Replacing")
        print(sentence)
        print("---")
        sentence = sentence.replace(f"{delimiter} ", delimiter)
    words_delimiters = re.compile(r"[\.,;—:\?\!’'”«»“/\-]|[^\.,;—”:\?\!’'«»“/\-\s]+")
    sentenceAsList = re.findall(words_delimiters, sentence)
    if delimiter in sentenceAsList:
        # Some workaround for when the delimiter is used on a token in the list of word delimiters.
        alone_delim_index = next(idx for idx, token in enumerate(sentenceAsList) if token == delimiter)
        try:
            to_merge = sentenceAsList.pop(alone_delim_index + 1)
        except IndexError:
            print(f"Index error on sentence:\n '{sentence}'")
            if sentence[-1] == delimiter:
                print("Last char of the sentence should not be the delimiter. Exiting")
            exit(0)
        sentenceAsList[alone_delim_index] = delimiter + to_merge
    return sentenceAsList



def align_labels(corresp, orig_labels, text, verbose=False):
    # function to align labels between the tokens in input and the tokenized tokens
    if verbose:
        print(corresp)
        print(len(corresp))
        print(orig_labels)
        print(f" Orig labels: {len(orig_labels)}")
    new_labels = [0 for _ in range(corresp[-1][1])]
    for index, label in enumerate(orig_labels):
        # label which is interesting : 1
        if label == 1:
            try:
                if len(new_labels) == corresp[index][1]:
                    new_labels[(corresp[index][1]) - 1] = 1
                else:
                    try:
                        new_labels[(corresp[index][1])] = 1
                    except IndexError:
                        print(f"Error with example:\n {text}.\n"
                              f"Exiting.")
                        print(new_labels)
                        exit(0)

            except IndexError as e:
                print("Error.")
                print(new_labels)
                print(len(new_labels))
                print(traceback.format_exc())
                print(f"Example: {text}")
                print(len(text.split()))
                print(f"Problematic index: {index}")
                print(len(corresp))
                exit(0)
        else:
            pass
    # for special tokens (automatically added by BERT tokenizer), value of 2
    new_labels.insert(0, 2)
    new_labels.append(2)
    return new_labels


# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertSentenceToSubWordsAndLabels(orig_sentence, tokenizer, delimiter="£",  max_length=380, verbose=False, output_masks=False):
    """
    This function takes a corpus and returns the tokenized corpus as subwords with their labels.
    :param corpus: A list of dicts of the shape
                            {"example": "tutti e tre £e domandarono quali armi il cavaliere ne portò £quand’e'",
                             "lang": "it"}
    """
    sentence = orig_sentence.replace(delimiter, f" {delimiter}")
    TokenizedSentence = tokenize_words(sentence, delimiter)
    example = tokenizer.tokenize(sentence)
    if len(example) > max_length:
        print("Example too long, removed:")
        print(sentence)
        return
    masks = []
    for token in TokenizedSentence:
        if delimiter in token:
            masks.append(1)
        else:
            masks.append(0)
    sentence_no_delim = sentence.replace(delimiter, "")

    toks = tokenizer(sentence_no_delim, padding="max_length", max_length=max_length, truncation=True,
                     return_tensors="pt")

    if verbose:
        print(orig_sentence)
        print(TokenizedSentence)
        print(len(TokenizedSentence))
        reverse_vocab = {value: key for key, value in tokenizer.get_vocab().items()}
        splitted_text = [reverse_vocab[item] for item in toks['input_ids'].tolist()[0]]
        print(splitted_text[:splitted_text.index("[PAD]")])
    # get the index correspondences between text and tok text
    tokens = tokenize_words(sentence_no_delim, delimiter)
    corresp = get_index_correspondence(tokens, tokenizer)
    # aligning the label
    new_labels = align_labels(corresp, masks, sentence_no_delim, verbose=verbose)
    # get the length of the tensor
    sq = (toks['input_ids'].squeeze())
    if verbose:
        reverse_vocab = {value: key for key, value in tokenizer.get_vocab().items()}
        splitted_text = [reverse_vocab[item] for item in toks['input_ids'].tolist()[0]][:len(new_labels)]
        print(list(zip(splitted_text, new_labels)))
    ### insert 2 (padding) for in the new_labels in order to get tensors with the same size !
    if len(sq) == len(new_labels):
        pass
    else:
        diff = len(sq) - len(new_labels)
        for elem in range(diff):
            new_labels.append(2)
    assert len(sq) == len(new_labels), f"Mismatch.\n" \
                                       f"Text: {sentence}\n" \
                                       f"{(sq.tolist())}\n" \
                                       f"{(new_labels)}\n" \
                                       f"sq: {len(sq)}\n" \
                                       f"new labels: {len(new_labels)}"
    # tensorize the new labels
    label = torch.tensor(new_labels)
    if output_masks:
        return example, toks['attention_mask'].squeeze(), toks['input_ids'], label
    else:
        return example, toks['input_ids'], label




def tensorize(array):
    tensorized_array = torch.tensor(array)
    return tensorized_array

def read_to_dict(path):
    with open(path, "r") as f:
        return json.load(f)

def serialize_dict(dictionnary, path):
    with open(path, "w") as f:
        json.dump(dictionnary, f, indent=3)

def encode_text(input_text:str, token_to_idx:dict, delimiters):
	return [token_to_idx[item.lower()] for item in re.split(delimiters, input_text) if item not in [None, '']]

def decode_text():
	pass

def identify_ambiguous_tokens(tokens,
                              labels,
                              id_to_word,
                              word_to_id):

    out_dict = {token: {} for token in tokens}
    for token, label in zip(tokens, labels):
        try:
            out_dict[token][label] += 1
        except KeyError:
            out_dict[token][label] = 1
    ambiguous_tokens = {id_to_word[token]: labels for token, labels in out_dict.items() if len(labels) > 1}
    # print(ambiguous_tokens)
    ambiguous_with_taux_ambiguite = [(token, {"label": label, "taux_ambiguite": taux_ambiguite(label)}) for token, label in ambiguous_tokens.items() if all(item > 2 for item in label.values())]
    ambiguous_with_taux_ambiguite.sort(key=lambda x: x[1]["taux_ambiguite"], reverse=True)
    ambiguous_tokens = [token[0] for token in ambiguous_with_taux_ambiguite]
    ambiguous_ids = [word_to_id[token] for token in ambiguous_tokens]
    return ambiguous_ids

def taux_ambiguite(label):
    values = label.values()
    pstdev = statistics.pstdev(values)
    return  statistics.mean(values)**2 / (pstdev + 1)



def get_labels_from_preds(preds):
    bert_labels = []
    for pred in preds[-1]:
        label = [idx for idx, value in enumerate(pred) if value == max(pred)][0]
        bert_labels.append(label)
    return bert_labels


def get_correspondence(sent, tokenizer, verbose=False):
    out = {}
    tokenized_index = 0
    for index, word in enumerate(sent):
        # print(tokenizer.tokenize(word))
        tokenized_word = tokenizer.tokenize(word)
        if verbose:
            print(tokenized_word)
        out[index] = tuple(item for item in range(tokenized_index, tokenized_index + len(tokenized_word)))
        tokenized_index += len(tokenized_word)
    human_split_to_bert = out
    bert_split_to_human_split = {value: key for key, value in human_split_to_bert.items()}
    return human_split_to_bert, bert_split_to_human_split


def apply_labels(text:list, labels:list):
    tokenized_sentence = " ".join([element if labels[index] != 1 else f"\n{element}" for index, element in enumerate(text)]).split("\n")
    return tokenized_sentence

def unalign_labels(human_to_bert, predicted_labels, splitted_text, verbose=False):
    predicted_labels = predicted_labels[1:-1]
    if verbose:
        print(f"Prediction: {predicted_labels}")
        print(human_to_bert)
        print(splitted_text)
    realigned_list = []

    # itering on original text
    final_prediction = []
    for index, value in enumerate(splitted_text):
        predicted = human_to_bert[index]
        # if no mismatch, copy the label
        if len(predicted) == 1:
            correct_label = predicted_labels[predicted[0]]
            if verbose:
                print(f"Position {index}")
                print(predicted_labels)
                print(predicted[0])
                print(correct_label)
        # mismatch
        else:
            correct_label = [predicted_labels[predicted[n]] for n in range(len(predicted))]
            if verbose:
                print(f"predicted labels mismatch :{predicted_labels}")
                print(f"len predicted mismatch {len(predicted)}")
                print(f"Corresponding labels in prediction: {correct_label}")
            # Dans ce cas on regarde s'il y a 1 dans n'importe quelle position des rangs correspondants:
            # on considère que BERT ne propose qu'une tokénisation plus importante que nous
            if any([n == 1 for n in correct_label]):
                correct_label = 1
        final_prediction.append(correct_label)

    assert len(final_prediction) == len(splitted_text), "List mismatch"

    tokenized_sentence = " ".join(
        [element if final_prediction[index] != 1 else f"\n{element}" for index, element in enumerate(splitted_text)]).split("\n")
    if verbose:
        print(f'final prediction {final_prediction}')
        print(tokenized_sentence)
    return tokenized_sentence

def format_examples(text, tokens_per_example, regexp, lang):
    regexp = re.compile(regexp)
    words = [item for item in re.split(regexp, text) if item]
    splitted = [words[i:i + tokens_per_example] for i in range(0, len(words), tokens_per_example)]
    examples = [" ".join(example) for example in splitted]
    langs = [lang for _ in range(len(splitted))]
    as_zip = zip(examples, langs)
    return as_zip

