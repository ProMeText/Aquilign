import json
import re
import torch
import os
import time
import tabulate

def write_accuracy(message, path):
    with open(f"{path}accuracies.txt", "a") as output_file:
        output_file.write(message)


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


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def format_results(results, header):
    to_print = tabulate.tabulate(results, headers=header, tablefmt='orgtbl')
    print(to_print)


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
    words_delimiters = re.compile(r"[\.,;—:\?!’'«»“/\-]|[^\.,;—:\?!’'«»“/\-\s]+")
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



def align_labels(corresp, orig_labels, text):
# function to align labels between the tokens in input and the tokenized tokens
    new_labels = [0 for r in range(corresp[-1][1])]
    for index, label in enumerate(orig_labels):
        # label which is interesting : 1
        if label == 1:
            try:
                if len(new_labels) == corresp[index][1]:
                    new_labels[(corresp[index][1]) - 1] = 1
            except IndexError:
                print(new_labels)
                print("Error.")
                exit(0)
            else:
                try:
                    new_labels[(corresp[index][1])] = 1
                except IndexError:
                    print(f"Error with example:\n {text}.\n"
                          f"Exiting.")
        else:
            pass
    # for special tokens (automatically added by BERT tokenizer), value of 2
    new_labels.insert(0, 2)
    new_labels.append(2)
    return new_labels


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
def convertSentenceToSubWordsAndLabels(sentence, tokenizer, delimiter="£",  max_length=380, verbose=False):
    """
    This function takes a corpus and returns the tokenized corpus as subwords with their labels.
    :param corpus: A list of dicts of the shape
                            {"example": "tutti e tre £e domandarono quali armi il cavaliere ne portò £quand’e'",
                             "lang": "it"}
    """
    sentence = sentence.replace(delimiter, f" {delimiter}")
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

    sentence = sentence.replace(delimiter, "")

    toks = tokenizer(sentence, padding="max_length", max_length=max_length, truncation=True,
                     return_tensors="pt")
    # get the index correspondences between text and tok text
    tokens = tokenize_words(sentence, delimiter)
    corresp = get_index_correspondence(tokens, tokenizer)
    # aligning the label
    new_labels = align_labels(corresp, masks, sentence)
    # get the length of the tensor
    sq = (toks['input_ids'].squeeze())
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
    return example, toks['input_ids'], label




def tensorize(array):
    tensorized_array = torch.tensor(array)
    return tensorized_array


def serialize_dict(dictionnary, path):
    with open(path, "w") as f:
        json.dump(dictionnary, f)

def encode_text(input_text:str, token_to_idx:dict, delimiters):
	return [token_to_idx[item.lower()] for item in re.split(delimiters, input_text) if item not in [None, '']]

def decode_text():
	pass