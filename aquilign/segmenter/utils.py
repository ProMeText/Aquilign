import re
import torch
import os
import time


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

def convertToWordsSentencesAndLabels(corpus:list[dict|str], delimiter="£") -> (list, list):
    """
    This function take a corpus as a list of examples and returns the masks for each token as words
    """

    sentencesList = []
    sentencesAsLabels = []
    sentences_as_list_of_tokens = []
    langsList = []
    for example in corpus:
        # On peut avoir une liste de dictionnaires ou de chaînes de caractères
        if isinstance(example, dict):
            text = example["example"]
            langsList.append(example["lang"])
        else:
            text = example
        sentenceAsList = tokenize_words(text, delimiter)
        sentences_as_list_of_tokens.append(sentenceAsList)
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append(sentence)
    return sentencesList, sentencesAsLabels, sentences_as_list_of_tokens, langsList


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
    num_max_length = functions.get_token_max_length(sentencesList, tokenizer)
    out_toks_and_labels = []
    for text, labels in zip(sentencesList, sentencesAsLabels):
        toks = tokenizer(text, padding="max_length", max_length=num_max_length, truncation=True,
                         return_tensors="pt")

        # get the text with the similar splits as for the creation of the data
        tokens = tokenize_words(text, delimiter)
        # get the index correspondences between text and tok text
        corresp = functions.get_index_correspondence(tokens, tokenizer)
        # aligning the label
        new_labels = functions.align_labels(corresp, labels, text)
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




def tensorize(array):
    tensorized_array = torch.tensor(array)
    return tensorized_array




def encode_text(input_text:str, token_to_idx:dict, delimiters):
	return [token_to_idx[item.lower()] for item in re.split(delimiters, input_text) if item not in [None, '']]

def decode_text():
	pass