import re
import aquilign.preproc.tok_trainer_functions as functions
import torch

def tokenize(text,num):
    words = text.split(" ")
    return [' '.join(words[i:i+num]) for i in range(0, len(words), num)]


def get_best_step(results):
    """
    This function gets the best metrics of label 1 (= delimiter) given the results of the trainer.
    As for now it is the weighted average of precision (w=2) and recall (w=1) 
    """
    print(results)
    result_dict = {}
    for result in results:
        try:
            result_dict[result['step']] = {**result_dict[result['step']], **result}
        except KeyError:
            result_dict[result['step']] = result

    all_metrics = {}
    for key, value in result_dict.items():
        metric = (value['eval_precision'][1] + value['eval_recall'][1]*2)/3
        all_metrics[key] = metric

    best_step = next(step for step, metric in all_metrics.items() if metric == max(all_metrics.values()))
    print(f"Best step according to precision: {best_step}")
    return best_step, result_dict[best_step]

def remove_punctuation(text:str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text
    


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


def convertToWordsSentencesAndLabels(corpus:list, delimiter="£") -> (list, list):
    """
    This function take a corpus as a list of examples and returns the masks for each token as words
    """

    sentencesList = []
    sentencesAsLabels = []
    sentences_as_list_of_tokens = []
    for text in corpus:
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
    return sentencesList, sentencesAsLabels, sentences_as_list_of_tokens


# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertToSubWordsSentencesAndLabels(corpus, tokenizer, delimiter="£",  verbose=False):
    """
    This function takes a corpus and returns the tokenized corpus as subwords with their labels.
    """
    if verbose:
        print("Converting to sentences and labels")
    sentencesList = []
    sentencesAsLabels = []
    for text in corpus:
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
