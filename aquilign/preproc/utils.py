import re
import aquilign.preproc.tok_trainer_functions as functions
import torch
import jsonschema
import json
import unicodedata

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

def json_corpus_to_lines(corpus:str, keep_punct, return_delimiter=False)-> list[dict]:
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
    with open("aquilign/tokenizer/dataSchema.json", "r") as input_file: 
        JsonSchema = json.load(input_file)
    test_data(examples, corpus, schema=JsonSchema)
    
    if return_delimiter:
        return examples["examples"], examples["metadata"]["delimiter"]
    else:
        return examples["examples"]

def test_data(data:dict, label:str, schema:dict) -> None:
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
