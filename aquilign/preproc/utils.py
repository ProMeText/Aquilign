import re
import aquilign.preproc.tok_trainer_functions as functions
import torch
import glob

def tokenize(text,num):
    words = text.split(" ")
    return [' '.join(words[i:i+num]) for i in range(0, len(words), num)]


def get_best_precision(results):
    """
    This function gets the best precision of label 1 (= delimiter) given the results of the trainer
    """
    result_dict = {}
    for result in results:
        try:
            result_dict[result['step']] = {**result_dict[result['step']], **result}
        except KeyError:
            result_dict[result['step']] = result

    all_precisions = {}
    for key, value in result_dict.items():
        precision = value['eval_precision'][1]
        all_precisions[key] = precision

    best_step = next(step for step, precision in all_precisions.items() if precision == max(all_precisions.values()))
    print(f"Best step according to precision: {best_step}")
    return best_step, result_dict[best_step]

def remove_punctuation(text:str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text
    


def tokenize_words(sentence:str) -> list:
    """
    Cette fonction tokénise une phrase selon un certain nombre de marqueurs
    """
    words_delimiters = re.compile(r"[\.,;—:\?!’'«»“/\-]|[^\.,;—:\?!’'«»“/\-\s]+")
    sentenceAsList = re.findall(words_delimiters, sentence)
    return sentenceAsList


def convertToWordsSentencesAndLabels(corpus:list, delimiter="£") -> (list, list):
    """
    This function take a corpus as a list of examples and returns the masks for each token as words
    """

    sentencesList = []
    sentencesAsLabels = []
    for text, lang in corpus:
        sentenceAsList = tokenize_words(text)
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append(sentence)
    return sentencesList, sentencesAsLabels


def get_lang_mapping(tokenizer, void_metadata):
    """
    This function gets the token ID for pseudo tokens used as metadata for training and inference. Returns the tokenizer 
    which vocab can be modified to add a new token
    """
    langs = []
    lang_mapping = {}
    list_dirs = glob.glob("data/tokenisation/*")
    assert len(list_dirs) != 0, "Train data should be in data/tokenisation/{lang} where {lang} is the ISO code of the lang as 2 chars (es, it, en, fr, etc)"
    for dir in list_dirs:
        langs.append(dir.split("/")[-1])
    tokens_to_add = []
    for lang in langs:
        encoded_token = tokenizer.encode(lang)
        if len(encoded_token) > 3:
            tokenizer.add_tokens(lang)
            tokens_to_add.append(lang)
            lang_mapping[lang] = tokenizer.encode(lang)[1]
        else:
            lang_mapping[lang] = encoded_token[1]
            
    if void_metadata:
        all_langs = lang_mapping.items()
        all_tokens = lang_mapping.values()
        lang_mapping = {lang:all_tokens[0] for lang in all_langs}
    return lang_mapping, tokens_to_add
        


# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertToSubWordsSentencesAndLabels(corpus, tokenizer, delimiter="£",  verbose=False, add_lang_metadata=True):
    """
    This function takes a corpus and returns the tokenized corpus as subwords with their labels, adding lang metadata.
    Returns the data and the tokenizer which vocab can be updated (worth a class transformation)
    """
    if verbose:
        print("Converting to sentences and labels")
    sentencesList = []
    sentencesAsLabels = []
    for text, lang in corpus:
        sentenceAsList = tokenize_words(text)
        # We start with 2 that represents the lang metadata
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append((sentence, lang))
    num_max_length = functions.get_token_max_length(sentencesList, tokenizer) + 1
    out_toks_and_labels = []
    # langs = "es it fr"
    # [4, 1058, 4290, 2123, 5]
    lang_mapping, tokens_to_add = get_lang_mapping(tokenizer, void_metadata=add_lang_metadata)
    tokenizer.add_tokens(tokens_to_add)
    for (text, lang), labels in zip(sentencesList, sentencesAsLabels):
        toks = tokenizer(text, padding="max_length", max_length=num_max_length, truncation=False,
                         return_tensors="pt")
        # get the text with the similar splits as for the creation of the data
        tokens = tokenize_words(text)
        # get the index correspondences between text and tok text
        corresp = functions.get_index_correspondence(tokens, tokenizer)
        
        # aligning the label and add label 3 at index 2 (=metadata)
        new_labels = functions.align_labels(corresp, labels)
        
        
        # get the length of the tensor
        
        # We add the metadata:
        # We first need to add a value to the attention mask tensors
        attention_mask:list = toks['attention_mask'].squeeze().tolist()
        attention_mask.insert(0, 1)
        attention_mask = torch.tensor(attention_mask)
        
        # We then add the lang metadata as a token, as index 1 (index 0 is CLS)
        squeezed_toks = (toks['input_ids'].squeeze())
        squeezed_toks = squeezed_toks.tolist()
        squeezed_toks.insert(1, lang_mapping[lang])
        squeezed_toks = torch.tensor(squeezed_toks)
        sq_as_list = squeezed_toks.tolist()
        # print(list(zip([tokenizer.convert_ids_to_tokens(t) for t in tokenizer.encode(text)], [t for t in tokenizer.encode(text)], [item for item in new_labels])))
        
        ### insert 2 for in the new_labels in order to get tensors with the same size !
        if len(squeezed_toks) == len(new_labels):
            pass
        else:
            diff = len(squeezed_toks) - len(new_labels)
            for elem in range(diff):
                new_labels.append(2)
        assert len(squeezed_toks) == len(new_labels), f"Mismatch.\n" \
                                           f"Text: {text}\n" \
                                           f"{(squeezed_toks.tolist())}\n" \
                                           f"{(new_labels)}\n" \
                                           f"squeezed_toks: {len(squeezed_toks)}\n" \
                                           f"new labels: {len(new_labels)}"
        label = torch.tensor(new_labels)
        
        assert all([type(current_data) == type(label) for current_data in [attention_mask, squeezed_toks]]), "Datatype error"
        
        out_toks_and_labels.append({'input_ids': squeezed_toks,
                                    'attention_mask': attention_mask,
                                    'labels': label})
    return out_toks_and_labels, tokenizer
