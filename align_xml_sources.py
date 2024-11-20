import json
import os
import lxml.etree as etree
import string
from numpyencoder import NumpyEncoder
import sys
import numpy as np
import subprocess
import copy
import tqdm
import random
# import collatex
import aquilign.align.utils as utils
import aquilign.preproc.syntactic_tokenization as syntactic_tokenization
from transformers import BertTokenizer, AutoModelForTokenClassification
from aquilign.align.encoder import Encoder
from aquilign.align.aligner import Bertalign
import pandas as pd
import argparse
import glob


class MyClass:
    def __init__(self):
        
        pass

class XMLAligner:

    def __init__(self, 
                 hierarchy: list[str] = ["tei:div[@type='livre']", "tei:div[@type='partie']","tei:div[@type='chapitre']"],
                 id_attribute: str = "n",
                 input_dir: str = "",
                 main_wit:str = "Rome_W",
                 tokenization_models:dict = {},
                 device:str = "cpu",
                 remove_punct:bool = False):
        """
        @param hierarchy: la hiérarchie sur laquelle boucler dans le document XML. Les documents source et cible doivent être
        déjà alignés au niveau de la hiérarchie structurelle donnée. Sous la forme d'une liste d'expressions xpath.
        Exemple: ['tei:div[@type='livre']', 'tei:div[@type='partie']', 'tei:div[@type='chapitre']']
        @param id_attribute: l'attribut qui contient l'identifiant de la division minimale
        @param witnesses: la liste des chemins vers chacun des témoins. Chaque témoin doit avoir son sigle encodée dans le xml:id
        de la racine du fichier
        """
        self.tok_models = tokenization_models
        self.tei_namespace = 'http://www.tei-c.org/ns/1.0'
        self.tei = '{http://www.tei-c.org/ns/1.0}'
        self.TEINSMAP = {None: self.tei_namespace}
        self.ns_decl = {'tei': self.tei_namespace}
        self.hierarchy = hierarchy
        self.id_attribute = id_attribute
        self.input_dir = input_dir
        self.device = device
        self.witnesses = glob.glob(f"{input_dir}/*.xml")
        self.witnesses_id = []
        self.segmented_witnesses = None
        self.parsed_witnesses = {}
        self.global_text_dict = {}
        self.main_wit = main_wit
        self.remove_punct = remove_punct
        if self.input_dir[-1] == "/":
            self.out_dir = f"{input_dir}out"
        else:
            self.out_dir = f"{input_dir}/out"
        try:
            os.mkdir(self.out_dir)
        except FileExistsError:
            pass

        self.sentences_model = Encoder("LaBSE", device=device)
        self.max_align = 3
        
    
    def node_to_text(self, node):
        return " ".join(node.xpath("descendant::tei:w/descendant::text()", namespaces=self.ns_decl))
        
    
    def align_corpus(self, division=None):
        clause_mapping = {}
        for chapter, divs in self.global_text_dict.items():
            if division and chapter != division:
                if chapter != division:
                    continue
            print(chapter)
            main_wit_node = divs[self.main_wit]
            other_wit = [wit for wit in divs.keys() if wit != self.main_wit][0]
            other_wit_node = divs[other_wit]
            for wit in divs.keys():
                clause_mapping[wit] = {idx: ID for idx, ID in zip(range(len(divs[wit].xpath("descendant::tei:cl", namespaces=self.ns_decl))), 
                                                                                      divs[wit].xpath("descendant::tei:cl/@xml:id", namespaces=self.ns_decl))}
            
            all_main_wit_clauses =  main_wit_node.xpath("descendant::tei:cl", namespaces=self.ns_decl)
            main_wit_text = [self.node_to_text(clause) for clause in all_main_wit_clauses]
            other_wit_clauses = other_wit_node.xpath("descendant::tei:cl", namespaces=self.ns_decl)
            other_wit_text = [self.node_to_text(clause) for clause in other_wit_clauses]

            profile = 0
            if profile == 0:
                margin = True
                len_penality = True
            else:
                margin = False
                len_penality = True
            aligner = Bertalign(self.sentences_model,
                                main_wit_text,
                                other_wit_text,
                                max_align=self.max_align,
                                win=5, skip=-.2,
                                margin=margin,
                                len_penalty=len_penality,
                                device=self.device)
            aligner.align_sents()
            all_orig_clauses = divs[self.main_wit].xpath("descendant::tei:cl", namespaces=self.ns_decl)
            all_target_clauses = divs[other_wit].xpath("descendant::tei:cl", namespaces=self.ns_decl)
            for orig, target in aligner.result:
                orig_clause = [clause_mapping[self.main_wit][item] for item in orig]
                target_clause = [clause_mapping[other_wit][item] for item in target]
                pos_and_id_orig = list(zip(orig, orig_clause))
                pos_and_id_target = list(zip(target, target_clause))
                try:
                    corresp = f"#{' #'.join([item[1] for item in pos_and_id_orig])}"
                    for item in pos_and_id_target:
                        all_target_clauses[item[0]].set("corresp", corresp)
                except IndexError:
                    print("problemo")
                    pass
                
                try:
                    corresp = f"#{' #'.join([item[1] for item in pos_and_id_target])}"
                    for item in pos_and_id_orig:
                        all_orig_clauses[item[0]].set("corresp", corresp)
                except IndexError:
                    print("Problemo")
                    pass

    def parse_witnesses(self):
        for witness in self.segmented_witnesses:
            as_tree = etree.parse(witness)
            try:
                ID = as_tree.xpath("@xml:id")[0]
            except IndexError:
                print(f"Error with witness {witness}: please add sigla as xml:id")
            self.parsed_witnesses[ID] = as_tree

    def basic_validation(self):
        all_good = {ID:True for ID, witness in self.parsed_witnesses.items()}
        for ID, witness in self.parsed_witnesses.items():
            try:
                lang = witness.xpath("descendant::tei:profileDesc/tei:langUsage/tei:language/@ident", namespaces=self.ns_decl)[0]
            except IndexError:
                print(f"Error with witness {ID}: please add language specification in "
                      f"descendant::tei:profileDesc/tei:langUsage/tei:language/@ident")
                all_good[ID] = False
        
        if False in all_good.values():
            print("Validation not passed, exiting")
            exit(0)
        else:
            print("All tests passed. Continuing")
        

    def align_divisions(self):
        """
        Fonction qui permet de récupérer les divisions données d'un document en respectant une hiérarchie donnée.
        Elle nourrit le dictionnaire `global_text_dict` qui contient des listes de noeuds à aligner.
        """
        for wit_identifier, witness in self.parsed_witnesses.items():
            path = "descendant::" + self.hierarchy
            # On itère sur chaque niveau hiérarchique. Un système récursif devrait fonctionner mieux.
            for minimal_division in witness.xpath(path, namespaces=self.ns_decl):
                div_identifier = minimal_division.xpath(f"@{self.id_attribute}", namespaces=self.ns_decl)[0]
                try:
                    self.global_text_dict[div_identifier][wit_identifier] = minimal_division
                except KeyError:
                    self.global_text_dict[div_identifier] = {wit_identifier: minimal_division}
        print("Corpus imported")
        
        
    def split_sentences(self,tokens_per_example, words):
        as_nodes = [words[i:i + tokens_per_example] for i in range(0, len(words), tokens_per_example)]
        as_text = [[word.xpath("descendant::text()")[0] for word in sent] for sent in as_nodes]
        return as_text
    
    def get_words_from_node(self, node, words_per_batch):
        all_words, splitted = [], []
        for paragraph in node.xpath("descendant::tei:p", namespaces=self.ns_decl):
            current_tokens_as_nodes = paragraph.xpath("descendant::node()[self::tei:pc|self::tei:w]", namespaces=self.ns_decl)
            all_words.extend(current_tokens_as_nodes)
            all_words_text = paragraph.xpath("descendant::node()[self::tei:pc|self::tei:w]/text()", namespaces=self.ns_decl)
            splitted.extend(self.split_sentences(words_per_batch, current_tokens_as_nodes))
        return all_words, splitted
    
    def bert_segmentation(self, 
                      remove_punct=False,
                      verbose=False,
                      codelang=None, 
                      xml_node=None,
                      new_model=None,
                      tokenizer=None,
                    ident=None):
        """
        Performs tokenization with given model, tokenizer on given file
        """

        

        # get the file
        division = xml_node.xpath("@n")[0]
        # get the number of tokens per fragment to tokenize
        tokens_per_example = self.tok_models[codelang]["tokens_per_example"]
        # split the full input text as slices
        all_tokens, text = self.get_words_from_node(xml_node, tokens_per_example)
        # prepare the data
        # apply the tok process on each slice of text
        all_delimiters = []
        all_lenght = 0
        actual_pos = 0
        div_as_string = str()
        for idx, i in enumerate(tqdm.tqdm(text)):
            as_string = " ".join(i)
            # BERT-tok
            enco_nt_tok = tokenizer.encode(as_string, truncation=True, padding=True, return_tensors="pt")
            enco_nt_tok = enco_nt_tok.to(device)
            # get the predictions from the model
            predictions = new_model(enco_nt_tok)
            preds = predictions[0]
            # apply the functions
            bert_labels = get_labels_from_preds(preds)
            human_to_bert, bert_to_human = get_correspondence(i, tokenizer)
            delimiter_index, prediction = unalign_labels_and_get_index(human_to_bert=human_to_bert, predicted_labels=bert_labels,
                                        splitted_text=i)
            new_i = copy.deepcopy(i)
            [new_i.insert(pos, "\n") for pos in reversed(delimiter_index)]
            div_as_string += " ".join(new_i) + "\n"
            absolute_delimiter_index = delimiter_index
            delimiter_index = [item + actual_pos - 1 for item in delimiter_index]
            delimiter_index.append(len(i) - 1 + actual_pos)
            actual_pos += len(prediction) 
            all_delimiters.extend(delimiter_index)
            all_lenght += len(prediction)
        
        # On écrit le fichier de texte
        try:
            os.mkdir(f"{self.out_dir}/logs/")
        except FileExistsError:
            pass
        with open(f"{self.out_dir}/logs/{ident}_{division}_segm.txt", "w") as output_segmentation_file:
            output_segmentation_file.write(div_as_string)
        
        # On va passer d'une liste de délimiteurs à une liste d'intervales
        all_delimiters.insert(0, 0)
        delims_as_intervals = [(all_delimiters[n] + 1, all_delimiters[n+1]) for n, _ in enumerate(all_delimiters[:-1])]
        delims_as_intervals.insert(0, (0, delims_as_intervals[0][1]))
        # delims_as_intervals.append((delims_as_intervals[-1][1] + 1, all_lenght - 1))
        
        # Il faut supprimer le 2e élément, le code de création des intervalles est pas bon
        delims_as_intervals.pop(1)
        
        # On va créer des noeuds `tei:cl` en utilisant les informations de tokénisation. 
        # On va aussi s'occuper des noeuds informationnels autres (pb, etc). Suppose que ces
        # noeuds ne contiennent pas d'enfants tei:w ou tei:pc.
        for idx, (low_delim, high_delim) in enumerate(delims_as_intervals):
            clause = etree.Element(self.tei+"cl", nsmap=self.TEINSMAP)
            clause.set("{http://www.w3.org/XML/1998/namespace}id", generateur_id(6))
            try:
                all_tokens[high_delim].addnext(clause)
            except IndexError:
                print(high_delim)
                print(len(all_tokens))
                print(all_lenght)
                print("Error")
                exit(0)
            for token in all_tokens[low_delim:high_delim + 1]:
                try:
                    following = token.xpath("following-sibling::node()[1]", namespaces=self.ns_decl)[0]
                    if following.tag.replace("{http://www.tei-c.org/ns/1.0}", "") not in ['w', 'pc', 'div', 'cl'] \
                            and len(following.xpath("node()[not(self::text())]")) == 0:
                        print(f"Appending {following}")
                        clause.append(token)
                        clause.append(following)
                    else:
                        clause.append(token)
                except IndexError:
                    print("Passing")
                    pass
        
        # On va tester que les clauses ne se chevauchent pas et que tous les tei:w ont une clause parent:
        parent_test = xml_node.xpath("descendant::tei:p/descendant::tei:cl/tei:cl", namespaces=self.ns_decl)
        orphan_token_test = xml_node.xpath("descendant::tei:p/descendant::node()[self::tei:w or self::tei:pc][not(parent::tei:cl)]", namespaces=self.ns_decl)
        if len(parent_test) > 0:
            print("Nested clauses, please check encoding and code")
        elif len(orphan_token_test) > 0:
            print("Orphan token, please check encoding and code")
            print(etree.tostring(orphan_token_test[0]))
        else:
            print("Test passed.")

    
    
    def segment_corpus(self, division=None):
        """
        Tokénisation des documents XML en mots `tei:w|tei:pc`, puis en segments `tei:cl` 
        """
        for file in self.witnesses:
            command = ["java", 
                            "-jar", 
                            "aquilign/preproc/xsl/saxon9he.jar", 
                            "-xi:on", 
                            file,
                            "aquilign/preproc/xsl/tokenisation.xsl", 
                            f"output_dir={self.out_dir}"]
            subprocess.run(command)
            
            # On ajoute des xml:id pour pouvoir réinjecter après.
            for file in glob.glob(f"{self.out_dir}/*tokenized.xml"):
                as_tree = etree.parse(file)
                for token in as_tree.xpath("descendant::node()[self::tei:pc or self::tei:w]", namespaces=self.ns_decl):
                    token.set("{http://www.w3.org/XML/1998/namespace}id", generateur_id(6))
                
            regularisation = ["java", "-jar", 
                            "aquilign/preproc/xsl/saxon9he.jar",
                              "-xi:on", 
                              file,
                              "aquilign/preproc/xsl/regularisation.xsl", 
                              f"output_dir={self.out_dir}"]
            subprocess.run(regularisation)
            
        for file in glob.glob(f"{self.out_dir}/*regularise.xml"):
            as_tree = etree.parse(file)
            if self.remove_punct:
                print("Removing punctuation nodes.")
                all_pc = as_tree.xpath("descendant::tei:pc", namespaces=self.ns_decl)
                [pc.getparent().remove(pc) for pc in all_pc]
            ID = as_tree.xpath("@xml:id")[0]
            codelang = as_tree.xpath("descendant::tei:profileDesc/tei:langUsage/tei:language/@ident", namespaces=self.ns_decl)[0]
            model_path = self.tok_models[codelang]["model"]
            tokens_per_example = self.tok_models[codelang]["tokens_per_example"]
            tokenizer_name = self.tok_models[codelang]["tokenizer"]
            print(f"Using {model_path} model and {tokenizer_name} tokenizer.")
            new_model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=3)
            tokenizer = BertTokenizer.from_pretrained(tokenizer_name, max_length=tokens_per_example)
            new_model.to(self.device)
            for chapter in as_tree.xpath(f"descendant::{self.hierarchy.split('/')[-1]}", namespaces=self.ns_decl):
                print(chapter.xpath("@n")[0])
                if division and chapter.xpath("@n")[0] != division: 
                   continue
                self.bert_segmentation(codelang=codelang, xml_node=chapter, new_model=new_model, tokenizer=tokenizer, ident=ID)
            with open(f"{self.out_dir}/{ID}.phrased.xml", "w") as output_file:
                output_file.write(etree.tostring(as_tree, pretty_print=True, encoding="utf8").decode('utf8'))
        print("Done")


def unalign_labels_and_get_index(human_to_bert, predicted_labels, splitted_text, verbose=False):
    """
    Réaligne les tokens BERT et les tokens humains, et produit en sortie une liste d'index de délimiteurs
    """
    # On supprime SOS et EOS
    predicted_labels = predicted_labels[1:-1]
    if verbose:
        print(f"Prediction: {predicted_labels}")
        print(human_to_bert)
        print(splitted_text)
    realigned_list = []

    # itering on original text
    final_prediction = []
    any_one = []
    for index, value in enumerate(splitted_text):
        predicted = human_to_bert[index]
        # if no mismatch, copy the label
        if len(predicted) == 1:
            correct_label = predicted_labels[predicted[0]]
            if verbose:
                pass
                # print(f"Position {index}")
                # print(predicted_labels)
                # print(predicted[0])
                # print(correct_label)
        # mismatch
        else:
            correct_label = [predicted_labels[predicted[n]] for n in range(len(predicted))]
            if verbose:
                print(f"predicted labels mismatch :{predicted_labels}")
                print(f"len predicted mismatch {len(predicted)}")
                print(f"Corresponding labels in prediction: {correct_label}")
            # Dans ce cas on regarde s'il y a 1 dans n'importe quelle position des rangs correspondants:
            # on considère que BERT ne propose qu'une tokénisation plus granulaire que nous
            if any([n == 1 for n in correct_label]):
                correct_label = 1
            else:
                correct_label = 0
        final_prediction.append(correct_label)
        
    assert len(final_prediction) == len(splitted_text), "List mismatch"

    # On récupère les lieux où on doit couper
    index_list = [idx for idx, token in enumerate(final_prediction) if
                  token == 1 or (isinstance(token, list) and 1 in token)]
    return index_list, final_prediction


# correspondences between our labels and labels from the BERT-tok
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

#get the labels
def get_labels_from_preds(preds):
    bert_labels = []
    for pred in preds[-1]:
        label = [idx for idx, value in enumerate(pred) if value == max(pred)][0]
        bert_labels.append(label)
    return bert_labels



def generateur_lettre_initiale(chars=string.ascii_lowercase):
    # Génère une lettre aléatoire
    return random.choice(chars)[0]


def generateur_id(size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits) -> str:
    random_letter = generateur_lettre_initiale()
    random_string = ''.join(random.choice(chars) for _ in range(size))
    return random_letter + random_string


def main(input_dir, main_wit, hierarchy, id_attribute, tokenization_models, device, remove_punct):
    TEIAligner = XMLAligner(input_dir=input_dir,
                            hierarchy=hierarchy,
                            main_wit=main_wit,
                            id_attribute=id_attribute,
                            tokenization_models=tokenization_models,
                            device=device,
                            remove_punct=remove_punct)

    # division = "3.3.11"
    division = None
    TEIAligner.segment_corpus(division=division)
    exit(0)
    
    # On réécrit la liste des témoins pour aller chercher dans les fichers de sortie
    TEIAligner.segmented_witnesses = glob.glob(f"{TEIAligner.out_dir}/*phrased.xml")
    TEIAligner.parse_witnesses()
    TEIAligner.basic_validation()
    TEIAligner.align_divisions()
    TEIAligner.align_corpus(division=division)
    
    for wit_ID, tree in TEIAligner.parsed_witnesses.items():
        with open(f"data/XML_test/out/{wit_ID}.aligned.xml", "w") as output_sp:
            output_sp.write(etree.tostring(tree, pretty_print=True, encoding='utf8').decode('utf8'))
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default=None,
                        help="Input directory where the .txt files are stored. Each XML document must state its language in "
                             "a @ident attribute in a `tei:profileDesc/tei:langUsage/tei:language`")
    parser.add_argument("-o", "--out_dir", default="out",
                        help="Path to output dir.")
    parser.add_argument("-rmp", "--remove_punctuation", action='store_true', default=False,
                        help="Remove punctuation before tokenizing texts (default: False).")
    parser.add_argument("-mw", "--main_wit",
                        help="Pivot witness ID.",
                        type=str,
                        default="Val_S")
    parser.add_argument("-a", "--attribute", default="n",
                        help="Attribute used to identify each division.")
    parser.add_argument("-hr", "--hierarchy", default="tei:div[@type='livre']/tei:div[@type='partie']/tei:div[@type='chapitre']",
                        help="Hierarchy to get to base division (each text must be equally structured).")
    parser.add_argument("-d", "--device", default='cpu',
                        help="Device to be used (default: cpu).")
    parser.add_argument("-t", "--tokenizer", default='bert-based',
                        help="Tokenizer to be used (None, regexp, bert-based)")
    parser.add_argument("-l", "--corpus_limit", default=None,
                        help="Limit alignment to given proportion of each text (float)")
    args = parser.parse_args()
    attribute = args.attribute
    hierarchy = args.hierarchy
    input_dir = args.input_dir
    remove_punct =  args.remove_punctuation
    main_wit = args.main_wit
    device = args.device
    corpus_limit = args.corpus_limit
    if corpus_limit:
        corpus_limit = float(corpus_limit)
    tokenizer = args.tokenizer
    tokenization_models = {"fr":
                      {"model": "ProMeText/aquilign_french_segmenter",
                       "tokenizer": "dbmdz/bert-base-french-europeana-cased",
                       "tokens_per_example": 12},
                  "es": {"model": "ProMeText/aquilign_spanish_segmenter",
                         "tokenizer": "dccuchile/bert-base-spanish-wwm-cased",
                         "tokens_per_example": 30},
                  "it": {"model": "ProMeText/aquilign_italian_segmenter",
                         "tokenizer": "dbmdz/bert-base-italian-xxl-cased",
                         "tokens_per_example": 12},
                  "la": {"model": "ProMeText/aquilign_segmenter_latin",
                         "tokenizer": "LuisAVasquez/simple-latin-bert-uncased",
                         "tokens_per_example": 50}}
    
    assert tokenizer in ["None", "regexp",
                         "bert-based"], "Authorized values for tokenizer are: None, regexp, bert-based"
    assert input_dir != None, "Input dir is mandatory"
    
    
    main(input_dir, main_wit, hierarchy, attribute, tokenization_models, device, remove_punct=remove_punct)


