import json
import os
import lxml.etree as etree
import string
from numpyencoder import NumpyEncoder
import sys
import numpy as np
import random
# import collatex
import aquilign.align.graph_merge as graph_merge
import aquilign.align.utils as utils
import aquilign.preproc.tok_apply as tokenize
import aquilign.preproc.syntactic_tokenization as syntactic_tokenization
from aquilign.align.encoder import Encoder
from aquilign.align.aligner import Bertalign
import pandas as pd
import argparse
import glob

class TEIAligner():
    """
    L'aligneur, qui prend des fichiers TEI en entrée et les tokénise
    """

    def __init__(self, files_path: dict, tokenize=False):
        self.tei_ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        with open("bertalign/delimiters.json", "r") as input_json:
            dictionary = json.load(input_json)
        single_tokens_punctuation = [punct for punct in dictionary['punctuation'] if len(punct) == 1]
        multiple_tokens_punctuation = [punct for punct in dictionary['punctuation'] if len(punct) != 1]
        single_token_punct = "".join(single_tokens_punctuation)
        multiple_tokens_punct = "|".join(multiple_tokens_punctuation)
        punctuation_subregex = f"({multiple_tokens_punct}|[{single_token_punct}])"
        tokens_subregex = "(" + " | ".join(dictionary['word_delimiters']) + ")"
        self.target_parsed_files = {ls}
        self.main_parsed_file = None
        files = files_path['target_files']
        main_file = files_path['main_file']
        # Faut-il tokéniser le document (mots, syntagmes) ?
        if tokenize:
            print("Tokenizing")
            tokenizer = tokenization.Tokenizer(regularisation=True)
            tokenizer.tokenisation(path=main_file, punctuation_regex=punctuation_subregex)
            regularized_file = main_file.replace('.xml', '.regularized.xml')
            utils.pretty_print_xml_tree(regularized_file)
            print("Word tokenization done.")
            tokenizer.subsentences_tokenisation(path=regularized_file, delimiters=tokens_subregex)
            self.main_file = (main_file, tokenizer.tokenized_tree)
            for file in files:
                tokenizer.tokenisation(path=file, punctuation_regex=punctuation_subregex)
                print("Word tokenization done.")
                regularized_file = file.replace('.xml', '.regularized.xml')
                utils.pretty_print_xml_tree(regularized_file)
                tokenizer.subsentences_tokenisation(path=regularized_file, delimiters=tokens_subregex)
                self.target_parsed_files[file] = tokenizer.tokenized_tree
            print("Done")
        else:
            self.target_parsed_files = {file: etree.parse(file) for file in files}
            self.main_file = (main_file, etree.parse(main_file))

    def alignementMultilingue(self):
        main_file_path, main_file_tree = self.main_file
        for path, tree in self.target_parsed_files.items():
            for chapter in tree.xpath("descendant::tei:div[@type='chapitre']", namespaces=self.tei_ns):
                source_tokens, target_tokens = list(), list()
                target_dict = {}
                source_dict = {}
                chapter_n = chapter.xpath("@n")[0]
                part_n = chapter.xpath("ancestor::tei:div[@type='partie']/@n", namespaces=self.tei_ns)[0]
                book_n = chapter.xpath("ancestor::tei:div[@type='livre']/@n", namespaces=self.tei_ns)[0]
                print(f"Treating {book_n}-{part_n}-{chapter_n}")
                for index, phrase in enumerate(chapter.xpath("descendant::tei:phr", namespaces=self.tei_ns)):
                    ident = utils.generateur_id(6)
                    phrase.set('{http://www.w3.org/XML/1998/namespace}id', ident)
                    target_dict[index] = ident
                    target_tokens.append(' '.join([token.text for token in
                                                   phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]",
                                                                namespaces=self.tei_ns)]))

                for index, phrase in enumerate(
                        main_file_tree.xpath(f"descendant::tei:div[@type='livre'][@n='{book_n}']/"
                                             f"descendant::tei:div[@type='partie'][@n='{part_n}']/"
                                             f"descendant::tei:div[@type='chapitre'][@n='{chapter_n}']/"
                                             f"descendant::tei:phr", namespaces=self.tei_ns)):
                    ident = utils.generateur_id(6)
                    phrase.set('{http://www.w3.org/XML/1998/namespace}id', ident)
                    source_dict[index] = ident
                    source_tokens.append(' '.join([token.text for token in
                                                   phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]",
                                                                namespaces=self.tei_ns)]))
                assert len(source_dict) == len(source_tokens), f'Error {len(source_dict)} {len(source_tokens)}'
                assert len(target_dict) == len(target_tokens), 'Error'
                aligner = Bertalign(source_tokens, target_tokens)
                aligner.align_sents()
                aligner.print_sents()
                tsource = []
                for tuple in aligner.result:
                    source, target = tuple
                    transformed_source = '#' + ' #'.join([source_dict[index] for index in source])
                    transformed_target = '#' + ' #'.join([target_dict[index] for index in target])
                    tsource.append((transformed_source, transformed_target))
                source_target_dict = {source: target for source, target in tsource}
                target_source_dict = {target: source for source, target in tsource}

                all_phrases = tree.xpath("descendant::tei:phr", namespaces=self.tei_ns)
                all_ids = tree.xpath("descendant::tei:phr/@xml:id", namespaces=self.tei_ns)
                ids_and_phrases = list(zip(all_ids, all_phrases))

                for index, (identifier, phrase) in enumerate(ids_and_phrases):
                    try:
                        match = [id for id in target_source_dict if identifier in id][0]
                        phrase.set('corresp', target_source_dict[match])
                    except IndexError:
                        phrase.set('corresp', 'None')

                with open(path.replace(".xml", ".final.xml"), "w") as output_target_file:
                    output_target_file.write(etree.tostring(tree, pretty_print=True).decode())

                all_phrases = main_file_tree.xpath(f"descendant::tei:div[@type='livre'][@n='{book_n}']/"
                                                   f"descendant::tei:div[@type='partie'][@n='{part_n}']/"
                                                   f"descendant::tei:div[@type='chapitre'][@n='{chapter_n}']/"
                                                   f"descendant::tei:phr", namespaces=self.tei_ns)
                all_ids = main_file_tree.xpath(f"descendant::tei:div[@type='livre'][@n='{book_n}']/"
                                               f"descendant::tei:div[@type='partie'][@n='{part_n}']/"
                                               f"descendant::tei:div[@type='chapitre'][@n='{chapter_n}']/"
                                               f"descendant::tei:phr/@xml:id", namespaces=self.tei_ns)
                ids_and_phrases = list(zip(all_ids, all_phrases))
                print(source_target_dict)
                for index, (identifier, phrase) in enumerate(ids_and_phrases):
                    try:
                        match = [id for id in source_target_dict if identifier in id][0]
                        phrase.set('corresp', source_target_dict[match])
                    except IndexError:
                        phrase.set('corresp', 'None')

                with open(main_file_path.replace(".xml", ".final.xml"), "w") as output_main_file:
                    output_main_file.write(etree.tostring(main_file_tree, pretty_print=True).decode())

    def inject_sents(self, results, source_zip, target_zip):
        """
        Avec cette fonction on récupère l'alignement sur le texte et on le réinjecte dans le fichier TEI
        """
        pass

    def alignement_de_structures(self):
        """
        On se sert de l'alignement sémantique pour aligner des structures sur un document cible à partir 
        d'un document source. Alignement puis identification de la borne supérieure de la structure (division, titre)
        On se servira d'un calcul de similarité pour identifier précisément la fin de la division dans le document cible
        """
        pass

class XMLAligner:
    
    def __init__(self, hierarchy:list[str] = ["tei:div[@type='livre']", "tei:div[@type='partie']", "tei:div[@type='chapitre']"],
                 id_attribute:str = "n", 
                 witnesses:list[str] = []):
        """
        :param: hierarchy: la hiérarchie sur laquelle boucler dans le document XML. Les documents source et cible doivent être
        déjà alignés au niveau de la hiérarchie structurelle donnée. Sous la forme d'une liste d'expressions xpath.
        Exemple: ['tei:div[@type='livre']', 'tei:div[@type='partie']', 'tei:div[@type='chapitre']']
        :param: id_attribute: l'attribut qui contient l'identifiant de la division minimale
        :param: witnesses: la liste des chemins vers chacun des témoins. Chaque témoin doit avoir son sigle encodée dans le xml:id
        de la racine du fichier
        """
        self.hierarchy = hierarchy
        self.id_attribute = id_attribute
        self.witnesses = witnesses
        self.global_text_dict = {}
        self.parsed_witnesses = {}
        self.parse_witnesses()
        self.align_divisions()
        self.ns_decl = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
    def parse_witnesses(self):
        for witness in self.witnesses:
            as_tree = etree.parse(witness)
            ID = as_tree.xpath("@xml:id")
            self.parsed_witnesses[ID] = as_tree
        
    
    def align_text(self):
        for division in self.align_divisions():
            pass
    
    def align_divisions(self):
        """
        Fonction qui permet de récupérer les divisions données d'un document en respectant une hiérarchie donnée.
        Elle nourrit le dictionnaire `global_text_dict` qui contient des listes de noeuds à aligner.
        """
        for witness in self.parsed_witnesses:
            path = "descendant::" + "/".join(hierarchy)
            # On itère sur chaque niveau hiérarchique. Un système récursif devrait fonctionner mieux.
            for minimal_division in witness.xpath(path):
                identifier = minimal_division.xpath(self.id_attribute, namespaces=self.ns_decl)[0]
                try:
                    self.global_text_dict[identifier].append(minimal_division)
                except KeyError:
                    self.global_text_dict[identifier] = [minimal_division]
    

if __name__ == '__main__':
    # TODO: intégrer les noeuds non w|pc pour ne pas perdre cette information.
    # TODO: transformer en dictionnaire en indiquant clairement qui est le témoin-source
    files = {"main_file": "/home/mgl/Bureau/Travail/Communications_et_articles/seminaire_caer/data/xml/Rome_W.xml",
             "target_files": [
                 "/home/mgl/Bureau/Travail/Communications_et_articles/seminaire_caer/data/xml/Val_S.citable.xml"]
             }

    Aligner = TEIAligner(files, tokenize=True)
    Aligner.alignementMultilingue()



def create_pairs(full_list: list, main_wit_index: int) -> list[tuple]:
    """
    From a list of witnesses and the main witness index, create all possible pairs with this witness. Returns a list 
    of tuples with the main wit and the wit to compare it to
    """
    pairs = []
    main_wit = full_list.pop(int(main_wit_index))
    for wit in full_list:
        pairs.append((main_wit, wit))
    return pairs


class Aligner:
    """
    La classe Aligner initialise le moteur d'alignement, fondé sur Bertalign
    """

    def __init__(self,
                 model,
                 corpus_limit: None,
                 max_align=3,
                 out_dir="out",
                 use_punctuation=True,
                 input_dir="in",
                 main_wit=None,
                 prefix=None,
                 device="cpu",
                 tokenizer="regexp",
                 tok_models=None
                 ):
        
        self.model = model
        self.alignment_dict = dict()
        self.text_dict = dict()
        self.files_path = glob.glob(f"{input_dir}/*/*.txt")
        self.device = device
        assert any([main_wit in path for path in
                    self.files_path]), "Main wit doesn't match witnesses paths, please check arguments. " \
                                       f"Main wit: {main_wit}, other wits: {self.files_path}"
        print(self.files_path)
        self.main_file_index = next(index for index, path in enumerate(self.files_path) if main_wit in path)
        self.corpus_limit = corpus_limit
        self.max_align = max_align
        self.out_dir = out_dir
        self.use_punctiation = use_punctuation
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.tok_models = tok_models
        self.wit_pairs = create_pairs(self.files_path, self.main_file_index)

        try:
            os.mkdir(f"result_dir")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"result_dir/{self.out_dir}/")
        except FileExistsError:
            pass

        # Let's check the paths are correct
        for file in self.files_path:
            assert os.path.isfile(file), f"Vérifier le chemin: {file}"

    def parallel_align(self):
        """
        This function procedes to the alignments two by two and then merges the alignments into a single alignement
        """
        pivot_text = self.wit_pairs[0][0]
        pivot_text_lang = pivot_text.split("/")[-2]
        if self.tokenizer is None:
            pass
        elif self.tokenizer == "regexp":
            first_tokenized_text = utils.clean_tokenized_content(
                syntactic_tokenization.syntactic_tokenization(input_file=pivot_text,
                                                              corpus_limit=self.corpus_limit,
                                                              use_punctuation=True,
                                                              lang=pivot_text_lang))
        else:
            first_tokenized_text = tokenize.tokenize_text(input_file=pivot_text,
                                                          corpus_limit=self.corpus_limit,
                                                          remove_punct=False,
                                                          tok_models=self.tok_models,
                                                          output_dir=self.out_dir,
                                                          device=self.device,
                                                          lang=pivot_text_lang)

        assert first_tokenized_text != [], "Erreur avec le texte tokénisé du témoin base"

        main_wit_name = self.wit_pairs[0][0].split("/")[-1].split(".")[0]
        utils.write_json(f"result_dir/{self.out_dir}/tokenized_{main_wit_name}.json", first_tokenized_text)
        utils.write_tokenized_text(f"result_dir/{self.out_dir}/tokenized_{main_wit_name}.txt", first_tokenized_text)

        # Let's loop and align each pair
        # We randomize the pairs. It can help resolving memory issue.
        random.shuffle(self.wit_pairs)
        for index, (main_wit, wit_to_compare) in enumerate(self.wit_pairs):
            main_wit_name = main_wit.split("/")[-1].split(".")[0]
            wit_to_compare_name = wit_to_compare.split("/")[-1].split(".")[0]
            current_wit_lang = wit_to_compare.split("/")[-2]
            print(len(first_tokenized_text))
            if self.tokenizer is None:
                pass
            elif self.tokenizer == "regexp":
                second_tokenized_text = utils.clean_tokenized_content(
                    syntactic_tokenization.syntactic_tokenization(input_file=wit_to_compare,
                                                                  corpus_limit=self.corpus_limit,
                                                                  use_punctuation=True,
                                                                  lang=current_wit_lang))
            else:
                second_tokenized_text = tokenize.tokenize_text(input_file=wit_to_compare,
                                                               corpus_limit=self.corpus_limit,
                                                               remove_punct=False,
                                                               tok_models=self.tok_models,
                                                               output_dir=self.out_dir,
                                                               device=self.device,
                                                               lang=current_wit_lang)
            assert second_tokenized_text != [], f"Erreur avec le texte tokénisé du témoin comparé {wit_to_compare_name}"
            utils.write_json(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.json",
                             second_tokenized_text)
            utils.write_tokenized_text(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.txt",
                                       second_tokenized_text)

            # This dict will be used to create the alignment table in csv format
            self.text_dict[0] = first_tokenized_text
            self.text_dict[index + 1] = second_tokenized_text

            # Let's align the texts
            print(f"Aligning {main_wit} with {wit_to_compare}")

            # Tests de profil et de paramètres
            profile = 0
            if profile == 0:
                margin = True
                len_penality = True
            else:
                margin = False
                len_penality = True
            aligner = Bertalign(self.model,
                                first_tokenized_text,
                                second_tokenized_text,
                                max_align=self.max_align,
                                win=5, skip=-.2,
                                margin=margin,
                                len_penalty=len_penality,
                                device=self.device)
            aligner.align_sents()

            # We append the result to the alignment dictionnary
            self.alignment_dict[index] = aligner.result
            utils.write_json(f"result_dir/{self.out_dir}/alignment_{str(index)}.json", aligner.result)
            utils.save_alignment_results(aligner.result, first_tokenized_text, second_tokenized_text,
                                         f"{main_wit_name}_{wit_to_compare_name}", self.out_dir)
        utils.write_json(f"result_dir/{self.out_dir}/alignment_dict.json", self.alignment_dict)

    def save_final_result(self, merged_alignments: list, delimiter="\t"):
        """
        Saves result to csv file
        """

        all_wits = [self.wit_pairs[0][0]] + [pair[1] for pair in self.wit_pairs]
        filenames = [wit.split("/")[-1].replace(".txt", "") for wit in all_wits]
        with open(f"result_dir/{self.out_dir}/final_result.csv", "w") as output_text:
            output_text.write(delimiter + delimiter.join(filenames) + "\n")
            # TODO: remplacer ça, c'est pas propre et ça sert à rien
            translation_table = {letter: index for index, letter in enumerate(string.ascii_lowercase)}
            for alignment_unit in merged_alignments:
                output_text.write("|".join(value for value in alignment_unit['a']) + delimiter)
                for index, witness in enumerate(merged_alignments[0]):
                    output_text.write("|".join(self.text_dict[translation_table[witness]][int(value)] for value in
                                               alignment_unit[witness]))
                    if index + 1 != len(merged_alignments[0]):
                        output_text.write(delimiter)
                output_text.write("\n")

        with open(f"result_dir/{self.out_dir}/readable.csv", "w") as output_text:
            output_text.write(delimiter.join(filenames) + "\n")
            # TODO: remplacer ça, c'est pas propre et ça sert à rien
            translation_table = {letter: index for index, letter in enumerate(string.ascii_lowercase)}
            for alignment_unit in merged_alignments:
                for index, witness in enumerate(merged_alignments[0]):
                    output_text.write(" ".join(self.text_dict[translation_table[witness]][int(value)] for value in
                                               alignment_unit[witness]))
                    if index + 1 != len(merged_alignments[0]):
                        output_text.write(delimiter)
                output_text.write("\n")

        with open(f"result_dir/{self.out_dir}/final_result_as_index.csv", "w") as output_text:
            output_text.write(delimiter + delimiter.join(filenames) + "\n")
            for alignment_unit in merged_alignments:
                for index, witness in enumerate(merged_alignments[0]):
                    output_text.write("|".join(value for value in
                                               alignment_unit[witness]))
                    if index + 1 != len(merged_alignments[0]):
                        output_text.write(delimiter)
                output_text.write("\n")

        data = pd.read_csv(f"result_dir/{self.out_dir}/final_result.csv", delimiter="\t")
        # Convert the DataFrame to an HTML table
        html_table = data.to_html()
        full_html_file = f"""<html>
                          <head>
                          <title>Alignement final</title>
                            <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                            </head>
                          <body>
                          {html_table}
                          </body>
                    </html>"""
        with open(f"result_dir/{self.out_dir}/final_result.html", "w") as output_html:
            output_html.write(full_html_file)


def run_alignments(out_dir, input_dir, main_wit, prefix, device, use_punctuation, tokenizer, tok_models,
                   corpus_limit=None):
    # TODO: augmenter la sensibilité à la différence sémantique pour apporter plus d'omissions dans le texte. La fin
    # Est beaucoup trop mal alignée, alors que ça irait bien avec + d'absence. Ça doit être possible vu que des omissions sont créés.

    # Initialize model 
    models = {0: "distiluse-base-multilingual-cased-v2", 1: "LaBSE", 2: "Sonar"}
    model = Encoder(models[int(1)], device=device)

    print(f"Punctuation for tokenization: {use_punctuation}")
    MyAligner = Aligner(model, corpus_limit=corpus_limit,
                        max_align=3,
                        out_dir=out_dir,
                        use_punctuation=use_punctuation,
                        input_dir=input_dir,
                        main_wit=main_wit,
                        prefix=prefix,
                        device=device,
                        tokenizer=tokenizer,
                        tok_models=tok_models)
    MyAligner.parallel_align()
    utils.write_json(f"result_dir/{out_dir}/alignment_dict.json", MyAligner.alignment_dict)
    align_dict = utils.read_json(f"result_dir/{out_dir}/alignment_dict.json")

    # Let's merge each alignment table into one and inject the omissions
    list_of_merged_alignments = graph_merge.merge_alignment_table(align_dict)

    # TODO: re-run the alignment on the units that are absent in the base wit.  

    # On teste si on ne perd pas de noeuds textuels
    print("Testing results consistency")
    possible_witnesses = string.ascii_lowercase[:len(align_dict) + 1]
    tested_table = utils.test_tables_consistency(list_of_merged_alignments, possible_witnesses)
    # TODO: une phase de test pour voir si l'alignement final est cohérent avec les alignements deux à deux

    # Let's save the final tables (indices and texts)
    MyAligner.save_final_result(merged_alignments=list_of_merged_alignments)

    return tested_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default=None,
                        help="Input directory where the .txt files are stored. Each linguistic version should be stored "
                             "in txt format in a single directory so that the file matches the expression `{input_dir}/*/*.txt`")
    parser.add_argument("-o", "--out_dir", default="out",
                        help="Path to output dir.")
    parser.add_argument("-punct", "--use_punctuation", default=True,
                        help="Use punctuation to tokenize texts (default: True).")
    parser.add_argument("-mw", "--main_wit",
                        help="Path to pivot witness.")
    parser.add_argument("-p", "--prefix", default=None,
                        help="Prefix for produced files (to be implemented).")
    parser.add_argument("-d", "--device", default='cpu',
                        help="Device to be used (default: cpu).")
    parser.add_argument("-t", "--tokenizer", default='regexp',
                        help="Tokenizer to be used (None, regexp, bert-based)")
    parser.add_argument("-l", "--corpus_limit", default=None,
                        help="Limit alignment to given proportion of each text (float)")

    args = parser.parse_args()
    out_dir = args.out_dir
    input_dir = args.input_dir
    main_wit = args.main_wit
    assert input_dir != None, "Input dir is mandatory"
    assert main_wit != None, "Main wit path is mandatory"
    prefix = args.prefix
    device = args.device
    corpus_limit = args.corpus_limit
    if corpus_limit:
        corpus_limit = float(corpus_limit)
    tokenizer = args.tokenizer
    tok_models = {"fr":
                      {"model": "models/fr",
                       "tokenizer": "dbmdz/bert-base-french-europeana-cased",
                       "tokens_per_example": 12},
                  "es": {"model": "models/es",
                         "tokenizer": "dccuchile/bert-base-spanish-wwm-cased",
                         "tokens_per_example": 30},
                  "it": {"model": "models/it",
                         "tokenizer": "dbmdz/bert-base-italian-xxl-cased",
                         "tokens_per_example": 12},
                  "la": {"model": "ProMeText/aquilign_segmenter_latin",
                         "tokenizer": "LuisAVasquez/simple-latin-bert-uncased",
                         "tokens_per_example": 50}}
    assert tokenizer in ["None", "regexp",
                         "bert-based"], "Authorized values for tokenizer are: None, regexp, bert-based"
    if tokenizer == "None":
        tokenizer = None
    use_punctuation = args.use_punctuation
    run_alignments(out_dir, input_dir, main_wit, prefix, device, use_punctuation, tokenizer, tok_models,
                   corpus_limit)


