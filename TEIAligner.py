from bertalign import Bertalign
import lxml.etree as etree
import bertalign.tokenization as tokenization
import bertalign.utils as utils
import json
import traceback


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
        self.target_parsed_files = {}
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


if __name__ == '__main__':
    # TODO: intégrer les noeuds non w|pc pour ne pas perdre cette information.
    # TODO: transformer en dictionnaire en indiquant clairement qui est le témoin-source
    files = {"main_file": "/home/mgl/Bureau/Travail/Communications_et_articles/seminaire_caer/data/xml/Rome_W.xml",
             "target_files": [
                 "/home/mgl/Bureau/Travail/Communications_et_articles/seminaire_caer/data/xml/Val_S.citable.xml"]
             }

    Aligner = TEIAligner(files, tokenize=True)
    Aligner.alignementMultilingue()