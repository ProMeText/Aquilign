import os
import string
import random
import aquilign.align.graph_merge as graph_merge
import aquilign.align.utils as utils
import aquilign.preproc.tok_apply as bert_tokenize
import aquilign.preproc.regex_tokenization as regex_tokenization
from aquilign.align.encoder import Encoder
from aquilign.align.aligner import Bertalign, Bertalign_Embbed
import pandas as pd
import argparse
import glob


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
                 tok_models=None,
                 multilingual=True
                 ):
        self.model = model
        self.alignment_dict = dict()
        self.text_dict = dict()
        self.files_path = glob.glob(f"{input_dir}/*/*.txt")
        self.device = device
        self.multilingual_segmentation_model = multilingual
        assert any([main_wit in path for path in
                    self.files_path]), "Main wit doesn't match witnesses paths, please check arguments. " \
                                       f"Main wit: {main_wit}, other wits: {self.files_path}"
        print(self.files_path)
        self.main_file_index = next(index for index, path in enumerate(self.files_path) if main_wit in path)
        self.corpus_limit = corpus_limit
        self.max_align = max_align
        self.out_dir = out_dir
        self.use_punctuation = use_punctuation
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
        if self.multilingual_segmentation_model and self.tokenizer == "bert-based":
            pivot_text_lang = "ml"
        else:
            pivot_text_lang = pivot_text.split("/")[-2]
        if not self.tokenizer:
            print("Code not implemented for input texts as lists. Exiting")
            exit(0)
        elif self.tokenizer == "regexp":
            first_tokenized_text = utils.clean_tokenized_content(
                regex_tokenization.regex_tokenization(input_file=pivot_text,
                                                      corpus_limit=self.corpus_limit,
                                                      use_punctuation=True,
                                                      lang=pivot_text_lang))
        else:
            first_tokenized_text = bert_tokenize.tokenize_text(input_file=pivot_text,
                                                               corpus_limit=self.corpus_limit,
                                                               remove_punct=False,
                                                               tok_models=self.tok_models,
                                                               output_dir=self.out_dir,
                                                               device=self.device,
                                                               lang=pivot_text_lang)

        sentence_embeddings = Bertalign_Embbed(model=self.model,
                                               max_align=self.max_align,
                                               sents=first_tokenized_text)
        pivot_vecs, pivot_lens, pivot_search_simple_vecs = sentence_embeddings.return_embbeds()

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
            if self.multilingual_segmentation_model:
                current_wit_lang = "ml"
            else:
                current_wit_lang = wit_to_compare.split("/")[-2]
            print(len(first_tokenized_text))
            if self.tokenizer is None:
                pass
            elif self.tokenizer == "regexp":
                second_tokenized_text = utils.clean_tokenized_content(
                    regex_tokenization.regex_tokenization(input_file=wit_to_compare,
                                                          corpus_limit=self.corpus_limit,
                                                          use_punctuation=True,
                                                          lang=current_wit_lang))
            else:
                second_tokenized_text = bert_tokenize.tokenize_text(input_file=wit_to_compare,
                                                                    corpus_limit=self.corpus_limit,
                                                                    remove_punct=False,
                                                                    tok_models=self.tok_models,
                                                                    output_dir=self.out_dir,
                                                                    device=self.device,
                                                                    lang=current_wit_lang)
            assert second_tokenized_text != [], f"Erreur avec le texte tokénisé du témoin comparé {wit_to_compare_name}"
            utils.write_json(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.json", second_tokenized_text)
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
            aligner = Bertalign(model=self.model,
                                src_sents=first_tokenized_text,
                                src_lens=pivot_lens,
                                src_vecs=pivot_vecs,
                                search_simple_vecs=pivot_search_simple_vecs,
                                tgt=second_tokenized_text,
                                max_align=self.max_align,
                                win=5,
                                skip=-.2,
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


def run_alignments(out_dir, input_dir, main_wit, prefix, device, use_punctuation, tokenizer, tok_models, multilingual,
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
                        tok_models=tok_models,
                        multilingual=multilingual)
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
    parser.add_argument("-ml", "--multilingual", default=True,
                        help="Use multilingual segmentation model.")
    parser.add_argument("-mw", "--main_wit",
                        help="Path to pivot witness.")
    parser.add_argument("-p", "--prefix", default=None,
                        help="Prefix for produced files (to be implemented).")
    parser.add_argument("-d", "--device", default='cpu',
                        help="Device to be used (default: cpu).")
    parser.add_argument("-t", "--tokenizer", default='regexp', help="Tokenizer to be used (None, regexp, bert-based)")
    parser.add_argument("-l", "--corpus_limit", default=None,
                        help="Limit alignment to given proportion of each text (float)")

    args = parser.parse_args()
    out_dir = args.out_dir
    multilingual = args.multilingual
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
                         "tokens_per_example": 50},
                  "ml": {"model": "ProMeText/aquilign-multilingual-segmenter",
                         "tokenizer": "google-bert/bert-base-multilingual-cased",
                         "tokens_per_example": 100}
                  }
    assert tokenizer in ["None", "regexp",
                         "bert-based"], "Authorized values for tokenizer are: None, regexp, bert-based"
    if tokenizer == "None":
        tokenizer = None
    use_punctuation = args.use_punctuation
    if device != "cpu":
        print("The segmentation will be performed on the GPU, as for the sentence embeddings, but "
              "alignment will be realized on the CPU for code maintenance reasons.")
    run_alignments(out_dir, input_dir, main_wit, prefix, device, use_punctuation, tokenizer, tok_models, multilingual,
                   corpus_limit)
