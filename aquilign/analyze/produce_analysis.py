import tqdm
import itertools
import aquilign.analyze.utils as utils
import aquilign.align.aligner as align
import aquilign.align.encoder as encoder
import sys 


def create_list(path_to_csv, delimiter):
    """
    This function computes the similarities of each alignment unit from a single csv file
    """
    with open(path_to_csv, "r") as input_csv:
        csv_file = input_csv.read()
    csv_list = csv_file.split("\n")
    csv_list = [row.split(delimiter) for row in csv_list]
    cleaned_list = []
    for unit in csv_list:
        interm_list = [utils.remove_punctuation(item) for item in unit[1:]]
        cleaned_list.append(interm_list)
    alignment_dict = {index: alignement for index, alignement in enumerate(cleaned_list[1:])}
    return alignment_dict

def batch_compute(batch_a, batch_b):
    pass

def compute(text_a, text_b, model):
    aligner = align.Bertalign(model=model, src=[text_a[1]], tgt=[text_b[1]], max_align=3, device=device)
    output = aligner.compute_distance()
    print(text_a)
    print(text_b)
    print(output)
    if output > .6:
        print("Texts seem similar")
    else:
        print("Texts seem disimilar")
    return output
        

def compute_similarity(alignments:list, model):
    combinaisons = list(set(itertools.combinations(alignments, 2)))
    current_dict = {}
    for combinaison in combinaisons:
        print("---")
        print("New pair.")
        cosine_sim = compute(combinaison[0], combinaison[1], model)
        current_dict[f"{combinaison[0][0]}-{combinaison[1][0]}"] = round(cosine_sim.item(), 3)
    return current_dict
    
    
    
def main(path, delimiter, model):
    alignment_dict = create_list(path, delimiter)
    alignments_as_similarities = []
    for index, alignments in tqdm.tqdm(alignment_dict.items()):
        print("\nNew alignment unit")
        non_empty_entries = len([(index, element) for index, element in enumerate(alignments) if element != ""])
        if non_empty_entries != 1:
            print(alignments)
            alignments_as_similarities.append(({index: al for index, al in enumerate(alignments)}, compute_similarity([(index, element.replace("|", " ")) for index, element in enumerate(alignments) if element != ""], model)))
        else:
            alignments_as_similarities.append([])
    print(alignments_as_similarities)
    result_dir = '/'.join(path.split("/")[:-1])
    print(result_dir)
    utils.write_json(f"{result_dir}/similarities_as_list.json", alignments_as_similarities)
        
        
    
    
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    device = "cuda:0"
    models = {0: "distiluse-base-multilingual-cased-v2", 1: "LaBSE", 2: "Sonar"}
    model = encoder.Encoder(models[int(1)], device=device)
    delimiter = "\t"
    main(input_file, delimiter, model)