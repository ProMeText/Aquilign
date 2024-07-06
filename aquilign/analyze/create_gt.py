import tqdm
import itertools
import aquilign.align.utils as utils
from aquilign.align.aligner import Bertalign
from aquilign.align.encoder import Encoder
import sys


def create_list(path_to_csv, delimiter="\t", wits_example=7):
    """
    This function computes the similarities of each alignment unit from a single csv file
    """
    with open(path_to_csv, "r") as input_csv:
        csv_file = input_csv.read()
    csv_list = csv_file.split("\n")
    csv_list = [row.split(delimiter) for row in csv_list]
    labels = {idx: value for idx, value in enumerate(csv_list[0][2:2+wits_example])}
    alignment_dict = {index: alignment[2:2+wits_example] for index, alignment in enumerate(csv_list[1:])}
    gt_dict = {index: alignment[2+wits_example:] for index, alignment in enumerate(csv_list[1:])}
    merged_dict = {}
    for index, _ in enumerate(csv_list[1:]):
        merged_dict[index] = [alignment_dict[index], gt_dict[index]]
    print(merged_dict)
    return merged_dict, csv_list[0][2:wits_example + 2], labels


def compute(text_a, text_b, model, device):
    aligner = Bertalign(model=model, src=[text_a[1]], tgt=[text_b[1]], max_align=3, device=device)
    output = aligner.compute_distance()
    print(text_a)
    print(text_b)
    print(output)
    if output > .6:
        print("Texts seem similar")
    else:
        print("Texts seem disimilar")
    return output


def compute_similarity(alignments: list, model, device, labels):
    combinaisons = list(set(itertools.combinations(alignments, 2)))
    results = []
    for combinaison in combinaisons:
        current_dict = {}
        cosine_sim = compute(combinaison[0], combinaison[1], model, device)
        current_dict['WitA'] = labels[combinaison[0][0]]
        current_dict['WitB'] = labels[combinaison[1][0]]
        current_dict['Dist'] = cosine_sim.item()
        current_dict['TextA'] = combinaison[0][1]
        current_dict['TextB'] = combinaison[1][1]
        results.append(current_dict)
    return results


def main(path, wit_number):
    device = "cuda:0"
    models = {0: "distiluse-base-multilingual-cased-v2", 1: "LaBSE", 2: "Sonar"}
    model = Encoder(models[int(1)], device="cuda:0")
    alignment_dict, wits, labels = create_list(path, wits_example=wit_number)
    alignments_as_similarities = {}
    similarities_only = []
    for index, alignments in tqdm.tqdm(alignment_dict.items()):
        sents, gts = alignments
        print("\nNew alignment unit")
        non_empty_entries = len([(index, element) for index, element in enumerate(sents) if element != ""])
        if non_empty_entries != 1:
            results = compute_similarity(
                alignments=[(index, element.replace("|", " ")) for index, element in enumerate(sents) if element != ""], 
                model=model, 
                device=device, 
            labels=labels)
        else:
            results = []
        alignments_as_similarities[index + 1] = (results, list(zip(wits, sents, gts)))
        similarities_only.append(results)
    print(alignments_as_similarities)
    result_dir = '/'.join(path.split("/")[:-1])
    utils.write_json(f"{result_dir}/similarities_as_list_{path.split('/')[-1].replace('.csv', '')}.json", alignments_as_similarities)
    utils.write_json(f"{result_dir}/similarities_{path.split('/')[-1].replace('.csv', '')}.json", similarities_only)


if __name__ == '__main__':
    input_file = sys.argv[1]
    nombre_temoins = int(sys.argv[2])
    main(input_file, nombre_temoins)