from sklearn import preprocessing
import numpy as np
import aquilign.analyze.utils as utils
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import sys
import json


def write_to_log(string):
    print(string)
    with open(".logs/log_clusters.txt", "a") as log_file:
        if type(string) != str:
            if type(string) in [list, dict]:
                json.dump(string, log_file)
            else:
                log_file.write(str(string))
        else:
            log_file.write(string)
        log_file.write("\n")


def create_clusters(path):
    """
    This function converts the distance matrix into a cluster matrix, using multiple methods.
    It take a json file containing all the distance pairs, turns it into a matrix, and then performs the
    clusterisation
    """
    with open(".logs/log_clusters.txt", "w") as input_log:
        try:
            input_log.truncate(0)
        except FileNotFoundError:
            pass
    write_to_log(path)
    np.printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
    alignements = utils.read_json(path)
    write_to_log(len(alignements))
    for index, alignement in enumerate(alignements):
        write_to_log("---\nNew unit")
        if alignement in [{}, [{}, {}], []]:
            continue
        text, al = alignement
        similarities = [(wits, similarities) for wits, similarities in al.items()]
        # https://stackoverflow.com/a/16193637
        similarities.sort(key=lambda x: (int(x[0].split("-")[0]), int(x[0].split("-")[1])))
        out_list = []
        range_of_elements_a = list(set([element[0].split("-")[0] for element in similarities]))
        range_of_elements_b = list(set([element[0].split("-")[1] for element in similarities]))
        full_list = list(set(range_of_elements_a + range_of_elements_b))
        full_list.sort()
        for i in full_list:
            interm_list = []
            for j in full_list:
                if i == j:
                    interm_list.append(1)
                else:
                    interm_list.append([elem[1] for elem in similarities if elem[0] == f"{i}-{j}" or elem[0] == f"{j}-{i}"][0])
            out_list.append(interm_list)
        similarities_as_array = np.asarray(out_list)
        clusters_1 = AgglomerativeClustering(n_clusters=None, distance_threshold=.7, linkage='ward').fit(similarities_as_array)
        clusters_2 = DBSCAN(min_samples=2).fit(similarities_as_array)
        write_to_log(clusters_1.labels_)
        classes = list(set(clusters_1.labels_.tolist()))
        clusters_as_dict = {int(index):int(cl) for index, cl in enumerate(clusters_1.labels_)}
        write_to_log(clusters_as_dict)
        
        out_dict = {}
        for index, label in clusters_as_dict.items():
            try:
                out_dict[label].append(index)
            except KeyError:
                out_dict[label] = [index]
        write_to_log(out_dict)
        
        for label in classes:
            write_to_log(f"Class {label}:")
            write_to_log("\n".join(text[str(item)] for item in out_dict[label]))
        # for in_class in classes:
            
        
        
        write_to_log(clusters_1.labels_)
        write_to_log(clusters_2.labels_)

if __name__ == '__main__':
    create_clusters(sys.argv[1])