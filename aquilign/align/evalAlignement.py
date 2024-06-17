# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import accuracy_score
import sys

## script for the evaluation of the alignment
## calculate the acc of the correct aligned units
## usage : python3 evalAlignment.py data.csv gt.csv ouput.txt
## where:
# data.csv is the data automaticaly produced by the alignment process
# gt.csv is the file with the correct data
## both files should be csv files that corresponds the one to the other : each column should have the same name for the same witness, and be separeted by a comma
# output.txt is the file where the results (acc for each text and average) will be saved

def compute_acc_align(data,gt):
    list_of_acc = []
    list_of_acc_tok = []
    global_results = {}
    global_results_tok = {}

    for i in range(len(data.columns)):
        # get the name of each column
        nom = data.iloc[:, i].name
        print(nom)

        # calculate accuracy with pb of tok
        dataALT = list(data[nom])
        gtALT = list(gt[nom])

        accTok = accuracy_score(dataALT, gtALT)
        list_of_acc_tok.append(accTok)
        local_results_tok = {nom : accTok}
        global_results_tok.update(local_results_tok)

        # remove the cells which are concerned by the problems of tok and calculate the acc
        index_to_remove = gt[gt[nom].str.contains(".5", regex=False, na=False)].index
        print(index_to_remove)
        gtSelectok = gt.drop(index=index_to_remove)
        dataSelectok = data.drop(index=index_to_remove)
        dataAL = list(dataSelectok[nom])
        gtAL = list(gtSelectok[nom])

        acc = accuracy_score(gtAL, dataAL)
        list_of_acc.append(acc)
        local_results = {nom: acc}
        global_results.update(local_results)

    acc_average = sum(list_of_acc) / len(list_of_acc)
    acc_tok_average = sum(list_of_acc_tok) / len(list_of_acc_tok)
    average_results = {'average' : acc_average}
    average_results_tok = {'average' : acc_tok_average}
    global_results.update(average_results)
    global_results_tok.update(average_results_tok)

    gr = f'Global results: {global_results}'
    grt = f'Global results with bad tokenization: {global_results_tok}'
    print(gr)
    print(grt)

    to_write = []
    to_write.append(gr)
    to_write.append(grt)

    return to_write


if __name__ == '__main__':
    data_file = sys.argv[1]
    gt_file = sys.argv[2]
    output_file = sys.argv[3]

    data = pd.read_csv(data_file, sep=",")
    gt = pd.read_csv(gt_file, sep=",")

    to_write = compute_acc_align(data,gt)

    with open(output_file, "w") as text_file:
        text_file.write(str(to_write))

