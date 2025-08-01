# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys

## script for the evaluation of the alignment
## calculate the acc of the correct aligned units
## usage : python3 eval_alignment.py data.csv gt.csv ouput.csv
## where:
# data.csv is the data automaticaly produced by the alignment process
# gt.csv is the file with the correct data
## both files should be csv files that corresponds the one to the other : each column should have the same name for the same witness, and be separeted by a comma
# output.csv is the file where the results (acc for each text and average) will be saved

def compute_acc_align(data,gt):
    list_of_acc = []
    list_of_acc_tok = []
    list_of_acc_maison = []
    list_of_acc_maison_tok = []
    list_of_aer = []
    list_of_aer_tok = []

    for i in range(len(data.columns)):
        # get the name of each column
        nom = data.iloc[:, i].name
        print(nom)

        # calculate accuracy with pb of tok
        dataALT = list(data[nom])
        print(dataALT)
        gtALT = list(gt[nom])
        print(gtALT)

        accTok = accuracy_score(dataALT, gtALT)
        list_of_acc_tok.append(accTok)

        f1_Tok = f1_score(dataALT, gtALT, average='macro')
        print(f'f1 : {f1_Tok}')
        precision_Tok = precision_score(dataALT, gtALT, average='micro', zero_division=0.0)
        print(f'precision : {precision_Tok}')
        recall_Tok = recall_score(dataALT, gtALT, average='macro', zero_division=0.0)
        print(f'recall : {recall_Tok}')

        # list
        val_liens_ok_tok = []
        # AER
        for i in range(len(dataALT)):
            if dataALT[i] == gtALT[i]:
                val = 1
            else:
                val = 0
            val_liens_ok_tok.append(val)
        sum_ok_tok = sum(val_liens_ok_tok)
        print(f'len list {len(val_liens_ok_tok)}')
        print(f'len dataALT {len(dataALT)}')

        # somme des éléments corrects sur le nombre d'individus
        acc_maison_tok = sum_ok_tok/len(dataALT)
        list_of_acc_maison_tok.append(acc_maison_tok)
        print(f'acc_maison_tok: {acc_maison_tok}')


        aer_tok = 1 - ((2 * sum_ok_tok) / (len(gtALT) + len(dataALT)))
        print(f'aer_tok : {aer_tok}')
        list_of_aer_tok.append(aer_tok)


        ###

        # remove the cells which are concerned by the problems of tok and calculate the acc
        index_to_remove = gt[gt[nom].str.contains(".5", regex=False, na=False)].index
        print(index_to_remove)
        gtSelectok = gt.drop(index=index_to_remove)
        dataSelectok = data.drop(index=index_to_remove)
        dataAL = list(dataSelectok[nom])
        gtAL = list(gtSelectok[nom])
        print(dataAL)

        acc = accuracy_score(gtAL, dataAL)
        list_of_acc.append(acc)

        ### AER
        val_liens_ok = []
        for i in range(len(dataAL)):
            if dataAL[i] == gtAL[i]:
                val = 1
            else:
                val = 0
            val_liens_ok.append(val)

        sum_ok = sum(val_liens_ok)
        print(val_liens_ok)
        print(f'len list {len(val_liens_ok)}')

        aer = 1 - ((2 * sum_ok) / (len(gtAL) + len(dataAL)))
        print(f'aer : {aer}')
        list_of_aer.append(aer)

        # sum of correct elements by the number of ind
        acc_maison = sum_ok / len(dataAL)
        list_of_acc_maison.append(acc_maison)
        print(f'acc_maison: {acc_maison}')

        new_row = {'witness' : nom, 'acc-bad-tok' : accTok, 'acc' : acc , 'acc-maison-tok' : acc_maison_tok, 'acc-maison' : acc_maison, 'aer-bad-tok' : aer_tok, 'aer' : aer}
        df_values_eval_align.loc[len(df_values_eval_align)] = new_row




    acc_average = sum(list_of_acc) / len(list_of_acc)
    acc_tok_average = sum(list_of_acc_tok) / len(list_of_acc_tok)
    acc_maison_average = sum(list_of_acc_maison) / len(list_of_acc_maison)
    acc_maison_tok_average = sum(list_of_acc_maison_tok) / len(list_of_acc_maison_tok)
    aer_average = sum(list_of_aer) / len(list_of_aer)
    aer_tok_average = sum(list_of_aer_tok) / len(list_of_aer_tok)
    average_results = {'witness' : 'average', 'acc-bad-tok' : acc_tok_average, 'acc' : acc_average, 'acc-maison-tok' : acc_maison_tok_average, 'acc-maison' : acc_maison_average, 'aer-bad-tok' : aer_tok_average, 'aer' : aer_average}

    df_values_eval_align.loc[len(df_values_eval_align)] = average_results

    print(df_values_eval_align)
    return df_values_eval_align


if __name__ == '__main__':
    data_file = sys.argv[1]
    gt_file = sys.argv[2]
    output_file = sys.argv[3]

    data = pd.read_csv(data_file, sep=",")
    gt = pd.read_csv(gt_file, sep=",")

    df_values_eval_align = pd.DataFrame(columns=['witness', 'acc-bad-tok', 'acc', 'acc-maison-tok', 'acc-maison', 'aer-bad-tok', 'aer'])

    final_df = compute_acc_align(data,gt)


    final_df.to_csv(output_file, index=False)


