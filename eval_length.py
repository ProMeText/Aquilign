# -*- coding: utf-8 -*-
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import join, isfile
import os

## script for the evaluation of the lengths of segments and units (segment = final unit in the alignment)
## usage python eval_length dir
# where dir contains all the csv file produced by regex ant bert tokenisation alignments (text in the csv, corresponding to 'final_result.csv' files)

# flatten function
def flatten(xss):
    return [x for xs in xss for x in xs]

# def segment length
def segment_length(data, name):
    data = data.iloc[:, 1:]
    list_of_lengths = []

    # loop on the dfs and count the length of the text
    for i in range(len(data.columns)):
        for j in range(len(data)):
            text = data.iloc[j, i]
            if type(text) == float:
                pass
            else:
                splitted_text = re.findall('\w+', text)
                length = len(splitted_text)
                list_of_lengths.append(length)

    # compute average and median of the lengths
    average = np.average(list_of_lengths)
    median = np.median(list_of_lengths)

    new_row = {'base': name, 'average': average, 'median': median}

    # produce plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(list_of_lengths, bins=30, edgecolor="black", color="#69b3a2", alpha=0.3)
    plt.title(f'Number of tokens per segment ({name})')
    plt.xlabel('number of tokens')
    plt.ylabel('number of segments')
    ax.axvline(median, color="black", ls="--", label="Median")
    fig.savefig(f'results_eval_length/{name}-segment.png')

    # get values for each text
    values, counts = np.unique(list_of_lengths, return_counts=True)
    df_values = pd.DataFrame(
        {'tokens per segment': values,
         'number of segments concerned': counts})
    df_values.to_csv(f'results_eval_length/results_tokens_per_segment_{name}.csv', index=False)

    #return new_row, values, counts
    return new_row, values, counts, list_of_lengths

# function which does the same as the previous one but for the units
def unit_length(data, name):
    data = data.iloc[:, 1:]
    list_of_lengths_units = []
    for i in range(len(data.columns)):
        for j in range(len(data)):
            text = data.iloc[j, i]
            if type(text) == float:
                pass
            else:
                # unit : count the length of a text between '|'
                if '|' in text:
                    frag = text.split('|')
                    for k in range(len(frag)):
                        splitted_text = re.findall('\w+', frag[k])
                        #length = len(frag[k].split())
                        length = len(splitted_text)
                        list_of_lengths_units.append(length)
                else:
                    splitted_text = re.findall('\w+', text)
                    #length = len(text.split())
                    length = len(splitted_text)
                    list_of_lengths_units.append(length)

    average = np.average(list_of_lengths_units)
    median = np.median(list_of_lengths_units)

    new_row = {'base': name, 'average': average, 'median': median}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(list_of_lengths_units, bins=30, edgecolor="black", color="#69b3a2", alpha=0.3)
    plt.title(f'Number of tokens per unit ({name})')
    plt.xlabel('number of tokens')
    plt.ylabel('number of units')
    ax.axvline(median, color="black", ls="--", label="Median")
    fig.savefig(f'results_eval_length/{name}-units.png')

    values, counts = np.unique(list_of_lengths_units, return_counts=True)
    df_values = pd.DataFrame(
        {'tokens per unit': values,
         'number of units concerned': counts})
    df_values.to_csv(f'results_eval_length/results_tokens_per_unit_{name}.csv', index=False)
    return new_row, values, counts, list_of_lengths_units


if __name__ == '__main__':
    dir = sys.argv[1]
    directory = os.fsencode(dir)

    try:
        os.mkdir('results_eval_length')
    except OSError as exception:
        pass

    # create the needed dfs
    df_segment_lengths = pd.DataFrame(columns=['base', 'average', 'median'])
    df_unit_lengths = pd.DataFrame(columns=['base', 'average', 'median'])
    df_values_s_bert = pd.DataFrame(columns=['tokens per segment', 'number of segments concerned'])
    df_values_u_bert = pd.DataFrame(columns=['tokens per unit', 'number of units concerned'])
    df_values_s_regex = pd.DataFrame(columns=['tokens per segment', 'number of segments concerned'])
    df_values_u_regex = pd.DataFrame(columns=['tokens per unit', 'number of units concerned'])

    global_list_lengths_s_bert = []
    global_list_lengths_s_regex = []
    global_list_lengths_u_bert = []
    global_list_lengths_u_regex = []

    # loop on each file in the directory
    for filename in listdir(directory):
        full_path = join(directory, filename)
        if isfile(full_path):
            full_name = full_path.decode("utf-8")
            name = filename.decode("utf-8").split('.')[0].split('-', 1)[1]
            print(name)
            if name.endswith('bert'):
                data = pd.read_csv(full_name, sep='\t')
            else:
                data = pd.read_csv(full_name, sep=',')

            ### segments
            new_row_sl, values_s, counts_s, lengths_s = segment_length(data, name)

            df_segment_lengths.loc[len(df_segment_lengths)] = new_row_sl

            # produce two dfs : one for bert results and one for regex results
            if name.endswith('bert'):
                df_values_s_bert = df_values_s_bert._append(pd.DataFrame({'tokens per segment': values_s,
                                                                          'number of segments concerned': counts_s}))
                global_list_lengths_s_bert.append(lengths_s)

            else:
                df_values_s_regex = df_values_s_regex._append(pd.DataFrame({'tokens per segment': values_s,
                                                                            'number of segments concerned': counts_s}))
                global_list_lengths_s_regex.append(lengths_s)

            ### units
            new_row_ul, values_u, counts_u, lengths_u = unit_length(data, name)

            df_unit_lengths.loc[len(df_unit_lengths)] = new_row_ul


            if name.endswith('bert'):
                df_values_u_bert = df_values_u_bert._append(pd.DataFrame({'tokens per unit': values_u,
                                                                          'number of units concerned': counts_u}))
                global_list_lengths_u_bert.append(lengths_u)
            else:
                df_values_u_regex = df_values_u_regex._append(pd.DataFrame({'tokens per unit': values_u,
                                                                            'number of units concerned': counts_u}))
                global_list_lengths_u_regex.append(lengths_u)

    # groupby to have for bert and regex and for segment and unit single values for each number of token
    dfsb = df_values_s_bert.groupby('tokens per segment')['number of segments concerned'].sum().reset_index()
    dfsr = df_values_s_regex.groupby('tokens per segment')['number of segments concerned'].sum().reset_index()
    dfub = df_values_u_bert.groupby('tokens per unit')['number of units concerned'].sum().reset_index()
    dfur = df_values_u_regex.groupby('tokens per unit')['number of units concerned'].sum().reset_index()

    df_segment_lengths.sort_values(by=['base']).to_csv('results_eval_length/results_lengths_seg_full.csv', index=False)

    df_unit_lengths.sort_values(by=['base']).to_csv('results_eval_length/results_lengths_unit_full.csv', index=False)

    # save the results
    dfsb.to_csv('results_eval_length/results_tokens_per_segment_bert_global.csv', index=False)
    dfsr.to_csv('results_eval_length/results_tokens_per_segment_regex_global.csv', index=False)
    dfub.to_csv('results_eval_length/results_tokens_per_unit_bert_global.csv', index=False)
    dfur.to_csv('results_eval_length/results_tokens_per_unit_regex_global.csv', index=False)


    average_sb = np.average(flatten(global_list_lengths_s_bert))
    average_sr = np.average(flatten(global_list_lengths_s_regex))
    average_ub = np.average(flatten(global_list_lengths_u_bert,))
    average_ur = np.average(flatten(global_list_lengths_u_regex))

    print(f'averages {average_sb, average_sr, average_ub, average_ur}')

    print(f"index à 0 {dfsb[dfsb['tokens per segment'] == 0].index}" )
    print(f"index à 0 {dfub[dfub['tokens per unit'] == 0].index}")

    dfsb.drop(dfsb[dfsb['tokens per segment'] == 0].index, inplace=True)
    dfsr.drop(dfsr[dfsr['tokens per segment'] == 0].index, inplace=True)
    dfub.drop(dfub[dfub['tokens per unit'] == 0].index, inplace=True)
    dfur.drop(dfur[dfur['tokens per unit'] == 0].index, inplace=True)


    def categorize(value):
        if value < 50:
            return value
        elif value >= 50 and value < 100:
            return '50-99'
        elif value >= 100 and value < 150:
            return '100-149'
        elif value >= 150 and value <= 200:
            return '150-200'
        else:
            return '>200'


    dfsb['nbTokVal'] = dfsb['tokens per segment'].apply(categorize)

    ##chatgpt
    g_dfsb = dfsb.groupby('nbTokVal')['number of segments concerned'].sum().reset_index()

    # Sort the dataframe by the grouped tokens
    sort_order = list(range(1, 51)) + ['50-99', '100-149', '150-200', '>200']
    g_dfsb['nbTokVal'] = pd.Categorical(g_dfsb ['nbTokVal'], categories=sort_order, ordered=True)
    g_dfsb =  g_dfsb.sort_values('nbTokVal')
    g_dfsb.set_index('nbTokVal', inplace=True)


    dfsr['nbTokVal'] = dfsr['tokens per segment'].apply(categorize)

    g_dfsr = dfsr.groupby('nbTokVal')['number of segments concerned'].sum().reset_index()

    g_dfsr['nbTokVal'] = pd.Categorical(g_dfsr['nbTokVal'], categories=sort_order, ordered=True)
    g_dfsr = g_dfsr.sort_values('nbTokVal')
    g_dfsr.set_index('nbTokVal', inplace=True)


    fig, axs = plt.subplots(figsize=(12, 8))

    width = 0.3

    g_dfsb.plot.bar(ax=axs, width=width, position=1, legend=False, color='tab:blue', label='bert segmentation')
    g_dfsr.plot.bar(ax=axs, width=width, position=0, legend=False, color='orange', label='regex segmentation')
    axs.legend(['bert segmentation', 'regex segmentation'])
    axs.set_ylabel("number of segments concerned")
    axs.set_xlabel('tokens per segment')
    axs.set_title('Number of tokens per segment')

    axs.figure.savefig('results_eval_length/number_of_segments_bar_global_new.png')

    #axu = dfub.plot(x='tokens per unit', y='number of units concerned', figsize=(8, 8),
                  #  title='Number of tokens per unit')
    #dfur.plot(ax=axu, x='tokens per unit', y='number of units concerned')
    axu = dfub.plot.bar(x='tokens per unit', y='number of units concerned', figsize=(8, 8),
                        title='Number of tokens per unit')
    dfur.plot.bar(ax=axu, color='orange')
    axu.set_ylabel("number of units concerned")
    axu.legend(["bert segmentation", "regex segmentation"])
    axu.set_ylabel("number of units concerned")
    axu.legend(["bert segmentation", "regex segmentation"])
    axu.figure.savefig('results_eval_length/number_of_units_bar_global.png')