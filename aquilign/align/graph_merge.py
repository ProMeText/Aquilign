import networkx as networkx
import string

def desambiguise(object:list, labels:list) -> list:
    """
    On ajoute à chaque noeud le document d'origine, pour pouvoir transformer dans un graphe global ensuite:
    ((0, 0), 
    (1, 1), 
    (2, 2), 
    (3, 3), etc] 
    -> 
    [(('0_a',), ('0_b',)), 
    (('1_a',), ('1_b',)),
    etc]
    """
    as_unique_nodes = []
    for alignment_unit in object:
        A, B = alignment_unit
        if isinstance(A, int): A = (A,)
        if isinstance(B, int): B = (B,)
        ranges_A = []
        ranges_B = []
        for element in A:
            ranges_A.append(f"{str(element)}_{labels[0]}")
        for element in B:
            ranges_B.append(f"{str(element)}_{labels[1]}")
        as_unique_nodes.append((tuple(ranges_A), tuple(ranges_B)))
    return as_unique_nodes

def deconnect(object:list) -> tuple:
    """
    On passe de groupes à des paires de positions
    [(('5_a', '6_a', '7_a'), ('5_b', '6_b'))]
    -> [ ('5_a', '5_b'), ('5_a', '6_b'), ('6_a', '5_b'), ('6_a', '6_b'), ('7_a', '5_b'), ('7_a', '6_b')]
    """
    
    final_list = []
    for alignment_unit in object:
        A, B = alignment_unit
        for item_A in A:
            for item_B in B:
                final_list.append((item_A, item_B))
    return tuple(final_list)

def merge_alignment_table(alignment_dict:dict, rerun_alignments) -> list:
    """
    Cette fonction va permettre de fusionner les items d'alignement à l'aide d'un passage par le graphe.
    Il en ressort une liste de dictionnaires, chaque item consistant en une unité d'alignement.
    """    
    # On désambiguise les noeuds
    G = networkx.petersen_graph()
    # On modifie la structure pour avoir des noeuds connectés 2 à 2 et des tuples
    possible_witnesses = string.ascii_lowercase[:len(alignment_dict) + 1]
    for index, value in alignment_dict.items():
        structured_a = deconnect(desambiguise(value, (possible_witnesses[0], possible_witnesses[int(index) + 1])))
        # Résultat de la forme: (('0_a', '0_b'), ('1_a', '1_b'), ('2_a', '2_b'), ('3_a', '3_b'), etc.)
        G.add_edges_from(structured_a)
    connected_nodes = []
    # On prend chaque noeud et on en ressort les noeuds connectés
    all_nodes = [node for node in G]
    for node in G:
        # https://stackoverflow.com/a/33089602
        connected_components = list(networkx.node_connected_component(G, node))
        connected_components.sort()
        connected_nodes.append(tuple(connected_components))

    # On supprime les noeuds redondants
    connected_nodes = list(set(connected_nodes))
    disconnected_nodes = list(set(all_nodes) - set(connected_nodes[0]))
    # Liste de la forme: [('27_a', '27_c', '28_a', '28_c', '29_c', '30_c', '33_b', '34_b'), ('19_c', '20_a', '20_c', '22_b'), etc.]
    
    # Ici c'est un bug: il y a des noeuds qui sont des entiers, je ne sais pas pourquoi. On les supprime.
    connected_nodes = [group for group in connected_nodes if not isinstance(group[0], int)]
    
    # On trie les noeuds par position
    connected_nodes.sort(key=lambda x: int(x[0].split('_')[0]))
    nodes_as_dict = []
    for connection in connected_nodes:
        wit_dictionnary = {}
        for document in possible_witnesses[:len(alignment_dict) + 1]:
            pos_list = [node.replace(f'_{document}', '') for node in connection if document in node]
            pos_list.sort(key=lambda x:int(x))
            wit_dictionnary[document] = pos_list
        nodes_as_dict.append(wit_dictionnary)
    nodes_as_dict.sort(key=lambda x: int(x['a'][0]))
    
    # Now we have to manage the omissions in the main document.
    omitted_pos = dict()
    # We gather all omissions for each witness in a dict
    print("Here")
    for idx, wit in enumerate(possible_witnesses):
        all_positions = [align_dict[wit] for align_dict in nodes_as_dict if align_dict[wit] != []]
        last_position = int(all_positions[-1][-1])
        not_present_positions = list(set(range(last_position + 1)) - set([int(position) for dictionnary in nodes_as_dict for position in dictionnary[wit]]))
        not_present_positions.sort()
        omitted_pos[wit] = not_present_positions
    print(f"Omitted pos dict: {omitted_pos}")
    
    if rerun_alignments:
        # print(omitted_pos)
        # On va chercher le dernier témoin
        # Ici il faut itérer sur chacun de lieux variants du dictionnaire d'alignement 
        # {0: [([0, 1], [0]), ([2], [])], 1: [([0, 1], [0]), ([2], [])]} où chaque item de liste correspond à une unité alignée
        print(f"Alignment dict: {alignment_dict}")
        print([bitext[-1][0] for bitext in alignment_dict.values()])
        if any([bitext[-1][0] == [] for bitext in alignment_dict.values()]):
            print("Ok, now we're talking")
            for idx, bitext in alignment_dict.items():
                print(f"Testing {idx}: {bitext}")
                if bitext[-1][0] != []:
                    continue
                else:
                    print("Stop")
                    print(bitext)
                    print(nodes_as_dict)
                    nodes_as_dict.append({possible_witness: [] for possible_witness in possible_witnesses})
                    # [nodes_as_dict[-1][possible_witnesses[int(idx) + 1]].extend(str(bitext[n][1][0])) for n, couple in enumerate(bitext) if couple[0] == []]
                    # nodes_as_dict[-1][possible_witnesses[int(idx) + 1]].extend(str(bitext[-1][1][0]))
                    print(f"Last_value: {bitext[-1][1][0]}")
                    # omitted_pos[wit].extend(bitext[-1][1])
                    print([(bitext[n][1]) for n, couple in
                     enumerate(bitext) if couple[0] == []])
                    [omitted_pos[wit].extend(bitext[n][1]) for n, couple in
                     enumerate(bitext) if couple[0] == []]

                    print(omitted_pos)
                    print(f"Result: {nodes_as_dict}")
                print(omitted_pos)
                
    # Sur ça: {0: [([0], [0, 1])], 1: [([0], [0, 1])], 2: [([0], [0, 1])], 3: [([0], [0, 1])], 
    #          4: [([0], [0, 1])], 5: [([0], [0, 1]), ([], [2]), ([], [3]), ([], [4])]}
    # Qui donne: [{'a': ['0'], 'b': ['0', '1'], 'c': ['0', '1'], 'd': ['0', '1'], 'e': ['0', '1'], 'f': ['0', '1'], 'g': ['0', '1']}]
    # Ça pose problème, il faut voir pourquoi (unité 6 par exemple). 
    # On peut tester d'ajouter le dernier index de la position 5, et de continuer la suite (détection des omissions), ça devrait marcher
    # Le problème semble arriver sur les dernières unités d'alignement d'un des témoins, quand il y a omission du témoin base
    
    
    for wit in possible_witnesses:
        print(f"\n{wit}")
        print(omitted_pos[wit])
        for omitted in omitted_pos[wit]:
            print(f"Omitted position {omitted} for wit {wit}")
            # Let's retrieve the corresponding alignment unit, we keep the index to allow the injection in case B (see below)
            if omitted != 0:
                try:
                    corresponding_alignment_unit = [(index, node) for index, node in enumerate(nodes_as_dict) if str(omitted - 1) in node[wit]][0]
                except IndexError:
                    corresponding_alignment_unit = [(index, node) for index, node in enumerate(nodes_as_dict)][-1]
            else:
                corresponding_alignment_unit = (0, nodes_as_dict[0])
            # print(corresponding_alignment_unit)
            # On a plusieurs cas de figure
            # Le premier: un fusion avec un "trou": (39, {'a': ['59', '60'], 'b': ['65', '67'], 'c': ['62', '63'], 'd': ['82'], 'e': ['58']})
            
            # Premier cas: il faut insérer l'omission dans un noeud déjà formé: 66, ['65', '67']
            if any(int(item) > omitted for item in corresponding_alignment_unit[1][wit]):
                print("Case A")
                copied_node = corresponding_alignment_unit[1]
                list_to_amend = copied_node[wit]
                list_to_amend.append(str(omitted))
                # print(f"Appending {omitted}")
                list_to_amend.sort(key=lambda x:int(x))
                copied_node[wit] = list_to_amend
                nodes_as_dict[corresponding_alignment_unit[0]] = copied_node
                
            # Deuxième cas, il faut créer une nouvelle unité d'alignement: 2005, ['2003', '2004']
            else:
                print("Case B")
                new_node = {wit:[] for wit in possible_witnesses}
                new_node[wit] = [str(omitted)]
                nodes_as_dict.insert(corresponding_alignment_unit[0] + 1, new_node)
                # print(new_node)
            # print("\n")
            # print(f"Results: {nodes_as_dict}")
    # print(f"Omission dict: {omitted_pos}")
    # print(f"Before reinjection: {alignment_dict}")
    # print(f"After reinjection: {nodes_as_dict}")
    return nodes_as_dict





if __name__ == '__main__':
    result_b = (((0), (0)), ((1, 2, 3), (1, 2)), ((4, 5), (3)), ((6, 7), (4)), ((8), (5)),
                ((9, 10, 11), (6)), ((12), (7, 8)), ((13), (9, 10)), ((14), (11)),
                ((15), (12)), ((16, 17), (13, 14, 15)), ((18), (16)), ((19), (17)),
                ((), (18)), ((20), (19, 20)), ((21), (21, 22)), ((22, 23, 24), (23)),
                ((25, 26), (24, 25, 26)), ((27), (27)), ((28), (28, 29, 30)), ((29), (31)),
                ((30, 31, 32), (32)), ((33), (33)), ((34, 35), (34)), ((36, 37), (35)),
                ((38), (36)), ((39), (37, 38, 39)))

    result_a = (((0), (0)), ((1), (1)), ((2), (2)), ((3), (3)), ((4), (4)), ((5, 6, 7), (5, 6)),
                ((8), (7, 8, 9, 10)), ((9), (11, 12)), ((10, 11), (13)), ((12), ()), ((13), (14)),
                ((14), (15, 16)), ((15), (17, 18)), ((16), (19)), ((17, 18, 19), (20, 21)),
                ((20), (22)), ((21), (23, 24)), ((22), (25)), ((23), (26)), ((24), (27)),
                ((25), (28, 29, 30)), ((26), (31, 32)), ((27, 28), (33, 34)), ((29), (35)),
                ((30, 31), (36, 37)), ((32), (38)), ((33), (39)), ((34), ()), ((35), ()),
                ((36), ()), ((37), ()), ((38), ()), ((39), ()))
    result = merge_alignment_table(result_a, result_b)
    print(result)
    