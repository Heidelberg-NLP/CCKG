import sys
import os
from pathlib import Path
import json
import argparse
from time import time
from tqdm import tqdm
import functools
import igraph as ig 

from methods.yens_algorithm import YensAlgorithm
from methods import data_handler as dh
from methods.utils import MyJsonSerializer

def add_args(parser):
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="filename of outputdir",
    )
    parser.add_argument(
        "--out_fn",
        type=str,
        required=True,
        help="filename of outputfile (relative to out_dir)",
    )
    parser.add_argument(
        "--edgeweight_to_pc",
        default='pc',
        type=str,
        required=False,
        help="if edgeweight is computed relative to : p -> premise; c -> conclusion; pc -> premise and conclusion jointly; min_p_c -> the minimum of premise and conclusion; min_p_c_pc -> the minimum of premise, conclusion and premise and conclusion jointly",
    )
    parser.add_argument(
        "--start_index",
        default=0,
        type=int,
        required=False,
        help="start index of data which should be used",
    )
    parser.add_argument(
        "--end_index",
        default=None,
        #type=Union[int, None, str],
        required=False,
        help="maximum index of data which should be used",
    )
    parser.add_argument(
        "--yen_edge_weight",
        default='weight',
        required=False,
        help="edgeweight to be used. either `None`, a string (edge-attribute)",
    )
    parser.add_argument(
        "--yen_num_shortest_paths",
        type=int,
        required=True,
        help="k: number of shorstest paths which Yen should find per concept-pair. ",
    )
    parser.add_argument(
        "--yen_avoid_circles",
        default=True,
        type=bool,
        required=False,
        help="whether to avoid circles in Yen's Algorithm. Turning this off might speed up computations, but might also lead to wrong results. ",
    )

def get_args(parser):
    args = parser.parse_args()
    if args.yen_edge_weight == 'None':
        args.yen_edge_weight = None
    if args.end_index == 'None' or args.end_index == 'inf':
        args.end_index = None
    elif isinstance(args.end_index, str):
        args.end_index = int(args.end_index)
    return args

def get_edge(vindex1:int, vindex2:int, g:ig.Graph):
    """
    :param vindex1: index of one node
    :param vindex2: index of the other node
    :param g: graph
    :return: tuple (relation, all_relations, weight, nl_relation, weight_to_pc, source, target)
    """
    es1 = g.es.select(_source=vindex1, _target=vindex2)
    es2 = g.es.select(_source=vindex2, _target=vindex1)
    if len(es1) == 0:
        e = es2[0]
    elif len(es2) == 0:
        e = es1[0]
    else:
        if es1[0]['weight'] < es2[0]['weight']:
            e = es1[0]
        else:
            e = es2[0]

    return (e['weight_relation'], e['relation'], e['weight'], e['nl_relation'], e['weight_to_pc'], e['weight_source'], e['weight_target'])

def main(args):    
    """
    creates shortest path dict in the format:
        {
            'explagraph_index': `index`,
            'times_per_task': {...},
            'shortest_paths': {
                '(sourcenode_index, targetnode_index)': {
                    'paths': [node_path_1, node_path_2, ..., node_path_k],
                    'paths_edges': [edge_path_1, edge_path_2, ..., edge_path_k],
                    'costs': [cost_path_1, cost_path_2, ..., cost_path_k],
                    'times': ...,
                }
            }
        }
    where node_path_i is:
        [(node_1_index, node_1_name), ..., (node_n_index, node_n_name)]
    and edge_path_i is:
        [edge_1, ..., edge_n-1]
    where edge_j is:
        (relation_woth_lowest_weight_on_edge, all_relations_on_edge, weight_of_edge, natural_language_relation, weight_to_pc, source_of_edge, target_of_edge)
    where weight_to_pc is:
        p -> weight was calculated by using the premise as a reference sentence
        c -> weight was calculated by using the conclusion as a reference sentence
        pc -> weight was calculated by using the premise and conclusion jointly as a reference sentence
    """
    # create outfilenames and save args
    out_fn = os.path.join(args.out_dir, args.out_fn+'.jsonl')
    
    # load data in out_fn and adjust start_index
    if Path(out_fn).is_file():
        with open(out_fn, 'r') as f:
            line = f.readlines()[-1]
        old_graph_dict = json.loads(line)
        args.start_index = old_graph_dict['index'] + 1
        if args.end_index != None:
            if args.start_index >= args.end_index:
                print(f'file is already fully done. Exiting now. \n{args.start_index=}, {args.end_index=}')
                sys.exit()
        print(f'{out_fn = } exists. skipping to {args.start_index = }')

    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f'will save to directory {args.out_dir}. Filename is {out_fn}')
    with open(os.path.join(args.out_dir, f"args_{args.out_fn}.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2, cls=MyJsonSerializer)

    print("loading datahandler")
    datahandler = args.datahandler(
        lookup=args.lookup,
        datasplit=args.datasplit,
        knowledgegraph=args.knowledgegraph,
        sbert_uri=args.sbert_uri,
        device=args.device,
        verbose=args.verbose,
        root=args.root, 
        r2nl=args.r2nl,
    )
    if args.end_index == None:
        args.end_index = len(datahandler)
    elif args.end_index > len(datahandler):
        args.end_index = len(datahandler)
        if args.start_index >= args.end_index:
                print(f'file is already fully done, because we had to adjust args.end_index to the length of the datase: {len(datahandler) = }. Exiting now. \n{args.start_index=}, {args.end_index=}')
                sys.exit()

    # iterate over explagraph data
    for index in range(args.start_index, args.end_index):
        print(f'\n--> {index = } / {args.end_index = }')
        t_start_index = time()
        result_dict = {'index': index,
                       'times_per_task': {},
                       'shortest_paths': {}}

        # add edge_weights
        t_tmp = time()
        print('(computing edge costs...)')
        if args.yen_edge_weight != None:
            datahandler.add_edge_attribute_costs_from_edge_embedding(index=index, g=datahandler.knowledgegraph, to_pc=args.edgeweight_to_pc)
        elif args.yen_edge_weight == None:
            datahandler.knowledgegraph.es['weight'] = 1

        datahandler.add_attribute_pc_concept(index=index, g=datahandler.knowledgegraph)
        datahandler.knowledgegraph_undirected = datahandler.knowledgegraph.copy()
        datahandler.directed_to_undirected(g=datahandler.knowledgegraph_undirected)
        result_dict['times_per_task']['add_edge_attribute_costs'] = time()-t_tmp

        # load yen
        yen = YensAlgorithm(
            graph_or_pathtograph=datahandler.knowledgegraph_undirected,
            debug=False,
            time=False,
            avoid_circles=args.yen_avoid_circles,
            load_in_yen=True
        )

        # concepts in premise and conclusion
        vs_pc = datahandler.knowledgegraph.vs.select(is_in_premise_union_conclusion=True)
        t_tmp = time()

        for i1, v1 in tqdm(enumerate(vs_pc), total=len(vs_pc)):
            v2s = vs_pc[i1+1:]
            if args.yen_num_shortest_paths == 1:
                targets = [v.index for v in v2s]
                result_dict_paths, result_dict_costs, result_dict_times = yen.call_dijkstra(
                    source=v1.index, 
                    targets=targets, 
                    weights='weight', 
                    return_time=True
                )
            for v2 in v2s:
                key = str((v1.index, v2.index))  
                result_dict['shortest_paths'][key] = {}

                if args.yen_num_shortest_paths > 1:
                    # get shortest paths
                    paths, costs, times = yen(
                        source=v1.index, 
                        target=v2.index, 
                        num_k=args.yen_num_shortest_paths,
                        weights='weight', 
                        return_time=True
                    )
                else:
                    paths = result_dict_paths[v2.index]
                    costs = result_dict_costs[v2.index]
                    times = result_dict_times[v2.index]

                    if args.yen_edge_weight != None:  # if a weight is used then only one shortest path is used, even if multiple ones have the same (shortest) cost. This is done to stay consistent to to yen.__call__()
                        if len(paths)>0:
                            paths = [paths[0]]
                            costs = [costs[0]]
                            times = [times[0]]

                # add node_name and node_weight to paths
                paths = [[(v, datahandler.knowledgegraph.vs.find(v)['name']) for v in path] for path in paths]

                if args.yen_edge_weight != None:
                    edges = [[get_edge(path[i][0], path[i+1][0], g=datahandler.knowledgegraph) for i in range(len(path)-1)] for path in paths]
                # save to dict
                result_dict['shortest_paths'][key]['paths'] = paths
                if args.yen_edge_weight != None:
                    result_dict['shortest_paths'][key]['paths_edges'] = edges
                result_dict['shortest_paths'][key]['costs'] = costs
                result_dict['shortest_paths'][key]['times'] = times
        result_dict['times_per_task']['get_all_shortest_paths'] = time()-t_tmp
        result_dict['times_per_task']['total_time'] = time()-t_start_index
        with open(out_fn, 'a') as f:
            json.dump(result_dict, f)
            f.write("\n")

if __name__ == '__main__':
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    dh.add_args(parser=parser)
    add_args(parser=parser)
    args = get_args(parser=parser)

    print('start main...')
    main(args=args)
    print('done with main')
