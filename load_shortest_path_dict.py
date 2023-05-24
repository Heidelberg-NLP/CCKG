import igraph as ig 
import argparse
from time import time
from tqdm import tqdm
import sys
import functools

from methods import data_handler as dh
from methods import shortest_path_dicts_loader as spd_loader

def get_args(parser):
    args = parser.parse_args()
    return args

def main(args):
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

    if args.num_loaded_graphs == float('inf'):
        args.num_loaded_graphs = len(datahandler)-args.start_index
    else:
        args.num_loaded_graphs = int(args.num_loaded_graphs)

    if args.num_loaded_graphs <= 0:
        print(f'There are no graphs to be loaded because {args.num_loaded_graphs = } is <= 0')
        sys.exit()

    print('load graph loader')
    graph_loader = spd_loader.ShortestPathDictToIGraph(datahandler=datahandler)

    print('load graphs')
    t1 = time()
    graphs = graph_loader.load_shortest_path_dict(
        fn=args.fn_shortest_path_dict,
        yen_k=args.loadgraph_yen_k,
        only_pc_shortest_paths=args.only_pc_shortest_paths,
        all_edges=args.all_edges,
        start_index=args.start_index,
        num_graphs=args.num_loaded_graphs,
        shortestpathordering=args.shortestpathordering,
    )
    print(f'    loading graphs took {time()-t1} seconds')

    # list of igraph graphs
    assert isinstance(graphs, list) 
    assert all((isinstance(g, ig.Graph) for g in graphs))

    pruner = spd_loader.PruneListOfGraphs(
        graphs=graphs,
        fn_prune=args.fn_prune_data,
    )

    pruned_graphs = pruner.prune_prunestep()
    # list of igraph graphs
    assert isinstance(pruned_graphs, list)
    assert all((isinstance(g, ig.Graph) for g in pruned_graphs))

    print('example graph:')
    print(graphs[0].summary())
    print()
    print('example pruned graph:')
    print(pruned_graphs[0].summary())


if __name__ == '__main__':
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    dh.add_args(parser)
    spd_loader.add_args(parser)

    args = get_args(parser)
    print(f'{args = }')

    print('start main')
    main(args)
    print('DONE')
