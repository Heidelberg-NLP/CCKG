import argparse
from time import time
from tqdm import tqdm
import json
import sys
import functools
from pathlib import Path

from methods import data_handler as dh
from methods import shortest_path_dicts_loader as spd_loader
from methods import shortest_path_dict_pruner as spd_pruner
from methods.utils import MyJsonSerializer

def add_args(parser):
    parser.add_argument(
        '--fn_out', 
        type=str, 
        required=True,
        help='path to save the pruned shortest path dict to'
    )

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
        if args.start_index + args.num_loaded_graphs > len(datahandler):
            args.num_loaded_graphs = len(datahandler) - args.start_index

    # load data in out_fn and adjust start_index
    if Path(args.fn_out).is_file():
        num_lines = sum(1 for _ in open(args.fn_out, 'r'))
        args.start_index = args.start_index + num_lines

        args.num_loaded_graphs -= num_lines

        print(f'{args.fn_out = } exists. skipping to {args.start_index = }')

    if args.num_loaded_graphs <= 0:
        print(f'file is already fully done. Exiting now. \n{args.start_index=}, {args.num_loaded_graphs=}')
        sys.exit()

    # create folder to save to
    Path(args.fn_out).parent.mkdir(parents=True, exist_ok=True)

    fn_args = args.fn_out.split('/')
    fn_args[-1] = 'args_'+fn_args[-1]
    fn_args = '/'.join(fn_args)
    print(f'save args to {fn_args=}')
    with open(fn_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2, cls=MyJsonSerializer)

    print('load graph loader')
    graph_loader = spd_loader.ShortestPathDictToIGraph(datahandler=datahandler)

    print('load pruner')
    pruner = args.pruner(
        datahandler=datahandler
    )

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

    print('prune and save graphs')
    for g in tqdm(graphs):
        if g.vcount() == 0:
            del_names = [[]]
        else:
            _, del_names = pruner.prune(
                g=g,
                keep_graph_connected=True,
                protected_concepts=['premise', 'conclusion']
            )

        with open(args.fn_out, 'a') as f:
            json.dump(del_names, f)
            f.write("\n")


if __name__ == '__main__':
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    dh.add_args(parser)
    spd_loader.add_args_loader(parser)
    spd_pruner.add_args(parser)
    add_args(parser)
    args = get_args(parser)
    print(f'{args = }')

    print('start main')
    main(args)
    print('DONE')
