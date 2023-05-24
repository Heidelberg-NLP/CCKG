from argparse import Namespace
from pathlib import Path
import json
import igraph as ig
from tqdm import tqdm
import torch as tc
from sentence_transformers import SentenceTransformer
import functools

class Args(Namespace):
    def __init__(self):
        super().__init__()

        self.fn_inKG = 'data/knowledgegraph/CN_withRelatedTo/natural_r2nl/graph.pickle'

        self.out_dir = 'data/knowledgegraph/CN_withRelatedTo/natural_r2nl/'
        self.fn_outKG = 'graph_new.pickle'
        self.fn_r2nl = 'data/knowledgegraph/CN_withRelatedTo/r2nls/natural_r2nl.json'

        self.add_edge_embeddings = True
        self.add_node_embeddings = True

        self.model_id = 'all-mpnet-base-v2' 

        self.device = 'cuda' if tc.cuda.is_available else 'cpu'
        assert self.device == 'cuda'

        self.r2nl, self.target_relation_source = json.load(open(self.fn_r2nl, 'r'))


if __name__ == '__main__':
    args = Args()
    print = functools.partial(print, flush=True)
    print(f'{args = }')

    # create directory for new graph
    print(f'create directory and save args to {args.out_dir = }')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True) 
    # save args
    json.dump(args.__dict__, open(args.out_dir+'args.json', 'w'), indent=2)

    # load knowledge graph (KG)
    print('load ConceptNet')
    g = ig.read(args.fn_inKG, format="pickle")

    print(f'load sBERT to {args.device=}')
    model = SentenceTransformer(args.model_id, device=args.device)

    g['sBERT_uri'] = args.model_id

    node_names = [name.replace('_', ' ') for name in g.vs['name']]

    print(f'{args.add_edge_embeddings = }')
    if args.add_edge_embeddings:
        for e in tqdm(g.es):
            e['nl_relation'] = [
                    node_names[e.source]+' '+args.r2nl[rel]+' '+node_names[e.target] \
                    if rel not in args.target_relation_source else \
                    node_names[e.target]+' '+args.r2nl[rel]+' '+node_names[e.source] \
                    for rel in e['relation']]
        
            e['embedding'] = list(model.encode(e['nl_relation'], convert_to_tensor=False))

    print(f'{args.add_node_embeddings = }')
    if args.add_node_embeddings:
        for i, v in tqdm(enumerate(g.vs), total=g.vcount()):
            n = node_names[i]
            v['embedding'] = list(model.encode([n], convert_to_tensor=False))

    # save
    print(f'write graph to {args.out_dir+args.fn_outKG}', flush=True)
    ig.write(g, args.out_dir+args.fn_outKG, format='pickle')

    # load with
    # g = ig.read(args.out_dir+args.fn_outKG, format='pickle')

    print('DONE')
