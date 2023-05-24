from functools import partial
import argparse
from typing import Union, Tuple, List, Dict, Set
from pathlib import Path
import json
import os
import numpy as np
from tqdm import tqdm
import torch as tc
import supar
import re
from itertools import chain
import gc
import time
import unicodedata

from methods import data_handler as dh
from methods.utils import MyJsonSerializer

def add_args(parser:argparse.ArgumentParser):
    parser.add_argument(
        "--num_edges", 
        action="store", 
        default=[1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100],
        nargs="+",
        type=int,
        required=False,
        help="number of edges to use for lookup. This is the hyper-parameter `m` in the paper."
    )
    parser.add_argument(
        "--out_dir", 
        type=str,
        required=True,
        help="directory to save the lookup files to."
    )
    parser.add_argument(
        "--parser_id", 
        type=str,
        required=False,
        default='crf-con-roberta-en',
        help="parser id from supar."
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        required=False,
        default=100,
        help="batch size for matrix multiplication. (batch size for embeddings is chosen automatically independent of this parameter)"
    )


def get_args(parser:argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    
    args.out_fns = [os.path.join(args.out_dir, f'numedges={num}.json') for num in args.num_edges]
    
    return args

def remove_duplicated(seq:list):
    """
    Remove duplicated elements in a list while preserving the order. Copied from https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_substrings(sentence:str, parser:supar.parsers.con.CRFConstituencyParser)->Tuple[Set[str]]:
    """
    :param sentence: sentence to extract substrings from
    :param parser: parser to use for extracting substrings
    :return: tuple of two sets. The first set contains all non-leaf constituents of the sentence (which includes the whole sentence itself). The second set contains all leaf constituents.
    """
    try:
        dataset = parser.predict(sentence, lang='en', prob=True, verbose=False)
    except IndexError:
        print(f'Warning: IndexError in parsing sentence. No subsentences (only whole sentence) will be used for:\n{sentence}')
        return {sentence}, set()
    except ValueError:
        print(f'Warning: ValueError in parsing sentence. No subsentences (only whole sentence) will be used for:\n{sentence}')
        return {sentence}, set()
    tree = dataset[0].values[2]
    substrings = set() 
    additional_substrings = set()
    for t in tree.subtrees():
        if t.label() in ['TOP', 'S', 'NP', 'VP']:
            regex_pattern = r'REGEXPATTERN'.join(t.leaves())
            regex_pattern = regex_pattern.replace('\\', '\\\\') \
                                         .replace('.','\.') \
                                         .replace('+','\+') \
                                         .replace('*','\*') \
                                         .replace('?','\?') \
                                         .replace('^','\^') \
                                         .replace('$','\$') \
                                         .replace('(','\(') \
                                         .replace(')','\)') \
                                         .replace('[','\[') \
                                         .replace(']','\]') \
                                         .replace('{','\{') \
                                         .replace('}','\}') \
                                         .replace('|','\|')
            regex_pattern = regex_pattern.replace('REGEXPATTERN' , '( |)*')  # the star is in case there are multiple spaces between two words (e.g. in ExplaGraphs dev split `urbanisation can  damage countryside.`)
            regex_pattern = re.compile(regex_pattern)

            mo = regex_pattern.search(sentence)
            assert mo != None, (regex_pattern, sentence)  # this can happen in very rare cases where the sentence contains some special characters that are not handled correctly by the parser or the regex pattern
            if t.height() >= 4:
                substrings.add(mo.group())
            else:
                additional_substrings.add(mo.group())

    return substrings, additional_substrings

def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

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

    print('get all sentences')
    print(f'{len(datahandler) = }')

    sentences = {datahandler.get_premise_and_conclusion(index=index) for index in range(len(datahandler))}
    sentences = {s[i] for s in sentences for i in range(2)}
    sentences = list(sentences)
    sentences.sort()  # just to make debugging easier due to more consistency. Result is independent of this sorting.

    if Path(os.path.join(args.out_dir, 'subsentences.json')).is_file():
        print('load subsentences from file')
        sentence_to_subsentence = json.load(open(os.path.join(args.out_dir, 'subsentences.json'), 'r'))
        sentence_to_subsentence = {k: set(v) for k, v in sentence_to_subsentence.items()}
    else:
        print('load parser')
        if args.device == 'cuda': 
            tc.cuda.set_device('cuda:0')
        parser = supar.Parser.load(args.parser_id)

        print('get substrings')
        sentence_to_subsentence:dict = {sentence: get_substrings(sentence, parser)[0] for sentence in tqdm(sentences, total=len(sentences))}
        del parser

        json.dump(sentence_to_subsentence, open(os.path.join(args.out_dir, 'subsentences.json'), 'w'), indent=2, cls=MyJsonSerializer)

    subsentences = list(set(chain.from_iterable(sentence_to_subsentence.values())))

    print('load sbert')
    assert datahandler.sbert_uri == datahandler.knowledgegraph['sBERT_uri'], (datahandler.sbert_uri, datahandler.knowledgegraph['sBERT_uri'])
    datahandler.load_sBERT()

    print('get subsentence embeddings')
    subsentence_embeddings = datahandler.sBERT.encode(sentences=subsentences, show_progress_bar=True, convert_to_tensor=False)

    print('get reference embeddings')
    reference_embeddings = np.vstack(datahandler.knowledgegraph.es['embedding'])  # converting to np.array first is faster than going from list of np.ararys to tc.tensor
    reference_embeddings = tc.tensor(reference_embeddings, requires_grad=False, device=args.device)
    
    num_relations_per_edge = [len(emb) for emb in datahandler.knowledgegraph.es['embedding']]
    embedding_index_to_edge_index = [edge_id for edge_id, num_relations in enumerate(num_relations_per_edge) for _ in range(num_relations)]  # list index is embedding index, value is edge index

    # delete embeddings from kg to free memory
    del datahandler.knowledgegraph.es['embedding']
    del datahandler.knowledgegraph.vs['embedding']
    gc.collect()

    print('get indices of embeddings with highest similarities')
    subsentence_embeddings = np.array_split(subsentence_embeddings, (len(subsentence_embeddings) // args.batch_size)+1, axis=0)

    subsentence_embeddings = [b for b in subsentence_embeddings if len(b) > 0]

    max_num_relevant_embeddings = max(max(args.num_edges)+50, max(args.num_edges)*2)  # only the highest max_num_relevant_embeddings similarities are considered. max_num_relevant_embeddings should be sufficiently large, such that when in the top-m similarity embeddings are edges with ore than one relation, then more embeddings can be considered to get top-m different edges. 

    indices = []
    for i, batch in enumerate(subsentence_embeddings):
        print(f'{time.localtime().tm_hour:0>2}:{time.localtime().tm_min:0>2}:{time.localtime().tm_sec:0>2} - batch {i+1}/{len(subsentence_embeddings)}')
        batch = tc.tensor(batch, requires_grad=False, device=args.device)
        sim = tc.matmul(batch, reference_embeddings.T).cpu().numpy()

        ind = np.argpartition(sim, -max_num_relevant_embeddings, axis=1)[:,-max_num_relevant_embeddings:]
        sim = sim[np.arange(sim.shape[0])[:,None], ind]  # subset of sim with only the highest max_num_relevant_embeddings similarities. This is done to avoid sorting the whole matrix.
        
        tmp_ind = np.argsort(sim, axis=1)[:,::-1]  # best indices on subset of sim
    
        ind = ind[np.arange(ind.shape[0])[:,None], tmp_ind]  # convert back to iriginal indices
        
        indices.append(ind)

    indices = np.vstack(indices)

    del subsentence_embeddings
    del reference_embeddings
    del sim

    print('get subsentence lookup dict')
    # initialize dict. key is number of edges, value is lookup_dict
    subsentence_lookup_dict_dict = {num:{} for num in args.num_edges}
    for subsentence_index, subsentence in tqdm(enumerate(subsentences), total=len(subsentences)):
        key = subsentence

        # ids[0] is index of embedding with lowest edgeweight
        ids = indices[subsentence_index, :]
        ids = [embedding_index_to_edge_index[id] for id in ids]  # convert embedding index to edge index
        ids = remove_duplicated(ids)

        nodes = set()
        for m, id in enumerate(ids[:max(args.num_edges)], start=1):
            e = datahandler.knowledgegraph.es[id]
            v1 = datahandler.knowledgegraph.vs.find(e.source)['name'] 
            v2 = datahandler.knowledgegraph.vs.find(e.target)['name'] 
            nodes = nodes.union({v1, v2})
            if m in subsentence_lookup_dict_dict.keys():
                subsentence_lookup_dict_dict[m][key] = list(nodes)

    print(f'now get and save real lookup dict to {args.out_fns = }')
    for fn, m in zip(args.out_fns, args.num_edges):
        lookup_dict = {}
        for sentence, subsentences in sentence_to_subsentence.items():
            key = sentence.lower()
            
            concepts = list(set(chain.from_iterable((subsentence_lookup_dict_dict[m][subsentence] for subsentence in subsentences))))
            
            lookup_dict[key] = concepts

        with open(fn, 'w') as f:
            json.dump(lookup_dict, f, indent=2)

if __name__ == "__main__":
    print = partial(print, flush=True)

    print('load args')
    parser = argparse.ArgumentParser()
    dh.add_args(parser=parser)
    add_args(parser=parser)
    args = get_args(parser=parser)

    print("starting main")
    main(args=args)
    print("done with main")
