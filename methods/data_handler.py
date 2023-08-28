import json
import os
import csv
import igraph as ig 
import numpy as np
import sentence_transformers as st
from typing import Optional, List, Tuple
import unicodedata

def str_to_datahandler(task:str):
    datahandlers = {
        'ExplaGraphs': ExplaGraphs_DataHandler,
        'ValNov': ValNovTaskA_DataHandler,
        # TDND include datahandler here
    }
    return datahandlers[task]

def str_to_bool(s:str):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError

def add_args(parser):
    """
    add arguments for datahandler to parser
    """
    parser.add_argument(
        "--datahandler",
        type=str_to_datahandler,
        required=True,
        help="which datahandler to use",
    )
    parser.add_argument(
        "--lookup",
        type=str,
        required=False,
        default='NONE',
        help="name of lookup",
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        required=True,
        help="datasplit",
    )
    parser.add_argument(
        "--knowledgegraph",
        type=str,
        required=False,
        default='NONE',
        help="name of knowledgegraph",
    )
    parser.add_argument(
        "--sbert_uri",
        type=str,
        required=True,
        help="uri to sBERT model",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default='cpu',
        help="device. cuda or cpu",
    )
    parser.add_argument(
        "--verbose",
        type=str_to_bool,
        required=False,
        default=False,
        help="set to True for printing of information",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=False,
        default='./',
        help="root of project",
    )
    parser.add_argument(
        "--r2nl",
        type=str,
        required=False,
        default=None,
        help="name of `relation to natural language` mapping",
    )

class AbstractDataHandler:
    def __init__(
        self,
        fn_raw_data:str,
        fn_knowledgegraph:str,
        fn_lookup_dict:str,
        sbert_uri:str,
        device:str,
        verbose:bool=False,
    ) -> None:

        self.fn_raw_data = fn_raw_data
        self.raw_data = self.load_raw_data(fn=fn_raw_data)
        if fn_knowledgegraph != None:
            self.knowledgegraph = self.load_knowledgegraph(fn=fn_knowledgegraph)
            # embs = list(itertools.chain.from_iterable(self.knowledgegraph.es['embedding']))
            # assert np.allclose(np.linalg.norm(embs, axis=1), 1), 'edge embeddings are not normalized'  # this assertion takes quite long
        if fn_lookup_dict != None:
            self.lookup_dict = self.load_lookup_dict(fn=fn_lookup_dict)

        self.sbert_uri = sbert_uri
        self.device = device
        self.sBERT = None
        self.verbose = verbose

    @classmethod
    def _to_json(cls):
        return cls.__name__

    def __len__(self) -> int:
        return len(self.raw_data)

    def load_raw_data(self, fn:str) -> list:
        raise NotImplementedError

    def load_knowledgegraph(self, fn:str) -> ig.Graph:
        g = ig.read(fn, format="pickle")
        return g

    def load_lookup_dict(self, fn:str) -> dict:
        if fn == None:
            return None

        lookup_dict = json.load(open(fn, 'r'))
        return lookup_dict

    def get_premise_and_conclusion(self, index:int) -> Tuple[str]:
        raise NotImplementedError

    def get_concepts_from_lookupdict(self, key) -> List[str]:
        return self.lookup_dict[key]

    def load_sBERT(self) -> None:
        self.sBERT = st.SentenceTransformer(self.sbert_uri, device=self.device)
    
    @staticmethod
    def get_edge_embeddings_matrix(g:ig.Graph) -> None:
        try:
            return g['edge_embeddings'], g['edge_members']
        except KeyError:
            embeddings = []
            members = []
            for e in g.es:
                tmp_members = []
                for tmp_embedding in e['embedding']:
                    embeddings.append(tmp_embedding)
                    tmp_members.append(len(embeddings)-1)
                members.append(tmp_members)
            embeddings = np.array(embeddings)
            assert len(members) == g.ecount()
            g['edge_embeddings'], g['edge_members'] = embeddings, members
            return g['edge_embeddings'], g['edge_members']

    def get_reference_embedding(self, index, to_pc, free_text):
            premise, conclusion = self.get_premise_and_conclusion(index=index)

            if to_pc == 'p':
                argument = [premise]
                weight_to_pc = ['p']
            elif to_pc == 'c':
                argument = [conclusion]
                weight_to_pc = ['c']
            elif to_pc == 'pc':
                argument = [premise+' '+conclusion]
                weight_to_pc = ['pc']
            elif to_pc == 'min_p_c':
                argument = [premise, conclusion]
                weight_to_pc = ['p', 'c']
            elif to_pc == 'min_p_c_pc':
                argument = [premise, conclusion, premise+' '+conclusion]
                weight_to_pc = ['p', 'c', 'pc']
            elif to_pc == 'free_text':
                assert free_text != None
                argument = [free_text]
                weight_to_pc = [f'free_text:{free_text}']
            else:
                raise ValueError
            assert len(argument) == len(weight_to_pc)

            if self.sBERT == None: 
                if self.verbose: print(f'loading sBERT {self.sBERT_uri}')
                self.load_sBERT()

            if self.verbose: print('get reference embedding')
            reference_embeddings = self.sBERT.encode(argument, convert_to_tensor=False)
            return reference_embeddings, weight_to_pc
    
    def add_edge_attribute_costs_from_edge_embedding(
        self, 
        index:Optional[int], 
        g:ig.Graph, 
        to_pc:str='pc', 
        free_text:Optional[str]=None,
        reference_embeddings:Optional[str]=None
    ) -> None:
        """
        :param to_pc:  p:      weight is to premise only
                c:      weight is to conclusion only
                pc:     weight is to premise and conclusion jointly
                min_p_c: weight is to min of weight to p and weight to c
                min_p_c_pc: weight is to min of weight to p and weight to c and weight to pc
                free_text: compute similarity to `free_text` instead of premise and conclusion
                reference_embedding: compute similarity to `reference_embeddings` instead of premise and conclusion or free_text
        :param free_text: if to_pc == 'free_text', then this is the text to compute similarity to
        :param reference_embeddings: if to_pc == 'reference_embeddings', then this is the embedding to compute similarity to
        """
        if self.verbose: print('adding edge weights')
        assert to_pc in ['p', 'c', 'pc', 'min_p_c', 'min_p_c_pc', 'free_text', 'reference_embeddings'], f'{to_pc=} is not a valid option'

        if to_pc == 'reference_embeddings':
            assert reference_embeddings is not None, f'you need to pass a reference embedding if to_pc == `reference_embeddings`\n{reference_embeddings = }'
            weight_to_pc = ['reference_embedding']
        else:
            reference_embeddings, weight_to_pc = self.get_reference_embedding(index, to_pc, free_text)

        reference_embeddings = reference_embeddings.T

        embeddings, members = self.get_edge_embeddings_matrix(g)
        all_cosine_sims = np.dot(embeddings, reference_embeddings)
        all_weights = 1 - (all_cosine_sims + 1) / 2

        for (e, mem) in zip(g.es, members):
            weights = []
            for m in mem:
                weights.append(all_weights[m,:])
            weights = np.array(weights)  # array of weights for all arguments and all edge relations (in case two concepts are connected by multiple relations)
            best_index = np.argmin(weights)
            assert len(weights.shape) == 2
            idx0 = best_index // weights.shape[1]
            idx1 = best_index%weights.shape[1]
            assert float(weights[idx0, idx1]) == float(np.min(weights)), str(weights) + '\n' + f'{weights[idx0, idx1]} -- {best_index} -- {idx0} -- {idx1}'

            e['weight'] = float(weights[idx0, idx1])  # minimal weight
            e['weight_relation'] = e['relation'][idx0]  # relation with minimal weight
            e['weight_source'] = e.source  # source with minimal weight
            e['weight_target'] = e.target  # target with minimal weight
            e['weight_to_pc'] = weight_to_pc[idx1]  # argument with minimal weight

    def add_attribute_pc_concept(self, index:int, g:ig.Graph) -> None:
        premise, conclusion = self.get_premise_and_conclusion(index=index)

        concepts_premise = self.get_concepts_from_lookupdict(premise.lower())
        concepts_conclusion = self.get_concepts_from_lookupdict(conclusion.lower())

        in_premise = [(v['name'] in concepts_premise) for v in g.vs]
        in_conclusion = [(v['name'] in concepts_conclusion) for v in g.vs]
        in_premise_union_conclusion = [(v['name'] in concepts_premise+concepts_conclusion) for v in g.vs]

        g.vs['is_in_premise'] = in_premise
        g.vs['is_in_conclusion'] = in_conclusion
        g.vs['is_in_premise_union_conclusion'] = in_premise_union_conclusion

    @staticmethod
    def directed_to_undirected(g:ig.Graph) -> None:
        combine_edges_dict = {'weight': 'min'}
        g.to_undirected(combine_edges=combine_edges_dict)


class ExplaGraphs_DataHandler(AbstractDataHandler):
    def __init__(
        self, 
        lookup : str, 
        datasplit : str, 
        knowledgegraph : str,
        sbert_uri : str,
        device : str,
        verbose : bool = False,
        root : str = '../', 
        r2nl : Optional[str] = None
    ) -> None:
        """
        :param lookup: filename to lookup dict
        :param datasplit: train or dev set
        :param knowledgegraph: `CN_withoutRelatedTo`, `CN_withRelatedTo` or `ExplaKnow`. Pass `NONE` if no knowledge graph is used. 
        :param sbert_uri: uri to sBERT model
        :param device: `cpu` or `cuda`
        :param verbose: bool. True for printing of information
        :param root: root directory of the project
        :param r2nl: determines how the triplets in the knowledge graph were verbalized. `natural` or `static`. Can be `NONE` if no knowledge graph is used
        """

        assert r2nl in ['natural', 'static', 'NONE']
        self.datasplit = datasplit
        self.knowledgegraph = knowledgegraph
        
        assert self.datasplit in ['train', 'dev'], self.datasplit
        assert self.knowledgegraph in ['CN_withoutRelatedTo', 'CN_withRelatedTo', 'ExplaKnow', 'NONE']

        fn_raw_data = os.path.join(root, f'data/tasks/ExplaGraphs/raw_data/{self.datasplit}.tsv')

        if self.knowledgegraph == 'NONE':
            print('WARNING: not loading a knowledgegraph. Some functions might not be working. ')
            fn_knowledgegraph = None
        else:
            fn_knowledgegraph = os.path.join(root, f'data/knowledgegraph/{self.knowledgegraph}/{r2nl}_r2nl/graph.pickle')

        if lookup == 'NONE':
            fn_lookup_dict=None
        elif lookup.startswith('sbert_'):
            assert len(lookup) in [len('sbert_?'), len('sbert_??')], (lookup, 'in principle higher values are possible. If that is what you want then just delete this assertion. I just put it here to avoid matchings like `sbert_random-word_5`')
            num_edges = int(lookup[len('sbert_'):])
            fn_lookup_dict = os.path.join(root, f'data/tasks/ExplaGraphs/lookups/{r2nl}_r2nl/{self.knowledgegraph}/sbert/numedges={num_edges}.json')
        else:
            raise ValueError(lookup)

        super().__init__(
            fn_raw_data=fn_raw_data, 
            fn_knowledgegraph=fn_knowledgegraph,
            fn_lookup_dict=fn_lookup_dict,
            sbert_uri=sbert_uri,
            device=device,
            verbose=verbose
        )

    def load_raw_data(self, fn:str) -> list:
        with open(fn, 'r') as f:
            raw_data = [line.strip().split('\t') for line in f.readlines()]
        return raw_data

    def get_label(self, index:int) -> str:
        stance = self.raw_data[index][2]
        return stance

    def get_premise_and_conclusion(self, index:int) -> Tuple[str]:
        premise = self.raw_data[index][1]
        conclusion = self.raw_data[index][0]
        return premise, conclusion

class ValNovTaskA_DataHandler(AbstractDataHandler):
    def __init__(
        self, 
        lookup : str, 
        datasplit : str, 
        knowledgegraph : str,
        sbert_uri : str,
        device : str,
        verbose : bool = False,
        root : str = '../', 
        r2nl : Optional[str] = None
    ) -> None:
        """
        :param lookup: filename to lookup dict
        :param datasplit: train or dev set
        :param knowledgegraph: `CN_withoutRelatedTo`, `CN_withRelatedTo` or `ExplaKnow`. Pass `NONE` if no knowledge graph is used. 
        :param sbert_uri: uri to sBERT model
        :param device: cuda or cpu
        :param verbose: set to True for printing of information
        :param root: root directory of the project
        :param r2nl: determines how the triplets in the knowledge graph were verbalized. `natural` or `static`. Can be `NONE` if no knowledge graph is used
        """

        assert r2nl in ['natural', 'static', 'NONE']
        self.datasplit = datasplit
        self.knowledgegraph = knowledgegraph
        
        assert self.datasplit in ['train', 'dev', 'test'], self.datasplit
        assert self.knowledgegraph in ['CN_withoutRelatedTo', 'CN_withRelatedTo', 'NONE']

        fn_raw_data = os.path.join(root, f'data/tasks/ValNov/raw_data/TaskA_{self.datasplit}.csv')

        if self.knowledgegraph == 'NONE':
            print('WARNING: not loading a knowledgegraph. Some functions might not be working. ')
            fn_knowledgegraph = None
        else:
            fn_knowledgegraph = os.path.join(root, f'data/knowledgegraph/{self.knowledgegraph}/{r2nl}_r2nl/graph.pickle')

        if lookup == 'NONE':
            fn_lookup_dict=None
        elif lookup.startswith('sbert_constituentparser_'):
            assert len(lookup) in [len('sbert_constituentparser_?'), len('sbert_constituentparser_??')], (lookup, 'in principle higher values are possible. If that is what you want then just delete this assertion. I just put it here to avoid matchings like `sbert-constituentparser_random-word_5`')
            num_edges = int(lookup[len('sbert_constituentparser_'):])
            fn_lookup_dict = os.path.join(root, f'data/tasks/ValNov/lookups/{r2nl}_r2nl/{self.knowledgegraph}/sbert_constituentparser/numedges={num_edges}.json')
        elif lookup.startswith('sbert_'):
            assert len(lookup) in [len('sbert_?'), len('sbert_??')], (lookup, 'in principle higher values are possible. If that is what you want then just delete this assertion. I just put it here to avoid matchings like `sbert_random-word_5`')
            num_edges = int(lookup[len('sbert_'):])
            fn_lookup_dict = os.path.join(root, f'data/tasks/ValNov/lookups/{r2nl}_r2nl/{self.knowledgegraph}/sbert/numedges={num_edges}.json')
        else:
            raise ValueError(lookup)

        super().__init__(
            fn_raw_data=fn_raw_data, 
            fn_knowledgegraph=fn_knowledgegraph,
            fn_lookup_dict=fn_lookup_dict,
            sbert_uri=sbert_uri,
            device=device,
            verbose=verbose,
        )
    
    def load_raw_data(self, fn:str)->list:
        raw_data = []
        with open(fn, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, row in enumerate(reader):
                if self.datasplit == 'test-without-labels':
                    if i == 0: # the file has a header line
                        assert row == ['topic','Premise','Conclusion'], row
                        continue 
                    assert len(row)==3
                elif self.datasplit == 'test':
                    if i == 0: # the file has a header line
                        assert row == ['topic','Premise','Conclusion', 'Validity','Validity-Confidence', 'Novelty', 'Novelty-Confidence', 'Topic-in-dev-split'], row
                        continue 
                    assert len(row)==8
                    assert row[3] in ['-1', '0', '1'], row
                    assert row[5] in ['-1', '0', '1'], row
                    assert row[4] in ['defeasible', 'majority', 'confident', 'very confident'], row
                    assert row[6] in ['defeasible', 'majority', 'confident', 'very confident'], row
                    assert row[7] in ['yes', 'no'], row
                else:
                    if i == 0: # the file has a header line
                        assert row == ['topic','Premise','Conclusion', 'Validity','Validity-Confidence', 'Novelty', 'Novelty-Confidence'], row
                        continue 
                    assert len(row)==7
                    assert row[3] in ['-1', '0', '1'], row
                    assert row[5] in ['-1', '0', '1'], row
                    assert row[4] in ['defeasible', 'majority', 'confident', 'very confident'], row
                    assert row[6] in ['defeasible', 'majority', 'confident', 'very confident'], row
                raw_data.append(row)
        return raw_data

    def get_premise_and_conclusion(self, index:int)->Tuple[str]:
        premise = self.raw_data[index][1]
        conclusion = self.raw_data[index][2]
        return premise, conclusion

    def get_label(self, index):
        assert self.datasplit != 'test-without-labels'
        validity = self.raw_data[index][3]
        validity_confidence = self.raw_data[index][4]
        novelty = self.raw_data[index][5]
        novelty_confidence = self.raw_data[index][6]
        return validity, validity_confidence, novelty, novelty_confidence

    def get_topic_in_dev_split(self, index):
        assert self.datasplit == 'test'
        topic_in_dev_split = self.raw_data[index][7]
        return topic_in_dev_split


"""
class ExampleNewData_DataHandler(AbstractDataHandler): # TDND change name
    def __init__(
        self, 
        lookup : str, 
        datasplit : str, 
        knowledgegraph : str,
        sbert_uri : str,
        device : str,
        verbose : bool = False,
        root : str = '../', 
        r2nl : Optional[str] = None
    ) -> None:
        self.datasplit = datasplit
        self.knowledgegraph = knowledgegraph

        fn_raw_data = os.path.join(root, f'path/to/raw/data')  # TDND change path

        if self.knowledgegraph == 'NONE':
            print('WARNING: not loading a knowledgegraph. Some functions might not be working. ')
            fn_knowledgegraph = None
        else:
            fn_knowledgegraph = os.path.join(root, f'path/to/graph.pickle')  # TDND change path

        if lookup == 'NONE':
            fn_lookup_dict=None
        elif lookup.startswith('sbert_'):  # TDND add other relevant lookups
            assert len(lookup) in [len('sbert_?'), len('sbert_??')], (lookup, 'in principle higher values are possible. If that is what you want then just delete this assertion. I just put it here to avoid matchings like `sbert_random-word_5`')
            num_edges = int(lookup[len('sbert_'):])
            fn_lookup_dict = os.path.join(root, f'path/to/lookup/numedges={num_edges}.json')  # TDND change path
        else:
            raise ValueError(lookup)

        super().__init__(
            fn_raw_data=fn_raw_data, 
            fn_knowledgegraph=fn_knowledgegraph,
            fn_lookup_dict=fn_lookup_dict,
            sbert_uri=sbert_uri,
            device=device,
            verbose=verbose
        )

    def load_raw_data(self, fn:str) -> list:
        # TDND load raw data
        raw_data = []
        return raw_data

    def get_premise_and_conclusion(self, index:int) -> Tuple[str]:
        # TDND get premise and conclusion from raw data `index`. If you work with non-argumentative data, then this function should return the two sentences between which you want to construct the CCKG. 
        # If you want to construct CCKGs for individual sentences, then return that sentence twice. # todo test this feature
        premise = ''
        conclusion = ''
        return premise, conclusion
"""