import igraph as ig 
import numpy as np 
import typing
import itertools

from methods import data_handler as dh

def str_to_pruner(task:str):
    pruners = {
        'sbert': PrunerBySBERT,
        'pagerank': PrunerByPagerank,
        'sbert_pagerank': PrunerBy_SBERT_Pagerank,
    }
    return pruners[task]

def add_args(parser):
    parser.add_argument(
        "--pruner",
        required=True,
        type=str_to_pruner,
        help='`sbert`, `pagerank` and `sbert_pagerank` are currently supported'
    )
    parser.add_argument(
        "--reference_pc",
        default='pc',
        type=str,
        required=False,
        help="`p` `c` or `pc`. Only Relevant for pruners using sBERT. Chooses if the reference embedding is done for premise, conclusion or premise-conclusion",
    )

class AbstractPruner():
    def __init__(self, datahandler:dh.AbstractDataHandler, kwargs: dict = {}):
        self.kwargs = kwargs  # kwargs for _get_node_ranking
        self.graph_to_undirected = True
        self.datahandler = datahandler  # actually only the sBERT model is needed, but the datahandler is loaded anyways

    @classmethod
    def _to_json(cls):
        return cls.__name__

    def _get_protected_nodes(self, g:ig.Graph, protected_concepts:typing.List[str]=['premise', 'conclusion']) -> typing.Set[str]:
        """
        returns set of nodenames of nodes which are in premsie / conclusion
        """
        assert len(protected_concepts) == len(set(protected_concepts)), 'there are duplicate entries'
        for ele in protected_concepts:
            assert ele in ['premise', 'conclusion'], f'{ele=} is not a valid option.'

        protected_nodes = set()
        if 'premise' in protected_concepts:
            tmp_protected_nodes =set(g.vs.select(is_in_premise=True)['name'])
            protected_nodes = protected_nodes.union(tmp_protected_nodes)

        if 'conclusion' in protected_concepts:
            tmp_protected_nodes =set(g.vs.select(is_in_conclusion=True)['name'])
            protected_nodes = protected_nodes.union(tmp_protected_nodes)

        return protected_nodes

    def _modify_graph(self, g:ig.Graph):
        """
        adds attribute `tmp_index` to graph nodes and makes graph undirected if needed
        """
        g.vs['tmp_index'] = [v.index for v in g.vs]  # index of original graph
        if self.graph_to_undirected:
            if 'weights' in self.kwargs.keys():
                combine_edges = {self.kwargs['weights']: min}
            else:
                combine_edges = {}
            g.to_undirected(combine_edges=combine_edges)
            assert not g.is_directed()
        else:
            assert g.is_directed()

    def _get_node_ranking(self, g:ig.Graph) -> typing.List[float]:
        """
        returns list of floats with one float per vertex. 
        Lower value means less relevant (i.e. gets deleted first)
        """
        raise NotImplementedError

    def _check_and_filter_if_seperator(self, g:ig.Graph, del_nodes:typing.Set[ig.Vertex]) -> typing.Set[ig.Vertex]:
        """
        Checks whether nodes can be removed from the graph without disconnecting the graph. If multiple nodes are given, and the multiple nodes are a separator, then it is checked whether a subset of the nodes is not a separator. 
        """
        if not g.is_connected(mode='weak'):
            # get subgraphs
            g.vs['tmp_cluster_index'] = [v.index for v in g.vs]
            clusters = g.clusters(mode='weak')
            # assign delnode to the correct subgraphs
            tmp_del_nodes = [set(nodes).intersection(del_nodes) for nodes in clusters]
            subgraphs = clusters.subgraphs()
            out_del_nodes = set()

            # check for each subgraph if del_nodes is a seperator
            for tmp_del_node, sub_g in zip(tmp_del_nodes, subgraphs):
                if len(tmp_del_node)==0: continue

                tmp_d = {sub_g.vs.find(tmp_cluster_index=d) for d in tmp_del_node}
                tmp_d = self._check_and_filter_if_seperator(sub_g, tmp_d)
                out_del_nodes = out_del_nodes.union({g.vs.find(d['tmp_cluster_index']) for d in tmp_d})

            del g.vs['tmp_cluster_index']
            return out_del_nodes
    
        if g.is_separator(del_nodes):
            if g.is_minimal_separator(del_nodes):  # none of the nodes can be deleted
                del_nodes = del_nodes - del_nodes
            else:  # some, but not all nodes can be deleted
                # check if removing num_nodes from del_nodes can make it not a seperator
                for num_nodes in range(len(del_nodes)-1, 0, -1):  # len(del_nodes) is enough as we already checked that it is not a minimal seperator
                    for subset_delnodes in itertools.combinations(del_nodes, num_nodes):
                        subset_delnodes = set(subset_delnodes)
                        if g.is_minimal_separator(subset_delnodes):  # none of the nodes can be deleted
                            del_nodes = del_nodes - subset_delnodes
                            
                            # in case remaining set of nodes is still a seperator
                            del_nodes = self._check_and_filter_if_seperator(g=g, del_nodes=del_nodes)
                            continue
                    continue

        return del_nodes

    def get_nodes_to_delete(self, g:ig.Graph, protected_nodes:typing.Set[str]=[], keep_graph_connected:bool=True) -> typing.Set[ig.Vertex]:
        """
        returns set of vertices to be deleted. 
        These vertices have the lowest ranking among all vertices which are not protected. 
        If the returned set is empty then all remaining vertices are protected
        """
        ranking = self._get_node_ranking(g=g)
        ranking = np.array(ranking)

        # iterate through unique values of ranking in increasing order
        for min_val in np.sort(np.unique(ranking)):
            del_indices = np.where(ranking == min_val)[0]  # all indices where value is min_val
            del_names = set(g.vs.select(del_indices)['name'])  # set of names of nodes to be deleted
            del_names = del_names - protected_nodes  # remove protected nodenames
            del_nodes = {g.vs.find(name) for name in del_names}
            
            if keep_graph_connected:  # removing all del_nodes would seperate graph
                del_nodes = self._check_and_filter_if_seperator(g=g, del_nodes=del_nodes)
            if len(del_nodes) == 0: continue

            return del_nodes

        return {}  # all nodes are necessary

    def prune_one_iteration(self, g:ig.Graph, protected_nodes:typing.Set[str]=[], keep_graph_connected:bool=True) -> typing.Tuple[ig.Graph, typing.Tuple[str]]:
        g = g.copy()
        del_nodes = self.get_nodes_to_delete(
            g=g,
            protected_nodes=protected_nodes,
            keep_graph_connected=keep_graph_connected
        )
        del_indices = [v.index for v in del_nodes]
        del_names = [v['name'] for v in del_nodes]
        g.delete_vertices(del_indices)
        return g, del_names

    def prune(self, g:ig.Graph, keep_graph_connected:bool=True, protected_concepts:typing.List[str]=['premise', 'conclusion'])-> typing.Tuple[typing.List[ig.Graph], typing.List[typing.List[str]]]:
        """
        input:
            g:                      graph to be pruned
            keep_graph_connected: if True then in each iteration it is checked that no separators (https://en.wikipedia.org/wiki/Vertex_separator) are removed
        returns:
            list of graphs from full graph to maximally greedily pruned graph as well as the names of the nodes that were deleted in each iteration
        """
        g_cp = g.copy()
        self._modify_graph(g=g_cp)

        protected_nodes = self._get_protected_nodes(g_cp, protected_concepts=protected_concepts)  # protected_nodes are nodes in premise or conclusion

        gs = [g]
        names = [()]
        while True:
            g_cp, del_names = self.prune_one_iteration(g=g_cp, protected_nodes=protected_nodes, keep_graph_connected=keep_graph_connected)

            if g_cp.vcount() == gs[-1].vcount():  # cant prune any more
                break
            else:
                gs.append(g.induced_subgraph(vertices=g_cp.vs['tmp_index']))
                names.append(del_names)
            
        return gs, names

class PrunerByPagerank(AbstractPruner):
    """
    prunes by pagerank of nodes
    """
    def _get_node_ranking(self, g: ig.Graph) -> typing.List[float]:
        ranking = g.pagerank(**self.kwargs)
        return ranking

class PrunerBySBERT(AbstractPruner):
    """
    prunes by sbert similarity of nodes to reference sentence
    """
    def __init__(self, datahandler, reference_pc:str='pc'):
        super().__init__(
            datahandler, 
            kwargs={}  # no kwargs for sbert
        )

        self.datahandler.load_sBERT()
        self.reference_pc = reference_pc
        assert self.reference_pc in ['p', 'c', 'pc'], f'give a valid option for {self.reference_pc = }'

    def _modify_graph(self, g: ig.Graph):
        super()._modify_graph(g)
        self._get_node_embeddings(g)
        self._get_reference_embedding(g)

    def _get_reference_sentence(self, g:ig.Graph):
        if self.reference_pc == 'p':
            return g['premise_sentence']
        elif self.reference_pc == 'c':
            return g['conclusion_sentence']
        elif self.reference_pc == 'pc':
            return g['premise_sentence'] + ' ' + g['conclusion_sentence']
        else:
            raise ValueError

    def _get_node_embeddings(self, g:ig.Graph):
        node_names = [name.replace('_', ' ').replace('-', ' ') for name in g.vs['name']]
        for v, n in zip(g.vs, node_names):
            v['embedding'] = list(self.datahandler.sBERT.encode(n, convert_to_tensor=False))

    def _get_reference_embedding(self, g:ig.Graph):
        reference_sentence = self._get_reference_sentence(g)
        reference_embeddings = self.datahandler.sBERT.encode(reference_sentence, convert_to_tensor=False)[None,:]  # np array with dim (1, embedding_dim)
        g['reference_embedding'] = reference_embeddings.T

    def _get_node_ranking(self, g:ig.Graph) -> typing.List[float]:
        cosine_sim = np.dot(g.vs['embedding'], g['reference_embedding'])[:,0].tolist()
        return cosine_sim  # nodes with low value get removed first, so I need to return the similarity and not the weight


class PrunerBy_SBERT_Pagerank(PrunerBySBERT, PrunerByPagerank):
    """
    prunes by sbert and pagerank
    """
    def __init__(self, datahandler, kwargs: dict = {}, reference_pc: str = 'pc', sBERT_uri: str = 'all-mpnet-base-v2', device: str = 'cpu'):
        PrunerBySBERT.__init__(self, datahandler, kwargs, reference_pc, sBERT_uri, device)
    
    def _get_node_ranking(self, g: ig.Graph) -> typing.List[float]:
        ranking1 = PrunerBySBERT._get_node_ranking(self, g)
        ranking2 = PrunerByPagerank._get_node_ranking(self, g)
        ranking = [r1+r2 for r1,r2 in zip(ranking1, ranking2)]
        return ranking
