import igraph as ig 
import numpy as np 
import json
import itertools
import typing as t

from methods import data_handler as dh

def str_to_bool(s:str):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError

def add_args(parser):
    add_args_loader(parser)
    add_args_pruner(parser)

def add_args_loader(parser):
    parser.add_argument(
        '--fn_shortest_path_dict',
        type=str,
        required=True,
        help='path to the shortest path dict to load'
    )
    parser.add_argument(
        '--loadgraph_yen_k',
        type=int,
        default=1,
        help='number of shortest paths to load. Can not be larger than the number of shortest paths used to compute the shortest path dict (no error will be thrown -- it will just load less paths without a warning).'
    )
    parser.add_argument(
        '--only_pc_shortest_paths',
        type=str_to_bool,
        default=False,
        help='Set to `True` to only consider shortest paths that connect a premise-concept to a conclusion-concept. Set to `False` to consider all shortest paths. In the paper we used `False`.',
    )
    parser.add_argument(
        '--all_edges',
        type=str_to_bool,
        default=False,
        help='Set to `False` to consider only edges that were part of at least one shortest path. This is the setting we used in the paper. When set to `True`, then the subgraph of the underlying knowledge graph which spans all concepts in all shortest paths is used instead. I.e. the graph contains the same cocnepts as when this parameter is set to `False`, but potentially contains more edges.',
    )
    parser.add_argument(
        '--shortestpathordering',
        type=str_to_bool,
        default=False,
        help='True for edges pointing from premise to conclusion. False for edge direction as in CN. Edges which do not have a clear direction (e.g. edges with a `DistinctFrom` relation) get a random direction.',
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='start index of the shortest path dict to load'
    )
    parser.add_argument(
        '--num_loaded_graphs',
        type=float,
        default=float('inf'),
        help='number of graphs to load. Needs to be concerted to int before it is passed to `load_shortest_path_dict`'
    )

def add_args_pruner(parser):
    parser.add_argument(
        '--fn_prune_data',
        type=str,
        required=True,
        help='path to the pruned data to load'
    )
    parser.add_argument(
        '--prunestep',
        type=float,
        default=1,
        help='Amount of pruning on a scale from 0 to 1. 0 means no pruning, 1 means all prunable concepts are pruned.'
    )

class ShortestPathDictToIGraph:
    """
    class to load ShortestPathDict as iGraph. The ShortestPathDicts are in the format:
        {
            'index': `index`,
            'times_per_task': {...},
            'shortest_paths': {
                '(sourcenode_index, targetnode_index)': {
                    'paths': [node_path_1, node_path_2, ..., node_path_k],
                    'paths_edges': [edge_path_1, edge_path_2, ..., edge_path_k],  # THIS IS OPTIONAL
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
        (relation_with_lowest_weight_on_edge, all_relations_on_edge, weight_of_edge, natural_language_relation, weight_to_pc,  source_of_edge_in_KG, target_of_edge_in_KG)
    """
    def __init__(self, datahandler:dh.AbstractDataHandler):
        self.datahandler = datahandler
    
    def _get_skip_paths_keys(self, graph_dict:dict, only_pc_shortest_paths:bool, concepts_premise:set, concepts_conclusion:set)->set:
        """
        :param graph_dict: dict in the format described above
        :param only_pc_shortest_paths: if True, only paths which connect premise and conclusion are considered
        :param concepts_premise: set of premise concepts
        :param concepts_conclusion: set of conclusion concepts
        :return: set of keys of paths which should be skipped
        """
        skip_path_keys = set()
        
        all_starts_targets = set()

        if only_pc_shortest_paths:
            # skip path if it does not connect premise and conclusion or if it has a start/end which is not in concepts_premise or concepts_conclusion
            for item in graph_dict['shortest_paths'].items():
                if len(item[1]['paths']) == 0: continue
                tmp_path = item[1]['paths'][0]
                if len(tmp_path) == 0: 
                    if len(item[1]['paths']) >= 2:
                        tmp_path = item[1]['paths'][1]
                    else:
                        continue
                    if len(tmp_path) == 0:
                        continue
                start_node = tmp_path[0][1]
                target_node = tmp_path[-1][1]

                all_starts_targets.add(start_node)
                all_starts_targets.add(target_node)

                sp = start_node in concepts_premise
                sc = start_node in concepts_conclusion
                tp = target_node in concepts_premise
                tc = target_node in concepts_conclusion

                if not((sp and tc) or (sc and tp)):
                    skip_path_keys.add(item[0])
        else:
            all_concepts = concepts_premise.union(concepts_conclusion)
            # skip path if it has a start/end which is not in concepts_premise or concepts_conclusion
            for item in graph_dict['shortest_paths'].items():
                if len(item[1]['paths']) == 0: continue
                tmp_path = item[1]['paths'][0]
                if len(tmp_path) == 0:
                    if len(item[1]['paths']) >= 2:
                        tmp_path = item[1]['paths'][1]
                    else:                
                        continue
                    if len(tmp_path) == 0:
                        continue
                start_node = tmp_path[0][1]
                target_node = tmp_path[-1][1]

                all_starts_targets.add(start_node)
                all_starts_targets.add(target_node)

                s = start_node in all_concepts
                t = target_node in all_concepts
                
                if not(s and t):
                    skip_path_keys.add(item[0])

        return skip_path_keys

    def _get_nodes_edges_costs_alledges(self, graph_dict, skip_paths, yen_k):
        tmp_nodes = {}
        tmp_edges = set()
        tmp_costs = {}  # costs of all shortest paths
        # get all nodes and tmp_costs
        for item in graph_dict['shortest_paths'].items():  # iterate over paths 
            if item[0] in skip_paths: continue
            for path in item[1]['paths'][:yen_k]:
                # get nodes
                for node in path:
                    tmp_nodes[node[0]] = node[1]  # key is index, value is name

            tmp_costs[item[0]] = item[1]['costs']
        if len(tmp_nodes) == 0:
            return tmp_nodes, tmp_edges, tmp_costs
        # get subgraph from conceptnet
        v_tmp_indices = [self.datahandler.knowledgegraph.vs.find(name=n).index for n in tmp_nodes.values()]
        tmp_g = self.datahandler.knowledgegraph.induced_subgraph(v_tmp_indices)
        self.datahandler.add_edge_attribute_costs_from_edge_embedding(index=graph_dict['index'], g=tmp_g, to_pc='pc')

        for e in tmp_g.es:
            source = tmp_g.vs.find(e['weight_source'])['name']
            target = tmp_g.vs.find(e['weight_target'])['name']
            relation = e['weight_relation']
            weight = e['weight']

            tmp_edges.add((source, target, relation, weight))

        return tmp_nodes, tmp_edges, tmp_costs

    def _get_nodes_edges_costs_onlypathedges(self, graph_dict, skip_paths, yen_k):
        tmp_nodes = {}
        tmp_edges = set()
        tmp_edges_names_only = set()
        tmp_costs = {}  # costs of all shortest paths
        for item in graph_dict['shortest_paths'].items():  # iterate over paths 
            if item[0] in skip_paths: continue
            for path in item[1]['paths'][:yen_k]:
                # get nodes
                for node in path:
                    tmp_nodes[node[0]] = node[1]  # key is index, value is name

            tmp_costs[item[0]] = item[1]['costs']
        
        if len(tmp_nodes) == 0:
            return tmp_nodes, tmp_edges, tmp_costs

        graph_was_weighted = False  # initialize as false. If the graph was weighted it will be switched to True below. 
        for item in graph_dict['shortest_paths'].items():  # iterate over paths 
            if item[0] in skip_paths: continue
            if 'paths_edges' in item[1].keys():  # graph was weighted
                graph_was_weighted = True
                for edge_path in item[1]['paths_edges'][:yen_k]:
                    for edge in edge_path:
                        source = tmp_nodes[edge[5]]
                        target = tmp_nodes[edge[6]]
                        relation = edge[0]
                        weight = edge[2]

                        tmp_edges.add((source, target, relation, weight))
            else:  # graph was not weighted
                for path in item[1]['paths'][:yen_k]:
                    for n1, n2 in zip(path[:-1], path[1:]):
                        tmp_edges_names_only.add(frozenset((n1[1], n2[1])))
       
        if not graph_was_weighted: # graph was unweighted
            # get spanning subgraph from conceptnet
            v_tmp_indices = [self.datahandler.knowledgegraph.vs.find(name=n).index for n in tmp_nodes.values()]
            tmp_g = self.datahandler.knowledgegraph.induced_subgraph(v_tmp_indices)
            self.datahandler.add_edge_attribute_costs_from_edge_embedding(index=graph_dict['index'], g=tmp_g, to_pc='pc')

            for e in tmp_g.es:
                source = tmp_g.vs.find(e['weight_source'])['name']
                target = tmp_g.vs.find(e['weight_target'])['name']
                if frozenset((source, target)) in tmp_edges_names_only:  # filter our relevant edges
                    relation = e['weight_relation']
                    weight = e['weight']

                    tmp_edges.add((source, target, relation, weight))

            # tmp_costs[item[0]] = item[1]['costs']

        return tmp_nodes, tmp_edges, tmp_costs

    def _get_nodes_edges_costs_shortestpathordering(self, graph_dict, skip_paths, yen_k, concepts_premise, concepts_conclusion):
        """
        Edges are directed such that they point from premise to conclusion. 
        """
        tmp_nodes = {}
        tmp_edge_direction = {} # key is frozenset of nodenames. Value is tuple (first entry is source, second entry is target)
        tmp_edges = set()
        tmp_costs = {}  # costs of all shortest paths
        for item in graph_dict['shortest_paths'].items():  # iterate over paths 
            if item[0] in skip_paths: continue
            for path in item[1]['paths'][:yen_k]:
                if len(path)==0:
                    continue
                # flip path if it was from conclusion to premise
                if path[0][1] in concepts_premise:
                    if path[-1][1] in concepts_conclusion:
                        pass
                    else:
                        assert path[0][1] in concepts_conclusion, path
                        assert path[-1][1] in concepts_premise, path
                        path = path[::-1]
                elif path[-1][1] in concepts_premise:
                    if path[0][1] in concepts_conclusion:
                        path = path[::-1]
                    else:
                        assert path[-1][1] in concepts_conclusion, path
                        assert path[0][1] in concepts_premise, path
                        pass
                else:
                    raise ValueError(path)
                # get nodes
                for node in path:
                    tmp_nodes[node[0]] = node[1]  # key is index, value is name
                
                # get edge order
                for source, target in zip(path[:-1], path[1:]):
                    s = source[1]
                    t = target[1]
                    tmp_edge_direction[frozenset({s,t})] = (s,t)
                

            if 'paths_edges' in item[1].keys():  # graph was weighted
                for edge_path in item[1]['paths_edges'][:yen_k]:
                    for edge in edge_path:
                        KG_source = tmp_nodes[edge[5]]
                        KG_target = tmp_nodes[edge[6]]
                        source, target = tmp_edge_direction[frozenset({KG_source, KG_target})]
                        relation = edge[0]
                        weight = edge[2]
                        
                        relation_index = edge[1].index(relation)
                        nl_relation = edge[3][relation_index]

                        tmp_edges.add((source, target, relation, weight, KG_source, KG_target, nl_relation))
            else:  # graph was not weighted
                raise NotImplementedError

            tmp_costs[item[0]] = item[1]['costs']
        return tmp_nodes, tmp_edges, tmp_costs

    def load_shortest_path_dict(self, fn:str, yen_k:int, num_graphs:int, only_pc_shortest_paths:bool, 
    all_edges:bool, shortestpathordering:bool, start_index:int)->t.List[ig.Graph]:
        """
        args:
            fn: filename of dict with shortest paths
            yen_k: number of shortest paths used between each pair of concepts
            num_graphs: number of graphs that get loaded. can also be a float, e.g. float('inf')
            only_pc_shortest_paths: shortest paths in shortest_path_dict that connect premise-premise or conlcusion-conclusion concepts get skipped. 
            all_edges: False only chooses edges which were in the paths. True picks all edges linking nodes in the subgraph together, even if they were not part of the shortest paths. 
            shortestpathordering: True for edges pointing from premise to conclusion. False for edge direction as in CN. Edges which do not have a clear direction (e.g. edges with a `DistinctFrom` relation) get a random direction.
        returns:
            list of igraph.Graph objects
        """

        # graph_dicts = []
        # # load dictionaries
        # with open(fn, 'r') as f:
        #     lines = f.readlines()
        # lines = lines[start_index:]
        # if len(lines) > num_graphs:  # still works if num_graphs is inf
        #     lines = lines[:int(num_graphs)]

        # graph_dicts = []
        # lines = [l for i,l in enumerate(lines) if i in i >= start_index and i < start_index+num_graphs]
    
        # graph_dicts = [json.loads(line) for line in lines]

        # del lines  # free some memory

        graph_dicts = [l for i,l in enumerate(open(fn, 'r')) if i >= start_index and i < start_index+num_graphs]
        graph_dicts = [json.loads(line) for line in graph_dicts]

        assert ([d['index'] for d in graph_dicts] == np.arange(start_index, start_index+len(graph_dicts))).all(), str([d['index'] for d in graph_dicts])

        graphs = []
        # create graphs
        for graph_dict in graph_dicts:  # iterate over graphs
            premise, conclusion = self.datahandler.get_premise_and_conclusion(graph_dict['index'])

            concepts_premise = set(self.datahandler.get_concepts_from_lookupdict(key=premise.lower()))
            concepts_conclusion = set(self.datahandler.get_concepts_from_lookupdict(key=conclusion.lower()))

            skip_paths = self._get_skip_paths_keys(graph_dict=graph_dict, only_pc_shortest_paths=only_pc_shortest_paths, concepts_premise=concepts_premise, concepts_conclusion=concepts_conclusion)

            if shortestpathordering:
                assert all_edges == False, all_edges
                assert only_pc_shortest_paths == True, only_pc_shortest_paths  # removing this probably will not throw an error. However, edges linking premise with premise or conclusion with conclusion will have a random direction. 
                tmp_nodes, tmp_edges, tmp_costs = self._get_nodes_edges_costs_shortestpathordering(
                    graph_dict=graph_dict, 
                    skip_paths=skip_paths, 
                    yen_k=yen_k,
                    concepts_premise=concepts_premise,
                    concepts_conclusion=concepts_conclusion
                )
            else:
                if all_edges:
                    tmp_nodes, tmp_edges, tmp_costs = self._get_nodes_edges_costs_alledges(
                        graph_dict=graph_dict, 
                        skip_paths=skip_paths, 
                        yen_k=yen_k
                    )
                else:
                    tmp_nodes, tmp_edges, tmp_costs = self._get_nodes_edges_costs_onlypathedges(
                        graph_dict=graph_dict, 
                        skip_paths=skip_paths, 
                        yen_k=yen_k
                    )
            # initialize graph and nodes
            tmp_g = ig.Graph(
                directed=True,
                n=len(tmp_nodes),
                vertex_attrs={
                    'name':list(tmp_nodes.values())
                },
                graph_attrs={
                    'index': graph_dict['index'],
                    'shortest_path_costs': tmp_costs,
                    'premise_sentence': premise,
                    'conclusion_sentence': conclusion
                }
            )

            # add node attributes
            concepts_intersection = concepts_premise.intersection(concepts_conclusion)
            concepts_union = concepts_premise.union(concepts_conclusion)

            tmp_g.vs['is_in_premise'] = [(v['name'] in concepts_premise) for v in tmp_g.vs]
            tmp_g.vs['is_in_conclusion'] = [(v['name'] in concepts_conclusion) for v in tmp_g.vs]
            tmp_g.vs['is_in_premise_intersection_conclusion'] = [(v['name'] in concepts_intersection) for v in tmp_g.vs]
            tmp_g.vs['is_in_premise_union_conclusion'] = [(v['name'] in concepts_union) for v in tmp_g.vs]

            # add edges with attributes
            for edge in tmp_edges:
                if shortestpathordering==True:
                    assert len(edge)==7, edge
                    tmp_g.add_edge(edge[0], edge[1], relation=edge[2], weight=edge[3], KG_source=edge[4], KG_target=edge[5], nl_relation=edge[6])
                elif shortestpathordering==False:
                    assert len(edge)==4, edge
                    tmp_g.add_edge(edge[0], edge[1], relation=edge[2], weight=edge[3])
                else:
                    raise ValueError((shortestpathordering, type(shortestpathordering)))

            # assert tmp_g.is_simple(), tmp_g  # can be non-simple if one node is connected to another node by two relations and both appear in the shortest paths

            graphs.append(tmp_g)
            
        return graphs

class PruneListOfGraphs:
    """
    class to prune iGraphs given the data for which concepts should pruned. (I.e. this class does not do the pruning -- it only applies the result of the pruning in hindsight when loading pruned graphs.)
    """
    def __init__(self, graphs:t.List[ig.Graph], fn_prune:str):
        """
        :param graphs: list of igraph.Graph objects or None
        :param fn_prune: path to file containing the prune data or None
        """
        if graphs is None:
            self.graphs = None
        else:
            self.set_graphs(graphs)
        
        if fn_prune is None:
            self.prune_data = None
        else:
            self.set_prune_data(fn_prune)

        self.assert_graphs_match_prunedata()

    def set_graphs(self, graphs):
        self.graphs = graphs

    def set_prune_data(self, fn_prune:str) -> t.List[t.List[str]]:
        with open(fn_prune, 'r') as f:
            lines = f.readlines()

        self.prune_data = [json.loads(l) for l in lines]

    def assert_graphs_match_prunedata(self):
        if self.graphs is None:
            print('no graphs to check. call `set_graphs` first')
            return None

        if self.prune_data is None:
            print('no prune data to check. call `set_prune_data` first')
            return None

        for g in self.graphs:
            p = self.prune_data[g['index']]
            all_pruned_names = list(itertools.chain(*p))
            all_names = g.vs['name']
            assert len(all_pruned_names) == len(set(all_pruned_names))
            assert len(all_names) == len(set(all_names))
            all_pruned_names = set(all_pruned_names)
            all_names = set(all_names)
            assert all_pruned_names.issubset(all_names), (g['index'], all_pruned_names, all_names, all_pruned_names - all_names, all_pruned_names.intersection(all_names))

    def prune(self)->t.List[t.List[ig.Graph]]:
        """
        prunes graphs completely according to prune data
        :return: list of list of pruned graphs. The outer list corresponds to different original graphs. The inner list corresponds to different amounts of pruning.
        """
        all_pruned_graphs = []
        for graph in self.graphs:
            prune = self.prune_data[graph['index']]
            tmp_g = graph.copy()
            tmp_gs = []
            for p_names in prune:
                p_ids = [tmp_g.vs.find(name).index for name in p_names]
                tmp_g.delete_vertices(p_ids)
                tmp_gs.append(tmp_g.copy())
            all_pruned_graphs.append(tmp_gs)
        return all_pruned_graphs
    
    def prune_prunestep(self, prunestep:float=1.0)->t.List[ig.Graph]:
        """
        prunes graphs partially according to prune data
        :param prunestep: float between 0 and 1
        :return: list of pruned graphs
        """
        pruned_graphs = []
        for graph in self.graphs:
            tmp_g = graph.copy()
            
            prune = self.prune_data[graph['index']]
            if prunestep == 1:
                prune = prune
            else:
                prune = prune[:int(len(prune) * prunestep)]

            p_names = list(itertools.chain.from_iterable(prune))

            p_ids = [tmp_g.vs.find(name).index for name in p_names]
            tmp_g.delete_vertices(p_ids)
            pruned_graphs.append(tmp_g)
        return pruned_graphs
