import igraph as ig
import queue
from time import time

class YensAlgorithm:
    """
    This class is adapted and modifies from 
    https://gist.github.com/ALenfant/5491853
    """
    def __init__(self, 
                graph_or_pathtograph, 
                debug=False, 
                time=False, 
                avoid_circles=True, 
                load_in_yen=True):
        """
        :param graph_or_pathtograph: either a path to a graph or a graph object
        :param debug: prints debug information
        :param time: prints time information
        :param avoid_circles: prevents paths from containing a node multiple times. Increases computation time, but is necessary for correct implementation of Yen's algorithm
        :param load_in_yen: loads graph from shallow copy instead of restoring edges. This can be faster for large graphs
        """

        self.debug = debug
        self.time = time
        self.avoid_circles = avoid_circles 
        self.load_in_yen = load_in_yen 

        if isinstance(graph_or_pathtograph, ig.Graph):
            self.graph = graph_or_pathtograph.copy()
        elif isinstance(graph_or_pathtograph, str):
            self.graph = ig.read(graph_or_pathtograph, format="pickle")
        else:
            print(f'{graph_or_pathtograph = }')
            assert False

        if self.load_in_yen:
            self.graph_backup = self.graph.copy()

    def load_graph(self):
        return self.graph_backup.copy()

    def path_cost(self, path, weights=None):
        pathcost = 0
        for i in range(len(path)):
            if i > 0:
                edge=self.graph.es.find(_source=path[i-1], _target=path[i])
                if weights != None:
                    pathcost += edge[weights]
                else:
                    #just count the number of edges
                    pathcost += 1
        return pathcost

    def __call__(self, source, target:list, num_k:int, weights:str='weight', return_time:bool=False):
        """
        Implementation of Yen's algorithm.
        :param source: source node
        :param target: target nodes
        :param num_k: number of shortest paths to find
        :param weights: edge relation to use as edge weights
        :param return_time: return time information
        """
        t0 = time()
        #Shortest path from the source to the target
        A = [self.graph.get_shortest_paths(source, to=target, weights=weights, output="vpath")[0]]
        if len(A[0]) == 0:
            print(f'no path between {source=} and {target=} exists.')
            if return_time:
                return A, [], []
            return A, []
        A_costs = [self.path_cost(A[0], weights)]
        A_times = [time()-t0]

        #Initialize the heap to store the potential kth shortest path
        B = queue.PriorityQueue()

        # initialize check whether all paths are found
        found_all_paths = False

        for k in range(1, num_k):
            t1 = time()
            #The spur node ranges from the first node to the next to last node in the shortest path
            for i in range(len(A[k-1])-1):
                if self.debug: print(f'{k = }, {i = }', flush=True)
                #Spur node is retrieved from the previous k-shortest path, k − 1
                spurNode = A[k-1][i]
                if self.debug: print(f'{spurNode = }', flush=True)
                #The sequence of nodes from the source to the spur node of the previous k-shortest path
                rootPath = A[k-1][:i]
                if self.debug: print(f'{rootPath = }', flush=True)
                #We store the removed edges
                if self.time: t_remove_edges = time()
                if not self.load_in_yen:
                    removed_edges = []

                for path in A:
                    if self.debug: print(f'{path = }', flush=True)
                    if len(path) - 1 > i and rootPath == path[:i]:
                        #Remove the links that are part of the previous shortest paths which share the same root path
                        edge = self.graph.es.select(_source=path[i], _target=path[i+1])
                        if self.load_in_yen:
                            self.graph.delete_edges(edge)
                        else:
                            if len(edge) == 0:
                                continue #edge already deleted
                            edge = edge[0]
                            removed_edges.append((path[i], path[i+1], edge.attributes()))
                            edge.delete()
                if self.avoid_circles:
                    """
                    The following part is written by Moritz Plenz. I added it, because it was missing in the original implementation. It corresponds to the following lines in the Pseudocode on Wikipedia:
                    ``
                    for each node rootPathNode in rootPath except spurNode:
                        remove rootPathNode from Graph;
                    ``
                    This step is necessary to avoid paths which visit a node more than once. 
                    """
                    for node in rootPath:
                        # instead of deleting nodes we delete all edges going to the node
                        # this way the node ids stay the same
                        if self.load_in_yen:
                            edge = self.graph.es.select(_target=node)
                            self.graph.delete_edges(edge)
                        else:
                            for nei in self.graph.neighbors(node): 
                                edge = self.graph.es.select(_source=nei, _target=node)
                                if len(edge) == 0:
                                    continue  # edge already deleted
                                edge = edge[0]
                                removed_edges.append((node, nei, edge.attributes()))
                                edge.delete()

                if self.time: print(f'{k=}, {i=}; removing edges took {time()-t_remove_edges} seconds', flush=True)
                #Calculate the spur path from the spur node to the sink
                if self.time: t_get_shortest_path = time()
                spurPath = self.graph.get_shortest_paths(spurNode, to=target, weights=weights, output="vpath")[0]
                

                if self.time: print(f'{k=}, {i=}; finding shortest path took {time()-t_get_shortest_path} seconds', flush=True)
                if self.time: t_restore_edges = time()
                #Add back the edges that were removed from the graph
                if self.load_in_yen:
                    self.graph = self.load_graph()
                else:
                    for removed_edge in removed_edges:
                        node_start, node_end, cost = removed_edge
                        self.graph.add_edge(node_start, node_end)
                        edge = self.graph.es.select(_source=node_start, _target=node_end)[0]
                        edge.update_attributes(cost)
                if self.time: print(f'{k=}, {i=}; restoring edges took {time()-t_restore_edges} seconds', flush=True)
                if len(spurPath) > 0:
                    #Entire path is made up of the root path and spur path
                    totalPath = rootPath + spurPath
                    if self.debug:
                        print(f'{rootPath = }', flush=True)
                        print(f'{spurPath = }', flush=True)
                        print(f'{totalPath = }', flush=True)
                    totalPathCost = self.path_cost(totalPath, weights)
                    #Add the potential k-shortest path to the heap
                    B.put((totalPathCost, totalPath))

            #Sort the potential k-shortest paths by cost
            #B is already sorted
            #Add the lowest cost path becomes the k-shortest path.
            while True:
                if B.empty():
                    print(f'only {k=} different paths. ({num_k=} shortest paths were required.)')
                    found_all_paths = True
                    break
                cost_, path_ = B.get()
                if path_ not in A:
                    #We found a new path to add
                    A.append(path_)
                    A_costs.append(cost_)
                    A_times.append(time()-t1)
                    break
            if self.time: print(f'{k=} took {time()-t1} seconds', flush=True)
            if found_all_paths:
                break

        if self.time: print(f'algorithm took {time()-t0} seconds', flush=True)
        if return_time:
            return A, A_costs, A_times  
        return A, A_costs

    def call_dijkstra(self, source, targets:list, weights:str='weight', return_time:bool=False):
        """
        Implementation of Dijsktra's algorithm that is compatible with the implementation of Yen's algorithm.
        :param source: source node
        :param target: target nodes
        :param weights: edge relation to use as edge weights
        :param return_time: return time information
        """
        assert not self.graph.is_directed()
        t0 = time()
        #Shortest path from the source to the target
        all_shortest_paths = self.graph.get_all_shortest_paths(source, to=targets, weights=weights)

        all_shortest_paths = [x for x in all_shortest_paths if x!=[]]  # filter out empty lists
        result_dict_paths = {target_index:
                    [sp for sp in all_shortest_paths if sp[-1]==target_index]
                    for target_index in targets}

        total_len = 0
        for key in result_dict_paths.keys():
            tmp_len = len(result_dict_paths[key])
            total_len += tmp_len
        assert total_len == len(all_shortest_paths), f'{total_len} {len(all_shortest_paths)}'


        result_dict_costs = {key:
                    [self.path_cost(sp, weights) for sp in result_dict_paths[key]]
                    for key in result_dict_paths.keys()}

        if return_time:
            t_per_path = (time()-t0) / (total_len+1e-10)
            result_dict_times = {key:
                        [t_per_path]*len(result_dict_paths[key])
                        for key in result_dict_paths.keys()}

        if self.time: print(f'algorithm took {time()-t0} seconds', flush=True)
        if return_time:
            return result_dict_paths, result_dict_costs, result_dict_times  
        return result_dict_paths, result_dict_costs

