#!/bin/bash

task=ExplaGraphs # ExplaGraphs ValNov
datasplit=dev # train dev test
knowledgegraph=ExplaKnow # ExplaKnow CN_withRelatedTo CN_withoutRelatedTo
lookup=sbert_1  # the numer after the underscore is `m`. CCKGs for lookups which are subsets of the chosen lookup can be reconstructed later. For example, if you want to experiment with CCKGs all from m=1 to m=10, you can set `lookup=sbert_10`
r2nl=natural # natural static
start_index=0
num_loaded_graphs=inf

loadgraph_yen_k=1  # k: the number of paths between each pair of concepts. Can be smaller or equal to the value used to construct the shortest path dict. 
only_pc_shortest_paths=False
all_edges=False

pruner=sbert # sbert, pagerank or sbert_pagerank
prunestep=1  # value between 0 and 1

shortestpathordering=False  # True or False. Set to True to have edges point from premise to conclusion. Set to False to have edge-directions from ConceptNet. 

sbert_uri=all-mpnet-base-v2
device=cpu
root=./

yen_num_shortest_paths=1 # k that was used when constructing the shortest path dict # only relevant for `fn_shortest_path_dict`
lookup_spd=sbert_2 # lookup that was used when constructing the shortest path dict # only relevant for `fn_shortest_path_dict`
edge_weight=weight # weight or None # only relevant for `fn_shortest_path_dict`
fn_shortest_path_dict=results/task=${task}/kg=${knowledgegraph}/weight=${edge_weight}/lookup=${lookup_spd}/r2nl=${r2nl}/${datasplit}_k=${yen_num_shortest_paths}.jsonl
fn_prune_data=results/task=${task}/kg=${knowledgegraph}/weight=${edge_weight}/lookup=${lookup}/r2nl=${r2nl}/prunedata/k=${loadgraph_yen_k}/only_pc_shortest_paths=${only_pc_shortest_paths}/all_edges=${all_edges}/${datasplit}.jsonl


python load_shortest_path_dict.py \
    --datahandler $task \
    --lookup $lookup \
    --datasplit $datasplit \
    --knowledgegraph $knowledgegraph \
    --sbert_uri $sbert_uri \
    --device $device \
    --verbose False \
    --root $root \
    --r2nl $r2nl \
    --fn_shortest_path_dict $fn_shortest_path_dict \
    --loadgraph_yen_k $loadgraph_yen_k \
    --only_pc_shortest_paths $only_pc_shortest_paths \
    --all_edges $all_edges \
    --shortestpathordering $shortestpathordering \
    --start_index $start_index \
    --num_loaded_graphs $num_loaded_graphs \
    --fn_prune_data $fn_prune_data \
    --prunestep $prunestep \
