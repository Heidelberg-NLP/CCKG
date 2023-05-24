#!/bin/bash

task=ExplaGraphs # ExplaGraphs ValNov
datasplit=dev # train dev test
knowledgegraph=ExplaKnow # ExplaKnow CN_withRelatedTo CN_withoutRelatedTo
lookup=sbert_2  # the numer after the underscore is `m`. CCKGs for lookups which are subsets of the chosen lookup can be reconstructed later. For example, if you want to experiment with CCKGs all from m=1 to m=10, you can set `lookup=sbert_10`
r2nl=natural # natural static
start_index=0
end_index=inf
yen_edge_weight=weight # weight or None
yen_num_shortest_paths=1  # k: the number of paths between each pair of concepts. CCKGs for smaller values can be reconstructed later. For example, if you want to experiment with CCKGs from k=1 to k=10, you can set `yen_num_shortest_paths=10`
edgeweight_to_pc=pc  # whether the edge weight is computed to the premise and conclusion together

sbert_uri=all-mpnet-base-v2
device=cpu
root=./ 

out_dir=results/task=${task}/kg=${knowledgegraph}/weight=${yen_edge_weight}/lookup=${lookup}/r2nl=${r2nl}
out_fn=${datasplit}_k=${yen_num_shortest_paths}_index=${start_index}  # relative to out_dir

python get_shortest_path_dict.py \
    --datahandler $task \
    --lookup $lookup \
    --datasplit $datasplit \
    --knowledgegraph $knowledgegraph \
    --sbert_uri $sbert_uri \
    --device $device \
    --verbose False \
    --root $root \
    --r2nl $r2nl \
    --out_dir $out_dir \
    --out_fn $out_fn \
    --start_index $start_index \
    --end_index $end_index \
    --edgeweight_to_pc $edgeweight_to_pc \
    --yen_edge_weight $yen_edge_weight \
    --yen_num_shortest_paths $yen_num_shortest_paths \
    --yen_avoid_circles True 

