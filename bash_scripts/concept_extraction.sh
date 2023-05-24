#!/bin/bash

task=ExplaGraphs # ExplaGraphs ValNov
datasplit=dev # train dev test
knowledgegraph=ExplaKnow # ExplaKnow CN_withRelatedTo CN_withoutRelatedTo
r2nl=natural # natural static

sbert_uri=all-mpnet-base-v2
device=cuda
root=./ 
num_edges="1 2 3 4 5 6 7 8 9 10"

batch_size=200

out_dir=data/tasks/${task}/lookups/${r2nl}_r2nl/${knowledgegraph}/sbert/${datasplit}

python concept_extraction.py \
    --datahandler $task \
    --lookup NONE \
    --datasplit $datasplit \
    --knowledgegraph $knowledgegraph \
    --sbert_uri $sbert_uri \
    --device $device \
    --verbose False \
    --root $root \
    --r2nl $r2nl \
    --num_edges $num_edges \
    --out_dir $out_dir \
    --batch_size $batch_size
