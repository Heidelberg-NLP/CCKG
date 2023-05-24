from functools import partial
from argparse import Namespace
from glob import glob
import json
from pathlib import Path
from tqdm import tqdm

class Args(Namespace):
    def __init__(self,
        task:str='ExplaGraphs',
        r2nl:str='natural',
        kg:str='ExplaKnow',
        lookup:str='sbert',
    ):
        super().__init__()

        self.num_edges = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100]
        self.in_fns = [f'data/tasks/{task}/lookups/{r2nl}_r2nl/{kg}/{lookup}/*/numedges={num}.json' for num in self.num_edges]  # glob can be used
        self.out_fns = [Path(f'data/tasks/{task}/lookups/{r2nl}_r2nl/{kg}/{lookup}/numedges={num}.json') for num in self.num_edges]

def main(args):
    in_fns = [[Path(fn) for fn in glob(in_fn)] for in_fn in args.in_fns]

    for in_fn, out_fn in tqdm(zip(in_fns, args.out_fns), total=len(args.out_fns)):
        lookup = {}
        for fn in in_fn:
            lookup.update(json.load(fn.open('r')))
        
        if out_fn.is_file():
               lookup.update(json.load(out_fn.open('r')))

        json.dump(lookup, open(out_fn, 'w'), indent=2)

if __name__ == "__main__":
    print = partial(print, flush=True)

    args = Args(
        task='ExplaGraphs',
        r2nl='natural',
        kg='ExplaKnow',
        lookup='sbert',
    )

    main(args)

    print("done with main")