import time
import argparse
import pprint as pp
import os

import numpy as np
from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=128000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()
    
    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"
    
    np.random.seed(opts.seed)
    
    if opts.filename is None:
        opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde.txt"
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in range(opts.num_samples//opts.batch_size):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes+1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes
            
            idx = 0
            while idx < opts.batch_size:
                nodes_coord = np.random.random([num_nodes, 2])
                solver = TSPSolver.from_data(nodes_coord[:, 0], nodes_coord[:, 1], norm="GEO")  
                solution = solver.solve()
                # Only write instances with valid solutions
                if (np.sort(solution.tour) == np.arange(num_nodes)).all():
                    f.write( " ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord) )
                    f.write( str(" ") + str('output') + str(" ") )
                    f.write( str(" ").join( str(node_idx+1) for node_idx in solution.tour) )
                    f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )
                    f.write( "\n" )
                    idx += 1
            
            assert idx == opts.batch_size
            
        end_time = time.time() - start_time
        
        assert b_idx == opts.num_samples//opts.batch_size - 1
        
    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time/60:.1f}m")
    print(f"Average time: {end_time/opts.num_samples:.1f}s")