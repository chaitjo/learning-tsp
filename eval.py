#!/usr/bin/env python

import math
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch

from train import rollout_groundtruth
from utils import load_model, move_to, get_best
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
from utils.functions import parse_softmax_temperature
from nets.nar_model import NARModel

import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")


def eval_dataset(dataset_path, decode_strategy, width, softmax_temp, opts):
    model, model_args = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dataset = model.problem.make_dataset(
        filename=dataset_path, batch_size=opts.batch_size, num_samples=opts.val_size, 
        neighbors=model_args['neighbors'], knn_strat=model_args['knn_strat'], supervised=True
    )
    
    results = _eval_dataset(model, dataset, decode_strategy, width, softmax_temp, opts, device)

    costs, tours, durations = zip(*results)
    costs, tours, durations = np.array(costs), np.array(tours), np.array(durations)
    gt_tours = dataset.tour_nodes
    gt_costs = rollout_groundtruth(model.problem, dataset, opts).cpu().numpy()
    opt_gap = ((costs/gt_costs - 1) * 100)
    
    results = zip(costs, gt_costs, tours, gt_tours, opt_gap, durations)
    
    print('Validation groundtruth cost: {:.3f} +- {:.3f}'.format(
        gt_costs.mean(), np.std(gt_costs)))
    print('Validation average cost: {:.3f} +- {:.3f}'.format(
        costs.mean(), np.std(costs)))
    print('Validation optimality gap: {:.3f}% +- {:.3f}'.format(
        opt_gap.mean(), np.std(opt_gap)))
    print('Average duration: {:.3f}s +- {:.3f}'.format(
        durations.mean(), np.std(durations)))
    print('Total duration: {}s'.format(np.sum(durations)/opts.batch_size))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    
    results_dir = os.path.join(opts.results_dir, dataset_basename)
    os.makedirs(results_dir, exist_ok=True)
    
    out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
        dataset_basename, model_name,
        decode_strategy,
        width if decode_strategy != 'greedy' else '',
        softmax_temp, opts.offset, opts.offset + len(costs), ext
    ))

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset(results, out_file)

    latex_str = ' & ${:.3f}\pm{:.3f}$ & ${:.3f}\%\pm{:.3f}$ & ${:.3f}$s'.format(
        costs.mean(), np.std(costs), opt_gap.mean(), np.std(opt_gap), np.sum(durations)/opts.batch_size)

    return latex_str


def _eval_dataset(model, dataset, decode_strategy, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp
    )

    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar, ascii=True):
        # Optionally move Tensors to GPU
        nodes, graph = move_to(batch['nodes'], device), move_to(batch['graph'], device)

        start = time.time()
        with torch.no_grad():
            
            if type(model) == NARModel:
                if decode_strategy == 'greedy':
                    _, _, sequences, costs = model.greedy_search(nodes, graph)
                    costs, sequences = costs.cpu().numpy(), sequences.cpu().numpy()
                else:
                    assert decode_strategy == 'bs', "NAR Decoder model only supports greedy/beam search"
                    _, _, sequences, costs = model.beam_search(nodes, graph, beam_size=width)
                
                batch_size = len(costs)
                
            else:
                if decode_strategy in ('sample', 'greedy'):
                    if decode_strategy == 'greedy':
                        assert width == 0, "Do not set width when using greedy"
                        assert opts.batch_size <= opts.max_calc_batch_size, \
                            "batch_size should be smaller than calc batch size"
                        batch_rep = 1
                        iter_rep = 1
                    elif width * opts.batch_size > opts.max_calc_batch_size:
                        assert opts.batch_size == 1
                        assert width % opts.max_calc_batch_size == 0
                        batch_rep = opts.max_calc_batch_size
                        iter_rep = width // opts.max_calc_batch_size
                    else:
                        batch_rep = width
                        iter_rep = 1
                    assert batch_rep > 0
                    # This returns (batch_size, iter_rep shape)
                    sequences, costs = model.sample_many(nodes, graph, batch_rep=batch_rep, iter_rep=iter_rep)
                    batch_size = len(costs)
                    ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
                else:
                    assert decode_strategy == 'bs'

                    cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                        nodes, graph, beam_size=width,
                        compress_mask=opts.compress_mask,
                        max_calc_batch_size=opts.max_calc_batch_size
                    )

                if sequences is None:
                    sequences = [None] * batch_size
                    costs = [math.inf] * batch_size
                else:
                    sequences, costs = get_best(
                        sequences.cpu().numpy(), costs.cpu().numpy(),
                        ids.cpu().numpy() if ids is not None else None,
                        batch_size
                    )
        
        duration = time.time() - start
        
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME in ("tsp", "tspsl"):
                seq = seq.tolist()  # No need to trim as all are same length
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)

            results.append((cost, seq, duration))

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', 
                        help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', 
                        help="Set true to overwrite")
    parser.add_argument("-o", default=None, 
                        help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=12800,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--decode_strategies', type=str, nargs='+',
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--widths', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str,
                        help="Path to model checkpoints directory")
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', 
                        help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', 
                        help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, 
                        help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', 
                        help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for DataLoaders')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"
    
    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    for decode_strategy, width in zip(opts.decode_strategies, opts.widths):
        latex_str = "{}-{}{}".format(opts.model, decode_strategy, width if decode_strategy != 'greedy' else '')
        for dataset_path in opts.datasets:
            latex_str += eval_dataset(dataset_path, decode_strategy, width, opts.softmax_temperature, opts)
            
        with open("results/results_latex.txt", "a") as f:
            f.write(latex_str+"\n")
