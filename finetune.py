#!/usr/bin/env python

import os
import time
import json
import argparse
import pprint as pp
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from nets.attention_model import AttentionModel
from nets.nar_model import NARModel
from nets.encoders.gat_encoder import GraphAttentionEncoder
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.mlp_encoder import MLPEncoder

from reinforce_baselines import *

from problems.tsp.problem_tsp import TSP
from utils import *
from train import *

from tensorboard_logger import Logger as TbLogger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")


def train_batch_ft(model, optimizer, baseline, epoch, 
                batch_id, step, batch, tb_logger, opts):
    # Unwrap baseline
    bat, bl_val = baseline.unwrap_batch(batch)
    
    # Optionally move Tensors to GPU
    x = move_to(bat['nodes'], opts.device)
    graph = move_to(bat['graph'], opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x, graph)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, graph, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    
    # Normalize loss for gradient accumulation
    loss = loss / opts.accumulation_steps

    # Perform backward pass
    loss.backward()
    
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    
    # Perform optimization step after accumulating gradients
    if step % opts.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values_ft(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
                   reinforce_loss, bl_loss, tb_logger, opts)

        
def log_values_ft(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
                  reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('\nepoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost/ft', avg_cost, step)

        tb_logger.log_value('actor_loss/ft', reinforce_loss.item(), step)
        tb_logger.log_value('nll/ft', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm/ft', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped/ft', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss/ft', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm/ft', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped/ft', grad_norms_clipped[1], step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ft_run_name", type=str, default="debug",
                        help="Run name to create logging sub-directory")
    parser.add_argument("--ft_strategy", type=str, default="active",
                        help="Finetuning strategy: active/fixed/random")
    
    parser.add_argument("--problem", type=str, default="tsp")
    parser.add_argument("--min_size", type=int, default=200)
    parser.add_argument("--max_size", type=int, default=200)
    parser.add_argument("--neighbors", type=float, default=0.20)
    parser.add_argument("--knn_strat", type=str, default="percentage")
    parser.add_argument("--data_distribution", type=str, default="random")
    
    parser.add_argument("--val_dataset", type=str, default="data/tsp/tsp200_test_concorde.txt",
                        help="Dataset to evaluate finetuned model on")
    parser.add_argument("--epoch_size", type=int, default=128000)
    parser.add_argument("--val_size", type=int, default=1280)
    parser.add_argument("--rollout_size", type=int, default=1280)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=100)
    
    parser.add_argument('--model', type=str,
                        help="Path to model checkpoints directory")
    parser.add_argument('--baseline', type=str, default="exponential",
                        help="Baseline for finetuning model: none/exponential/rollout")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument("--lr_ft", type=float, default=0.00001)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for DataLoaders')
    parser.add_argument('--no_tensorboard', action='store_true', 
                        help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', 
                        help='Disable progress bar')
    parser.add_argument('--log_step', type=int, default=100, 
                        help='Log info every log_step steps')
    parser.add_argument('--val_every', type=int, default=1, 
                        help='Validate every val_every epochs')
    
    opts = parser.parse_args()
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.ft_run_name = "{}_{}".format(opts.ft_run_name, time.strftime("%Y%m%dT%H%M%S"))
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    # Opts from checkpoint
    args = load_args(os.path.join(opts.model, 'args.json'))
    
    os.makedirs(os.path.join(args["save_dir"], opts.ft_run_name))
    # Save arguments so exact configuration can always be found
    with open(os.path.join(args["save_dir"], opts.ft_run_name, "args-ft.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)
    
    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    
    # Find model file
    if os.path.isfile(opts.model):
        model_filename = opts.model
        path = os.path.dirname(model_filename)
    elif os.path.isdir(opts.model):
        epoch = max(
            int(os.path.splitext(filename)[0].split("-")[1])
            for filename in os.listdir(opts.model)
            if os.path.splitext(filename)[1] == '.pt'
        )
        model_filename = os.path.join(opts.model, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(opts.model)

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # Configure tensorboard
    tb_logger = TbLogger(os.path.join(
        args["log_dir"], "{}_{}-{}".format(args["problem"], args["min_size"], args["max_size"]), args["run_name"], opts.ft_run_name))

    # Figure out what's the problem
    problem = load_problem(args["problem"])

    # Load data from load_path
    load_data = {}
    print('\nLoading data from {}'.format(opts.model))
    load_data = torch_load_cpu(model_filename)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'nar': NARModel,
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    encoder_class = {
        'gnn': GNNEncoder,
        'gat': GraphAttentionEncoder,
        'mlp': MLPEncoder
    }.get(args.get('encoder', 'gnn'), None)
    assert encoder_class is not None, "Unknown encoder: {}".format(encoder_class)
    model = model_class(
        problem=problem,
        embedding_dim=args['embedding_dim'],
        encoder_class=encoder_class,
        n_encode_layers=args['n_encode_layers'],
        aggregation=args['aggregation'],
        aggregation_graph=args['aggregation_graph'],
        normalization=args['normalization'],
        learn_norm=args['learn_norm'],
        track_norm=args['track_norm'],
        gated=args['gated'],
        n_heads=args['n_heads'],
        tanh_clipping=args['tanh_clipping'],
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        checkpoint_encoder=args['checkpoint_encoder'],
        shrink_size=args['shrink_size']
    ).to(opts.device)

    # Compute number of network parameters
    print(model)
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters: ', nb_param)

    # Overwrite model parameters by parameters to load
    print('\nOverwriting model parameters from checkpoint')
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(args["exp_beta"])

    elif opts.baseline == 'critic':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetwork(
                    embedding_dim=args["embedding_dim"],
                    encoder_class=encoder_class,
                    n_encode_layers=args["n_encode_layers"],
                    aggregation=args["aggregation"],
                    normalization=args["normalization"],
                    learn_norm=args["learn_norm"],
                    track_norm=args["track_norm"],
                    gated=args["gated"],
                    n_heads=args["n_heads"]
                )
            ).to(opts.device)
        )

        print(baseline.critic)
        nb_param = 0
        for param in baseline.get_learnable_parameters():
            nb_param += np.prod(list(param.data.size()))
        print('Number of parameters (BL): ', nb_param)

    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)

    else:
        # assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data and opts.baseline == args["baseline"]:
        print('\nOverwriting baseline from checkpoint')
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': args["lr_model"]}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': args["lr_critic"]}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        print('\nOverwriting optimizer from checkpoint')
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Set finetuning learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = opts.lr_ft

    # Load random state
    torch.set_rng_state(load_data['rng_state'])
    if opts.use_cuda:
        torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

    # Dumping of state was done before epoch callback, so do that now (model is loaded)
    baseline.epoch_callback(model, epoch)

    print("Resuming after epoch {}".format(epoch))
    epoch_start = epoch + 1
    step = 0

    # Evaluate on held-out set
    val_dataset = TSP.make_dataset(
        filename=opts.val_dataset, batch_size=opts.batch_size, num_samples=opts.val_size, 
        neighbors=opts.neighbors, knn_strat=opts.knn_strat, supervised=True
    )
    avg_reward, avg_opt_gap = validate(model, val_dataset, problem, opts)
    tb_logger.log_value('val_ft/avg_reward', avg_reward, step)
    tb_logger.log_value('val_ft/opt_gap', avg_opt_gap, step)

    if opts.ft_strategy == "active":
        # Active search: finetune on the test set
        train_dataset = baseline.wrap_dataset(val_dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    elif opts.ft_strategy == "fixed":
        # Fixed finetuning: finetune on a fixed training set
        train_dataset = baseline.wrap_dataset(
            problem.make_dataset(
                min_size=opts.min_size, max_size=opts.max_size, batch_size=opts.batch_size, 
                num_samples=opts.epoch_size, distribution=opts.data_distribution, 
                neighbors=opts.neighbors, knn_strat=opts.knn_strat
            ))
        train_dataloader = DataLoader(
            train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    # Start finetuning loop
    for epoch in range(epoch_start, epoch_start + opts.n_epochs):
        print("\nStart finetuning epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], args["run_name"]))
        start_time = time.time()

        # Put model in train mode!
        model.train()
        optimizer.zero_grad()
        set_decode_type(model, "sampling")

        if opts.ft_strategy == "random":
            # Random finetuning: finetune on new/random samples each epoch
            train_dataset = baseline.wrap_dataset(
                problem.make_dataset(
                    min_size=opts.min_size, max_size=opts.max_size, batch_size=opts.batch_size, 
                    num_samples=opts.epoch_size, distribution=opts.data_distribution, 
                    neighbors=opts.neighbors, knn_strat=opts.knn_strat
                ))
            train_dataloader = DataLoader(
                train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

        for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)):

            train_batch_ft(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )

            step += 1

        epoch_duration = time.time() - start_time
        print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
        
        if epoch % opts.val_every == 0:
            # Evaluate on held-out set
            avg_reward, avg_opt_gap = validate(model, val_dataset, problem, opts)  
            tb_logger.log_value('val_ft/avg_reward', avg_reward, step)
            tb_logger.log_value('val_ft/opt_gap', avg_opt_gap, step)

        baseline.epoch_callback(model, epoch)

    print('\nSaving model and state...')
    torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all()
        },
        os.path.join(args["save_dir"], opts.ft_run_name, 'epoch-{}-ft.pt'.format(epoch))
    )
