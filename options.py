import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters for training learning-driven solvers for TSP")

    # Data
    parser.add_argument('--problem', default='tsp', 
                        help="The problem to solve, default 'tsp', use 'tspsl' if Supervised Learning")
    parser.add_argument('--min_size', type=int, default=20, 
                        help="The minimum size of the problem graph")
    parser.add_argument('--max_size', type=int, default=50, 
                        help="The maximum size of the problem graph")
    parser.add_argument('--neighbors', type=float, default=20, 
                        help="The k-nearest neighbors for graph sparsification")
    parser.add_argument('--knn_strat', type=str, default=None, 
                        help="Strategy for k-nearest neighbors (None/'percentage')")
    parser.add_argument('--n_epochs', type=int, default=100, 
                        help='The number of epochs to train')
    parser.add_argument('--epoch_size', type=int, default=1000000, 
                        help='Number of instances per epoch during training')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Number of instances per batch during training')
    parser.add_argument('--accumulation_steps', type=int, default=1, 
                        help='Gradient accumulation step during training '
                             '(effective batch_size = batch_size * accumulation_steps)')
    parser.add_argument('--train_dataset', type=str, default=None, 
                        help='Dataset file to use for training (SL only)')
    parser.add_argument('--val_datasets', type=str, nargs='+', default=None, 
                        help='Dataset files to use for validation')
    parser.add_argument('--val_size', type=int, default=1000, 
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--rollout_size', type=int, default=10000, 
                        help='Number of instances used for updating rollout baseline')
    
    # Model/GNN Encoder
    parser.add_argument('--model', default='attention', 
                        help="Model: 'attention'/'nar'")
    parser.add_argument('--encoder', default='gnn', 
                        help="Graph encoder: 'gat'/'gnn'/'mlp'")
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, 
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--aggregation', default='max', 
                        help="Neighborhood aggregation function: 'sum'/'mean'/'max'")
    parser.add_argument('--aggregation_graph', default='mean', 
                        help="Graph embedding aggregation function: 'sum'/'mean'/'max'")
    parser.add_argument('--normalization', default='layer', 
                        help="Normalization type: 'batch'/'layer'/None")
    parser.add_argument('--learn_norm', action='store_true', 
                        help="Enable learnable affine transformation during normalization")
    parser.add_argument('--track_norm', action='store_true',
                        help="Enable tracking batch statistics during normalization")
    parser.add_argument('--gated', action='store_true', 
                        help="Enable edge gating during neighborhood aggregation")
    parser.add_argument('--n_heads', type=int, default=8, 
                        help="Number of attention heads")
    parser.add_argument('--tanh_clipping', type=float, default=10., 
                        help='Clip the parameters to within +- this value using tanh. Set to 0 to not do clipping.')

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, 
                        help="Set the learning rate for the actor network, i.e. the main model")
    parser.add_argument('--lr_critic', type=float, default=1e-4, 
                        help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, 
                        help='Learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                        help='Maximum L2 norm for gradient clipping (0 to disable clipping)')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem')
    parser.add_argument('--eval_only', action='store_true', 
                        help='Set this value to only evaluate model')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')

    # Misc
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for DataLoaders')
    parser.add_argument('--log_step', type=int, default=100, 
                        help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', 
                        help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', 
                        help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', 
                        help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', 
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', 
                        help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', 
                        help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', 
                        help='Disable progress bar')
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}-{}".format(opts.problem, opts.min_size, opts.max_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    
    return opts
