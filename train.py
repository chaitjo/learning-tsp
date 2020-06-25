import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel

from utils.log_utils import log_values, log_values_sl
from utils.data_utils import BatchedRandomSampler
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def validate(model, dataset, problem, opts):
    # Validate
    print(f'\nValidating on {dataset.size} samples from {dataset.filename}...')
    cost = rollout(model, dataset, opts)
    gt_cost = rollout_groundtruth(problem, dataset, opts)
    opt_gap = ((cost/gt_cost - 1) * 100)
    
    print('Validation groundtruth cost: {:.3f} +- {:.3f}'.format(
        gt_cost.mean(), torch.std(gt_cost)))
    print('Validation average cost: {:.3f} +- {:.3f}'.format(
        cost.mean(), torch.std(cost)))
    print('Validation optimality gap: {:.3f}% +- {:.3f}'.format(
        opt_gap.mean(), torch.std(opt_gap)))

    return cost.mean(), opt_gap.mean()


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat['nodes'], opts.device), move_to(bat['graph'], opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(
            DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers), 
            disable=opts.no_progress_bar, ascii=True
        )
    ], 0)


def rollout_groundtruth(problem, dataset, opts):
    return torch.cat([
        problem.get_costs(bat['nodes'], bat['tour_nodes'])[0]
        for bat in DataLoader(
            dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_datasets, problem, tb_logger, opts):
    print("\nStart train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    train_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            min_size=opts.min_size, max_size=opts.max_size, batch_size=opts.batch_size, 
            num_samples=opts.epoch_size, distribution=opts.data_distribution, 
            neighbors=opts.neighbors, knn_strat=opts.knn_strat
        ))
    train_dataloader = DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    # Put model in train mode!
    model.train()
    optimizer.zero_grad()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)):

        train_batch(
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
    
    lr_scheduler.step(epoch)

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    for val_idx, val_dataset in enumerate(val_datasets):
        avg_reward, avg_opt_gap = validate(model, val_dataset, problem, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val{}/avg_reward'.format(val_idx+1), avg_reward, step)
            tb_logger.log_value('val{}/opt_gap'.format(val_idx+1), avg_opt_gap, step)

    baseline.epoch_callback(model, epoch)


def train_batch(model, optimizer, baseline, epoch, 
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
        log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
                   reinforce_loss, bl_loss, tb_logger, opts)

        
def train_epoch_sl(model, optimizer, lr_scheduler, epoch, train_dataset, val_datasets, problem, tb_logger, opts):
    print("\nStart train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Create data loader with random sampling
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, 
                                  sampler=BatchedRandomSampler(train_dataset, opts.batch_size))

    # Put model in train mode!
    model.train()
    optimizer.zero_grad()
    set_decode_type(model, "greedy")

    for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)):

        train_batch_sl(
            model,
            optimizer,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1
    
    lr_scheduler.step(epoch)

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    for val_idx, val_dataset in enumerate(val_datasets):
        avg_reward, avg_opt_gap = validate(model, val_dataset, problem, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val{}/avg_reward'.format(val_idx+1), avg_reward, step)
            tb_logger.log_value('val{}/opt_gap'.format(val_idx+1), avg_opt_gap, step)
    

def train_batch_sl(model, optimizer, epoch, batch_id, 
                   step, batch, tb_logger, opts):
    # Optionally move Tensors to GPU
    x = move_to(batch['nodes'], opts.device)
    graph = move_to(batch['graph'], opts.device)
    
    if opts.model == 'nar':
        targets = move_to(batch['tour_edges'], opts.device)
        # Compute class weights for NAR decoder
        _targets = batch['tour_edges'].numpy().flatten()
        class_weights = compute_class_weight("balanced", classes=np.unique(_targets), y=_targets)
        class_weights = move_to(torch.FloatTensor(class_weights), opts.device)
    else:
        class_weights = None
        targets = move_to(batch['tour_nodes'], opts.device)
    
    # Evaluate model, get costs and loss
    cost, loss = model(x, graph, supervised=True, targets=targets, class_weights=class_weights)
    
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
        log_values_sl(cost, grad_norms, epoch, batch_id, 
                      step, loss, tb_logger, opts)
