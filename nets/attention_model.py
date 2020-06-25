import math
import numpy as np
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel

from utils.tensor_functions import compute_in_batches
from utils.beam_search import CachedLookup
from utils.functions import sample_many


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 encoder_class,
                 n_encode_layers,
                 aggregation="sum",
                 aggregation_graph="mean",
                 normalization="layer",
                 learn_norm=True,
                 track_norm=False,
                 gated=True,
                 n_heads=8,
                 tanh_clipping=10.0,
                 mask_inner=True,
                 mask_logits=True,
                 mask_graph=False,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 extra_logging=False,
                 *args, **kwargs):
        """
        Models with a GNN/Transformer/MLP encoder and the Autoregressive decoder using attention mechanism

        Args:
            problem: TSP/TSPSL, to identify the learning paradigm
            embedding_dim: Hidden dimension for encoder/decoder
            encoder_class: GNN/Transformer/MLP encoder
            n_encode_layers: Number of layers for encoder
            aggregation: Aggregation function for GNN encoder
            aggregation_graph: Graph aggregation function
            normalization: Normalization scheme ('batch'/'layer'/'none')
            learn_norm: Flag for enabling learnt affine transformation during normalization
            track_norm: Flag to enable tracking training dataset stats instead of using batch stats during normalization
            gated: Flag to enbale anisotropic GNN aggregation
            n_heads: Number of attention heads for Transformer encoder/MHA in decoder
            tanh_clipping: Constant value to clip decoder logits with tanh
            mask_inner: Flag to use visited mask during inner function of decoder
            mask_logits: Flag to use visited mask during log computation of decoder
            mask_graph: Flag to use graph mask during decoding
            checkpoint_encoder: Whether to use checkpoints for encoder embeddings
            shrink_size: N/A
            extra_logging: Flag to perform extra logging, used for plotting histograms of embeddings

        References:
            - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
            - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
        """

        super(AttentionModel, self).__init__()
        
        self.problem = problem
        self.embedding_dim = embedding_dim
        self.encoder_class = encoder_class
        self.n_encode_layers = n_encode_layers
        self.aggregation = aggregation
        self.aggregation_graph = aggregation_graph
        self.normalization = normalization
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.mask_graph = mask_graph
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        
        # Extra logging updates self variables with batch statistics (without returning them)
        self.extra_logging = extra_logging
        
        self.decode_type = None
        self.temp = 1.0
        
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'

        # Problem specific context parameters (placeholder and step context dimension)
        # Un-used, as we only tackle TSP
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity/remaining length/remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand/prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  
                # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        
        else:  # TSP
            assert problem.NAME in ("tsp", "tspsl"), "Unsupported problem: {}".format(problem.NAME)

            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y

            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        
        # Input embedding layer
        self.init_embed = nn.Linear(node_dim, embedding_dim, bias=True)        
        
        # Encoder model
        self.embedder = self.encoder_class(n_layers=n_encode_layers, 
                                           n_heads=n_heads,
                                           hidden_dim=embedding_dim, 
                                           aggregation=aggregation, 
                                           norm=normalization, 
                                           learn_norm=learn_norm,
                                           track_norm=track_norm,
                                           gated=gated)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, nodes, graph, supervised=False, targets=None, class_weights=None, return_pi=False):
        """
        Args:
            nodes: Input graph nodes (B x V x 2)
            graph: Graph as **NEGATIVE** adjacency matrices (B x V x V)
            supervised: Toggles SL training, teacher forcing and NLL loss computation
            targets: Targets for teacher forcing and NLL loss
            return_pi: Toggles returning the output sequences 
                       (Not compatible with DataParallel as the results
                        may be of different lengths on different GPUs)
        """
        # Embed input batch of graph using GNN (B x V x H)
        if self.checkpoint_encoder:
            embeddings = checkpoint(self.embedder, self._init_embed(nodes), graph)
        else:
            embeddings = self.embedder(self._init_embed(nodes), graph)
        
        if self.extra_logging:
            self.embeddings_batch = embeddings

        # Supervised learning
        if self.problem.NAME == 'tspsl' and supervised:
            assert targets is not None, "Pass targets during training in supervised mode"
            
            # Run inner function
            _log_p, pi = self._inner(nodes, graph, embeddings, supervised=supervised, targets=targets)
            
            if self.extra_logging:
                self.log_p_batch = _log_p
                self.log_p_sel_batch = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)
            
            # Get predicted costs
            cost, mask = self.problem.get_costs(nodes, pi)
            
            # Compute NLL loss
            logits = _log_p.permute(0, 2, 1)  # B x V x output_vocab -> B x output_vocab x V
            # Set -inf values to -1000 for handling NLL loss
            logits[logits == -float(np.inf)] = -1000  
            loss = nn.NLLLoss(reduction='mean')(logits, targets)
            
            if return_pi:
                return cost, loss, pi 
            return cost, loss
        
        # Reinforcement learning or inference
        else:
            # Run inner function
            _log_p, pi = self._inner(nodes, graph, embeddings)
            
            if self.extra_logging:
                self.log_p_batch = _log_p
                self.log_p_sel_batch = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)
            
            # Get predicted costs
            cost, mask = self.problem.get_costs(nodes, pi)
            
            # Log likelihood is calculated within the model since 
            # returning it per action does not work well with DataParallel 
            # (since sequences can be of different lengths)
            ll = self._calc_log_likelihood(_log_p, pi, mask)
            
            if return_pi:
                return cost, ll, pi
            return cost, ll

    def beam_search(self, *args, **kwargs):
        """Helper method to call beam search
        """
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, nodes, graph):
        embeddings = self.embedder(self._init_embed(nodes), graph)
        # Use a CachedLookup such that if we repeatedly index this object with 
        # the same index, we only need to do the lookup once... 
        # this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10

        # Parent is row idx of ind_topk,
        # can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, nodes):
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(nodes['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        nodes['loc'],
                        *(nodes[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        
        return self.init_embed(nodes)

    def _inner(self, nodes, graph, embeddings, supervised=False, targets=None):
        outputs = []
        sequences = []
        
        # Create problem state for masking (tracks which nodes have been visited)
        state = self.problem.make_state(nodes, graph)

        # Compute keys, values for the glimpse and keys for the logits for reuse
        fixed = self._precompute(embeddings)

        batch_size, num_nodes, _ = nodes.shape

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            # Get log probabilities of next action
            log_p, mask = self._get_log_p(fixed, state)
            
            # Select the indices of the next nodes in the sequences
            if self.problem.NAME == 'tspsl' and supervised:
                # Teacher-forcing during training in supervised mode
                t_idx = torch.LongTensor([i]).to(nodes.device)
                selected = targets.index_select(dim=-1, index=t_idx).view(batch_size)
            
            else:
                selected = self._select_node(
                    log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            
            # Update problem state
            state = state.update(selected)

            # Make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, graph, batch_rep=1, iter_rep=1):
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        # Params: inner_func, get_cost_func, input, batch_rep, iter_rep
        return sample_many(
            # Need to unpack tuple into arguments
            lambda input: self._inner(*input),
            # Don't need embeddings as input to get_costs
            lambda input, pi: self.problem.get_costs(input[0], pi),  
            # Pack input with embeddings (additional input)
            (input, graph, self.embedder(self._init_embed(input), graph)),
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any NaNs"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        
        return selected

    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        if self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        elif self.aggregation_graph == "mean":
            graph_embed = embeddings.mean(1)
        else:  # Default: dissable graph embedding
            graph_embed = embeddings.sum(1) * 0.0
        
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, unlike torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(
                log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        
        # Compute the mask, for masking next action based on previous actions
        mask = state.get_mask()
        
        graph_mask = None
        if self.mask_graph:
            # Compute the graph mask, for masking next action based on graph structure 
            graph_mask = state.get_graph_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once 
        (for efficient evaluation of the model)
        """
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                    ),
                    -1
                )
        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP
        
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions to (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -1e10
            if self.mask_graph:
                compatibility[graph_mask[None, :, :, None, :].expand_as(compatibility)] = -1e10

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        
        # From the logits compute the probabilities by masking the graph, clipping, and masking visited
        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -1e10 
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -1e10

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        if self.is_vrp and self.allow_partial:
            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
