import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from utils.nar_beam_search import Beamsearch
from utils.functions import get_best


class NARModel(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 encoder_class,
                 n_encode_layers,
                 aggregation="sum",
                 normalization="layer",
                 learn_norm=True,
                 track_norm=False,
                 gated=True,
                 n_heads=8,
                 mask_graph=False,
                 *args, **kwargs):
        """
        Models with a GNN/Transformer/MLP encoder and the Non-autoregressive decoder using attention mechanism

        Args:
            problem: TSP/TSPSL, to identify the learning paradigm
            embedding_dim: Hidden dimension for encoder/decoder
            encoder_class: GNN/Transformer/MLP encoder
            n_encode_layers: Number of layers for encoder
            aggregation: Aggregation function for GNN encoder
            normalization: Normalization scheme ('batch'/'layer'/'none')
            learn_norm: Flag for enabling learnt affine transformation during normalization
            track_norm: Flag to enable tracking training dataset stats instead of using batch stats during normalization
            gated: Flag to enbale anisotropic GNN aggregation
            n_heads: Number of attention heads for Transformer encoder/MHA in decoder
            mask_graph: Flag to use graph mask during decoding

        References:
            - A. Nowak, S. Villar, A. S. Bandeira, and J. Bruna. A note on learning algorithms for quadratic assignment with graph neural networks. arXiv preprint arXiv:1706.07450, 2017.
            - C. K. Joshi, T. Laurent, and X. Bresson. An efficient graph convolutional network technique for the travelling salesman problem. arXiv preprint arXiv:1906.01227, 2019.
        """

        assert problem.NAME in ['tsp', 'tspsl'], "NAR Attention Decoder only supports TSP and TSPSL."

        super(NARModel, self).__init__()

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.encoder_class = encoder_class
        self.n_encode_layers = n_encode_layers
        self.aggregation = aggregation
        self.normalization = normalization
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.n_heads = n_heads
        self.mask_graph = mask_graph
        
        # Input embedding layer
        self.init_embed = nn.Linear(2, embedding_dim, bias=True)        
        
        # Encoder model
        self.embedder = self.encoder_class(n_layers=n_encode_layers, 
                                           n_heads=n_heads,
                                           hidden_dim=embedding_dim, 
                                           aggregation=aggregation, 
                                           norm=normalization, 
                                           learn_norm=learn_norm,
                                           track_norm=track_norm,
                                           gated=gated)
        
        # Edge prediction layer
        self.project_node_emb = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.project_graph_emb = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.edge_pred = nn.Linear(embedding_dim, 2, bias=True)

    def set_decode_type(self, decode_type, **kwargs):
        self.decode_type = decode_type

    def forward(self, nodes, graph, supervised=False, targets=None, class_weights=None, return_pi=False):
        """
        Args:
            nodes: Input graph nodes (B x V x 2)
            graph: Graph as **NEGATIVE** adjacency matrices (B x V x V)
            supervised: Toggles SL training, teacher forcing and BCE loss computation
            targets: Targets for teacher forcing and BCE loss
            return_pi: Toggles returning the output sequences 
                       Not compatible with DataParallel as the results 
                       may be of different lengths on different GPUs
        """

        # Supervised learning
        if self.problem.NAME == 'tspsl' and supervised:
            assert targets is not None, "Pass targets during training in supervised mode"
            assert class_weights is not None, "Pass class weights during training in supervised mode for NAR models"
            
            if return_pi:
                # Perform greedy search during training if we want to log cost, pi during training
                logits, _, pi, cost = self.greedy_search(nodes, graph)
                loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')(
                    logits.permute(0, 3, 1, 2), targets)
                return cost, loss, pi

            else:
                # Only perform _inner function for faster training
                logits, _ = self._inner(nodes, graph)
                loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')(
                    logits.permute(0, 3, 1, 2), targets)
                return torch.zeros(nodes.shape[0]), loss

        # Reinforcement learning or inference
        else:
            _, log_p, pi, cost = self.greedy_search(nodes, graph)
            ll = self._calc_log_likelihood(log_p[:, :, :, 1], pi)

            if return_pi:
                return cost, ll, pi
            return cost, ll

    def _init_embed(self, nodes):
        return self.init_embed(nodes)

    def _inner(self, nodes, graph):
        """
        Returns:
            logits: Unnormalized logits over graph edges (B x V x V x 2)
            log_p: Log-Softmax over final dimension of `logits` (B x V x V)
        """
        batch_size, num_nodes, _ = nodes.shape
        
        # Compute node embeddings
        embeddings = self.embedder(self._init_embed(nodes), graph)
        
        # Compute edge embeddings (B x V x V x H)
        Ux = self.project_node_emb(embeddings)
        Gx = self.project_graph_emb(embeddings.mean(dim=1))
        edge_embeddings = F.relu(Ux[:, :, None, :] + Ux[:, None, :, :] + Gx[:, None, None, :])
        
        if self.mask_graph:
            edge_embeddings[graph[:, :, :, None].expand_as(edge_embeddings)] = 0
        
        # Compute logits
        logits = self.edge_pred(edge_embeddings)  # B x V x V x 2
        log_p = F.log_softmax(logits, dim=3)
        
        return logits, log_p

    def _calc_log_likelihood(self, _log_p, a):
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        
        # Calculate log_likelihood
        return log_p.sum(1)

    def beam_search(self, nodes, graph, beam_size):
        """Method to perform graph search (beam search or greedy search)

        Args:
            nodes: Input graph nodes (B x V x 2)
            graph: Graph as **NEGATIVE** adjacency matrices (B x V x V)
            beam_size: Beam search width (=1 to enable greedy search)

        Returns:
            logits, log_p: Outputs of inner function
            pi: Tour sequence for shortest out of `beam_size` candidate tours
            cost: Tour length for shortest out of `beam_size` candidate tours
        """
        batch_size, num_nodes, _ = nodes.shape

        # Compute logits
        logits, log_p = self._inner(nodes, graph)
        
        # Perform beamsearch
        with torch.no_grad():
            _log_p = log_p.clone().detach()[:, :, :, 1]
            _log_p[_log_p == 0] = -1e-10  # Set 0s (i.e. log(1)s) to very small negative number
            
            beamsearch = Beamsearch(beam_size, batch_size, num_nodes, device=_log_p.device, decode_type=self.decode_type)
            trans_probs = _log_p.gather(1, beamsearch.get_current_state())
            for step in range(num_nodes - 1):
                beamsearch.advance(trans_probs)
                trans_probs = _log_p.gather(1, beamsearch.get_current_state())

            # Find TSP tour with highest probability among beam_size candidates
            ends = torch.zeros(batch_size, 1, device=_log_p.device)
            pi = beamsearch.get_hypothesis(ends)
            
            if beam_size == 1:
                # Compute tour costs
                cost, _ = self.problem.get_costs(nodes, pi)

            elif beam_size > 1:
                # Beam search
                sequences = []
                costs = []
                ids = []
                
                # Iterate over all positions in beam
                for pos in range(0, beam_size):
                    ends = pos * torch.ones(batch_size, 1, device=_log_p.device)  # New positions
                    
                    try:
                        pi_temp = beamsearch.get_hypothesis(ends)
                        cost_temp, _ = self.problem.get_costs(nodes, pi_temp)
                        cost_temp, pi_temp = cost_temp.cpu().numpy(), pi_temp.cpu().numpy()
                        
                        sequences.append(pi_temp)
                        costs.append(cost_temp)
                        ids.append(list(range(batch_size)))

                    except AssertionError:
                        # Handles error if the temporary solution is an invalid tour
                        continue
                    
                sequences = np.array(sequences)
                costs = np.array(costs)
                ids = np.array(ids)

                # Reshape/permute sequences, costs, ids
                valid_beam, batch_size, num_nodes = sequences.shape
                sequences = sequences.reshape(valid_beam * batch_size, num_nodes)
                s_idx = []
                for i in range(batch_size):
                    for j in range(valid_beam):
                        s_idx.append(i + j * batch_size)
                
                # Get sequences and costs of shortest tours
                pi, cost = get_best(sequences[s_idx], costs.T.flatten(), ids.T.flatten())

        return logits, log_p, pi, cost

    def greedy_search(self, nodes, graph):
        return self.beam_search(nodes, graph, beam_size=1)

    def sample_many(self, nodes, graph, batch_rep=1, iter_rep=1):
        assert batch_rep == 1 and iter_rep == 1, "Sampling solutions is not supported. Use beam search instead."
        return self.greedy_search(nodes, graph)
